"""
evaluation_validator.py
────────────────────────
Hybrid LLM + rule-based evaluation validation layer.

Architecture position:
    LLM evaluation output → EvaluationValidator.validate() → validated result / deferred

Three tiers of response:

    Tier 1 — Hard reject + re-evaluate (Rules 1, 2, 3)
        Evaluation is structurally wrong. Re-call LLM with failure described.
        Max 2 re-evaluation attempts.

    Tier 2 — Soft flag + accept (Rules 4, 5)
        Evaluation looks unusual but may be correct.
        Accept with validation_flag field set. Surface in expert review.

    Tier 3 — Accept with note (everything else)
        Minor anomaly logged. Accepted as-is.

Deferred scoring:
    When both re-evaluations fail validation, the turn is deferred.
    Score = None. interview continues. Mandatory expert review before report.

Integration:
    from evaluation_validator import EvaluationValidator, ValidationResult
    validator = EvaluationValidator(session)
    result = validator.validate(raw_eval, question, answer, difficulty, question_type)
    # result.eval is the validated (or fallback) evaluation dict
    # result.status tells you what happened
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────

MAX_REEVAL_ATTEMPTS = 2

# Valid field values — any value outside these is a hard reject
VALID_QUALITY     = {"strong","adequate","weak","honest_admission","poor_articulation"}
VALID_ACCURACY    = {"correct","partial","wrong","not_applicable"}
VALID_CONFIDENCE  = {"high","medium","low"}
VALID_QUADRANT    = {"genuine_expert","genuine_nervous","dangerous_fake","honest_confused"}

# Quality → score range rules (Rule 2)
QUALITY_SCORE_RULES: dict[str, tuple[int, int]] = {
    "strong":            (7, 10),
    "adequate":          (4, 8),
    "weak":              (0, 5),
    "honest_admission":  (4, 8),   # honesty is valued — never 0
    "poor_articulation": (3, 7),
}

# Quadrant → required accuracy values (Rule 3)
QUADRANT_ACCURACY_RULES: dict[str, set[str]] = {
    "genuine_expert":  {"correct", "partial"},
    "genuine_nervous": {"correct", "partial", "wrong"},   # can be nervous AND wrong
    "dangerous_fake":  {"wrong", "partial"},
    "honest_confused": {"wrong", "partial", "not_applicable"},
}

# Level gap plausibility bounds (Rule 4)
# (level_gap, quality) combinations that are contradictory
LEVEL_GAP_CONTRADICTIONS: list[tuple[int, str, str]] = [
    # (level_gap_threshold, direction, quality)
    # level_gap > +2 AND quality is weak → contradiction
    # level_gap < -2 AND quality is strong → contradiction
]


# ── Enumerations ──────────────────────────────────────────────────────────

class ValidationStatus(str, Enum):
    ACCEPTED         = "accepted"          # passed all rules
    ACCEPTED_FLAGGED = "accepted_flagged"  # passed with soft flag — needs human review
    REEVAL_ACCEPTED  = "reeval_accepted"   # failed, re-evaluated, accepted
    DEFERRED         = "deferred"          # all attempts failed — score is None
    RULE_BASED       = "rule_based"        # emergency fallback — score from rules only


class FailedRule(str, Enum):
    R1_RANGE         = "R1_score_out_of_range"
    R2_QUALITY_SCORE = "R2_quality_score_mismatch"
    R3_QUADRANT_ACC  = "R3_quadrant_accuracy_contradiction"
    R4_LEVEL_GAP     = "R4_level_gap_implausible"
    R5_MISSING_SCORE = "R5_missing_points_vs_score"
    R6_SHORT_HIGH    = "R6_very_short_answer_high_score"


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class RuleViolation:
    rule:    FailedRule
    tier:    int          # 1 = hard reject, 2 = soft flag
    message: str
    field:   str          # which field triggered this


@dataclass
class ValidationResult:
    """
    Output of EvaluationValidator.validate().

    eval:              The evaluation dict to use (original, re-evaluated, or rule-based)
    status:            What happened during validation
    violations:        List of all rule violations found
    attempts:          How many LLM evaluation attempts were made
    validation_flag:   Set when Tier 2 violations exist — surfaces in expert review
    deferred:          True when scoring was deferred (score is None)
    expert_priority:   "high" / "normal" / "none" — for expert review queue
    """
    eval:              Optional[dict]
    status:            ValidationStatus
    violations:        list[RuleViolation]  = field(default_factory=list)
    attempts:          int                  = 1
    validation_flag:   Optional[str]        = None
    deferred:          bool                 = False
    expert_priority:   str                  = "none"
    log:               list[str]            = field(default_factory=list)


# ── Main class ─────────────────────────────────────────────────────────────

class EvaluationValidator:
    """
    Validates LLM evaluation output against 6 rules.
    Re-evaluates on hard failures. Defers on persistent failure.

    Usage:
        validator = EvaluationValidator(session)
        result = validator.validate(
            raw_eval,
            question,
            answer,
            difficulty,
            question_type,
            reeval_fn,        # callable(prompt) → dict
        )
        # use result.eval as the final evaluation
        # check result.deferred before using score
    """

    def __init__(self, session: dict):
        self.session = session
        self.level   = session.get("resume", {}).get("level", "trained_fresher")
        self.domain  = session.get("resume", {}).get("domain", "physical_design")

        if "validation_log" not in session:
            session["validation_log"] = []
        if "deferred_turns" not in session:
            session["deferred_turns"] = []

    # ── Public interface ─────────────────────────────────────────────────

    def validate(self,
                 raw_eval:       dict,
                 question:       str,
                 answer:         str,
                 difficulty:     str,
                 question_type:  str,
                 reeval_fn=None) -> ValidationResult:
        """
        Main entry point. Validates raw_eval against all rules.

        Args:
            raw_eval:      Raw dict returned by LLM evaluation
            question:      The question that was asked
            answer:        The candidate's answer
            difficulty:    Difficulty string (foundational/basic/intermediate/advanced/expert)
            question_type: Question type string
            reeval_fn:     Optional callable(failure_description: str) → dict
                           Called for Tier 1 failures. If None, defers immediately.
        """
        log: list[str] = []

        # Step 1: Validate structure (missing required fields)
        if not raw_eval or not isinstance(raw_eval, dict):
            return self._handle_deferred(
                "LLM returned empty or non-dict evaluation", log, question, answer
            )

        # Step 2: Run all rules
        violations = self._run_all_rules(raw_eval, question, answer, difficulty, log)

        # Separate by tier
        tier1 = [v for v in violations if v.tier == 1]
        tier2 = [v for v in violations if v.tier == 2]

        # No violations
        if not violations:
            log.append("All 6 rules passed — evaluation accepted")
            self._record(raw_eval, ValidationStatus.ACCEPTED, violations, 1, log)
            return ValidationResult(
                eval=raw_eval,
                status=ValidationStatus.ACCEPTED,
                violations=[],
                attempts=1,
                log=log,
            )

        # Only Tier 2 violations — accept with flag
        if not tier1:
            flag = "; ".join(v.rule.value for v in tier2)
            log.append(f"Tier 2 violations — accepted with flag: {flag}")
            self._record(raw_eval, ValidationStatus.ACCEPTED_FLAGGED, violations, 1, log)
            return ValidationResult(
                eval=raw_eval,
                status=ValidationStatus.ACCEPTED_FLAGGED,
                violations=violations,
                attempts=1,
                validation_flag=flag,
                expert_priority="normal",
                log=log,
            )

        # Tier 1 violations — attempt re-evaluation
        if reeval_fn is None:
            return self._handle_deferred(
                f"Tier 1 violations: {[v.rule.value for v in tier1]}",
                log, question, answer
            )

        failure_desc = self._describe_failures(tier1, raw_eval)
        log.append(f"Tier 1 violations found: {[v.rule.value for v in tier1]}")
        log.append(f"Requesting re-evaluation: {failure_desc[:120]}")

        for attempt in range(2, MAX_REEVAL_ATTEMPTS + 2):
            try:
                reeval_result = reeval_fn(failure_desc)
            except Exception as e:
                log.append(f"Re-evaluation attempt {attempt} failed: {e}")
                continue

            if not reeval_result:
                log.append(f"Re-evaluation attempt {attempt} returned empty")
                continue

            # Validate the re-evaluation
            new_violations = self._run_all_rules(
                reeval_result, question, answer, difficulty, []
            )
            new_tier1 = [v for v in new_violations if v.tier == 1]

            if not new_tier1:
                # Re-evaluation passed
                new_tier2 = [v for v in new_violations if v.tier == 2]
                flag = "; ".join(v.rule.value for v in new_tier2) if new_tier2 else None
                status = (ValidationStatus.REEVAL_ACCEPTED if not new_tier2
                          else ValidationStatus.ACCEPTED_FLAGGED)
                log.append(f"Re-evaluation attempt {attempt} passed — status={status.value}")
                self._record(reeval_result, status, new_violations, attempt, log)
                return ValidationResult(
                    eval=reeval_result,
                    status=status,
                    violations=new_violations,
                    attempts=attempt,
                    validation_flag=flag,
                    expert_priority="normal" if flag else "none",
                    log=log,
                )

            # Re-evaluation also failed
            log.append(
                f"Re-evaluation attempt {attempt} still has Tier 1 violations: "
                f"{[v.rule.value for v in new_tier1]}"
            )
            # Update failure description for next attempt
            failure_desc = self._describe_failures(new_tier1, reeval_result)

        # All re-evaluations failed — force accept with high-priority flag
        log.append(
            "All re-evaluation attempts failed — force accepting with validation_flag=force_accepted"
        )
        flag = "force_accepted; " + "; ".join(v.rule.value for v in tier1)
        self._record(raw_eval, ValidationStatus.ACCEPTED_FLAGGED, violations,
                     MAX_REEVAL_ATTEMPTS + 1, log)
        return ValidationResult(
            eval=raw_eval,
            status=ValidationStatus.ACCEPTED_FLAGGED,
            violations=violations,
            attempts=MAX_REEVAL_ATTEMPTS + 1,
            validation_flag=flag,
            expert_priority="high",
            log=log,
        )

    def get_validation_stats(self) -> dict:
        """Returns validation statistics for observability."""
        log = self.session.get("validation_log", [])
        deferred = self.session.get("deferred_turns", [])
        total = len(log)
        return {
            "total_evaluations":        total,
            "accepted":                 sum(1 for e in log if e["status"] == "accepted"),
            "accepted_flagged":         sum(1 for e in log if e["status"] == "accepted_flagged"),
            "reeval_accepted":          sum(1 for e in log if e["status"] == "reeval_accepted"),
            "deferred":                 len(deferred),
            "deferral_rate":            round(len(deferred) / max(total, 1), 3),
            "validation_rejection_rate": round(
                sum(1 for e in log if e.get("tier1_violations", 0) > 0) / max(total, 1), 3
            ),
            "expert_review_high":       sum(1 for e in log if e.get("expert_priority") == "high"),
            "most_common_violation":    self._most_common_violation(log),
        }

    # ── Rule engine ───────────────────────────────────────────────────────

    def _run_all_rules(self, eval_dict: dict, question: str,
                        answer: str, difficulty: str,
                        log: list[str]) -> list[RuleViolation]:
        """Runs all 6 rules. Returns list of all violations found."""
        violations: list[RuleViolation] = []

        v = self._rule1_range(eval_dict, log)
        if v: violations.append(v)

        v = self._rule2_quality_score(eval_dict, log)
        if v: violations.append(v)

        v = self._rule3_quadrant_accuracy(eval_dict, log)
        if v: violations.append(v)

        v = self._rule4_level_gap(eval_dict, log)
        if v: violations.append(v)

        v = self._rule5_missing_vs_score(eval_dict, log)
        if v: violations.append(v)

        v = self._rule6_short_answer_high_score(eval_dict, answer, log)
        if v: violations.append(v)

        return violations

    # ── Rule 1: Range check ───────────────────────────────────────────────

    def _rule1_range(self, ev: dict, log: list[str]) -> Optional[RuleViolation]:
        """Score must be 0–10. level_gap must be -3 to +3."""
        score = ev.get("score")
        level_gap = ev.get("level_gap", 0)

        # Missing score
        if score is None:
            log.append("Rule 1 FAIL — score field missing")
            return RuleViolation(
                rule=FailedRule.R1_RANGE, tier=1,
                message="score field is missing from evaluation",
                field="score",
            )

        # Convert to number
        try:
            score = float(score)
        except (TypeError, ValueError):
            log.append(f"Rule 1 FAIL — score not numeric: {score!r}")
            return RuleViolation(
                rule=FailedRule.R1_RANGE, tier=1,
                message=f"score is not a number: {score!r}",
                field="score",
            )

        if not (0 <= score <= 10):
            log.append(f"Rule 1 FAIL — score {score} out of [0, 10]")
            return RuleViolation(
                rule=FailedRule.R1_RANGE, tier=1,
                message=f"score={score} is outside valid range [0, 10]",
                field="score",
            )

        # level_gap range
        try:
            level_gap = float(level_gap)
        except (TypeError, ValueError):
            level_gap = 0

        if not (-3 <= level_gap <= 3):
            log.append(f"Rule 1 FAIL — level_gap {level_gap} out of [-3, 3]")
            return RuleViolation(
                rule=FailedRule.R1_RANGE, tier=1,
                message=f"level_gap={level_gap} is outside valid range [-3, 3]",
                field="level_gap",
            )

        # Validate enum fields
        for field_name, valid_set in [
            ("quality",          VALID_QUALITY),
            ("accuracy",         VALID_ACCURACY),
            ("confidence_level", VALID_CONFIDENCE),
            ("quadrant",         VALID_QUADRANT),
        ]:
            val = ev.get(field_name)
            if val not in valid_set:
                log.append(f"Rule 1 FAIL — {field_name}={val!r} not in {valid_set}")
                return RuleViolation(
                    rule=FailedRule.R1_RANGE, tier=1,
                    message=f"{field_name}={val!r} is not a valid value",
                    field=field_name,
                )

        log.append("Rule 1 pass — all ranges valid")
        return None

    # ── Rule 2: Quality–score consistency ─────────────────────────────────

    def _rule2_quality_score(self, ev: dict, log: list[str]) -> Optional[RuleViolation]:
        """
        Quality label and score must be consistent.
        strong → 7–10, weak → 0–5, etc.
        """
        quality = ev.get("quality", "")
        score   = ev.get("score")
        if quality not in VALID_QUALITY or score is None:
            return None  # Rule 1 will catch this

        try:
            score = float(score)
        except (TypeError, ValueError):
            return None

        bounds = QUALITY_SCORE_RULES.get(quality)
        if bounds is None:
            return None

        lo, hi = bounds
        if not (lo <= score <= hi):
            log.append(
                f"Rule 2 FAIL — quality={quality!r} requires score in [{lo},{hi}] "
                f"but got {score}"
            )
            return RuleViolation(
                rule=FailedRule.R2_QUALITY_SCORE, tier=1,
                message=(
                    f"quality={quality!r} implies score in [{lo},{hi}] "
                    f"but score={score}"
                ),
                field="quality/score",
            )

        log.append(f"Rule 2 pass — quality={quality!r}, score={score} consistent")
        return None

    # ── Rule 3: Quadrant–accuracy consistency ─────────────────────────────

    def _rule3_quadrant_accuracy(self, ev: dict,
                                  log: list[str]) -> Optional[RuleViolation]:
        """
        Quadrant and accuracy must be logically consistent.
        dangerous_fake → accuracy must be wrong/partial.
        genuine_expert → accuracy must be correct/partial.
        """
        quadrant = ev.get("quadrant", "")
        accuracy = ev.get("accuracy", "")

        if quadrant not in VALID_QUADRANT or accuracy not in VALID_ACCURACY:
            return None  # Rule 1 handles this

        allowed = QUADRANT_ACCURACY_RULES.get(quadrant, set())
        if accuracy not in allowed:
            log.append(
                f"Rule 3 FAIL — quadrant={quadrant!r} requires accuracy in "
                f"{allowed} but got {accuracy!r}"
            )
            return RuleViolation(
                rule=FailedRule.R3_QUADRANT_ACC, tier=1,
                message=(
                    f"quadrant={quadrant!r} is incompatible with accuracy={accuracy!r}. "
                    f"Allowed: {allowed}"
                ),
                field="quadrant/accuracy",
            )

        log.append(f"Rule 3 pass — quadrant={quadrant!r}, accuracy={accuracy!r} consistent")
        return None

    # ── Rule 4: Level gap plausibility ────────────────────────────────────

    def _rule4_level_gap(self, ev: dict, log: list[str]) -> Optional[RuleViolation]:
        """
        level_gap and quality must not contradict each other.
        level_gap > +2 and quality weak → contradiction.
        level_gap < -2 and quality strong → contradiction.
        """
        try:
            level_gap = float(ev.get("level_gap", 0))
        except (TypeError, ValueError):
            level_gap = 0

        quality = ev.get("quality", "")

        if level_gap > 2 and quality == "weak":
            log.append(
                f"Rule 4 FAIL — level_gap={level_gap} > 2 but quality=weak — contradiction"
            )
            return RuleViolation(
                rule=FailedRule.R4_LEVEL_GAP, tier=1,
                message=(
                    f"level_gap={level_gap} indicates above-level performance "
                    f"but quality=weak is contradictory"
                ),
                field="level_gap/quality",
            )

        if level_gap < -2 and quality == "strong":
            log.append(
                f"Rule 4 FAIL — level_gap={level_gap} < -2 but quality=strong — contradiction"
            )
            return RuleViolation(
                rule=FailedRule.R4_LEVEL_GAP, tier=1,
                message=(
                    f"level_gap={level_gap} indicates below-level performance "
                    f"but quality=strong is contradictory"
                ),
                field="level_gap/quality",
            )

        log.append(f"Rule 4 pass — level_gap={level_gap}, quality={quality!r} consistent")
        return None

    # ── Rule 5: Missing points vs score ───────────────────────────────────

    def _rule5_missing_vs_score(self, ev: dict,
                                 log: list[str]) -> Optional[RuleViolation]:
        """
        Tier 2 soft flag only.
        4+ missing points AND score > 8 is suspicious.
        Flag for human review but do not reject.
        """
        missing = ev.get("missing_points", [])
        score   = ev.get("score")

        if not isinstance(missing, list) or score is None:
            return None

        try:
            score = float(score)
        except (TypeError, ValueError):
            return None

        if len(missing) >= 4 and score > 8:
            log.append(
                f"Rule 5 FLAG — {len(missing)} missing points but score={score} > 8 "
                f"(Tier 2 soft flag)"
            )
            return RuleViolation(
                rule=FailedRule.R5_MISSING_SCORE, tier=2,
                message=(
                    f"{len(missing)} missing points identified but score={score} > 8. "
                    "Human review recommended."
                ),
                field="missing_points/score",
            )

        log.append(f"Rule 5 pass — {len(missing)} missing points, score={score}")
        return None

    # ── Rule 6: Very short answer + high score ────────────────────────────

    def _rule6_short_answer_high_score(self, ev: dict, answer: str,
                                        log: list[str]) -> Optional[RuleViolation]:
        """
        Tier 2 soft flag only.
        Answer under 15 words AND score >= 9 is suspicious.
        Could be a very concise expert answer OR a hallucination.
        Flag for review.
        """
        score       = ev.get("score")
        word_count  = len(answer.strip().split()) if answer else 0

        if score is None:
            return None

        try:
            score = float(score)
        except (TypeError, ValueError):
            return None

        if word_count < 15 and score >= 9:
            log.append(
                f"Rule 6 FLAG — answer only {word_count} words but score={score} >= 9 "
                f"(Tier 2 soft flag)"
            )
            return RuleViolation(
                rule=FailedRule.R6_SHORT_HIGH, tier=2,
                message=(
                    f"Very short answer ({word_count} words) scored {score}/10. "
                    "Could be valid expert brevity or LLM error. Flag for review."
                ),
                field="answer_length/score",
            )

        log.append(f"Rule 6 pass — {word_count} words, score={score}")
        return None

    # ── Deferred scoring ─────────────────────────────────────────────────

    def _handle_deferred(self, reason: str, log: list[str],
                          question: str, answer: str) -> ValidationResult:
        """
        Marks this turn as deferred. Score is None.
        Turn is added to deferred_turns list for expert review.
        """
        turn = self.session.get("turn", 0)
        log.append(f"DEFERRED — {reason}")

        self.session["deferred_turns"].append({
            "turn":     turn,
            "reason":   reason,
            "question": question[:150],
            "answer":   (answer or "")[:150],
            "ts":       time.time(),
        })

        self._record(None, ValidationStatus.DEFERRED, [], 0, log)

        return ValidationResult(
            eval=None,
            status=ValidationStatus.DEFERRED,
            deferred=True,
            expert_priority="high",
            log=log,
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    def _describe_failures(self, violations: list[RuleViolation],
                            eval_dict: dict) -> str:
        """Builds a re-evaluation prompt description from violations."""
        parts = []
        for v in violations:
            parts.append(
                f"Rule {v.rule.value}: {v.message} "
                f"(field: {v.field}, current value: {eval_dict.get(v.field.split('/')[0], '?')!r})"
            )
        return (
            "Your previous evaluation had the following consistency errors. "
            "Re-evaluate and fix all of them:\n" + "\n".join(f"- {p}" for p in parts)
        )

    def _record(self, eval_dict: Optional[dict], status: ValidationStatus,
                violations: list[RuleViolation], attempts: int,
                log: list[str]) -> None:
        self.session["validation_log"].append({
            "ts":                time.time(),
            "turn":              self.session.get("turn", 0),
            "status":            status.value,
            "tier1_violations":  sum(1 for v in violations if v.tier == 1),
            "tier2_violations":  sum(1 for v in violations if v.tier == 2),
            "attempts":          attempts,
            "expert_priority":   "high" if status == ValidationStatus.DEFERRED else "none",
            "violation_rules":   [v.rule.value for v in violations],
        })

    @staticmethod
    def _most_common_violation(log: list[dict]) -> Optional[str]:
        from collections import Counter
        all_rules = [r for e in log for r in e.get("violation_rules", [])]
        if not all_rules:
            return None
        return Counter(all_rules).most_common(1)[0][0]


# ── Session integration ────────────────────────────────────────────────────

def validate_evaluation(session: dict,
                         raw_eval: dict,
                         question: str,
                         answer: str,
                         difficulty: str,
                         question_type: str,
                         reeval_fn=None) -> ValidationResult:
    """
    Single entry point for the rest of the codebase.
    Replaces the inline 'if result and "level_gap" not in result' check
    in evaluate_answer_llm().

    Args:
        session:       Current interview session dict
        raw_eval:      Raw dict from LLM evaluation call
        question:      Question that was asked
        answer:        Candidate's answer
        difficulty:    Difficulty string
        question_type: Question type string
        reeval_fn:     Optional callable for re-evaluation on Tier 1 failures

    Returns:
        ValidationResult — use result.eval for the final evaluation dict
    """
    validator = EvaluationValidator(session)
    return validator.validate(
        raw_eval, question, answer, difficulty, question_type, reeval_fn
    )


def should_defer_report(session: dict) -> bool:
    """
    Returns True if the session has enough deferred turns to
    require expert review before report generation.
    Threshold: 3 or more deferred turns in a session.
    """
    deferred = session.get("deferred_turns", [])
    return len(deferred) >= 3


def get_deferred_summary(session: dict) -> dict:
    """Returns deferred turn data for expert review queue."""
    deferred = session.get("deferred_turns", [])
    return {
        "count":           len(deferred),
        "requires_review": len(deferred) >= 3,
        "turns":           deferred,
    }