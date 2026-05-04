"""
repetition_guard.py
────────────────────
Three-layer repetition prevention system.

Architecture position:
    StrategyEngine.decide() → QuestionIntent → RepetitionGuard.check() → allow/block → LLM

Layer 1 — Exact hash check (free, Python only)
    Hashes every asked question. Blocks verbatim repeats instantly.
    Cost: zero. Speed: microseconds.

Layer 2 — Semantic embedding similarity (OpenAI embeddings)
    Embeds generated question, compares cosine similarity to all previous.
    Differentiates valid follow-up from paraphrase repetition using strategy reason code.
    Cost: ~$0.000001 per check. Speed: 150–300ms.
    Falls back to Layer 1 + 3 only if embedding service unavailable.

Layer 3 — Concept node deduplication (Python only)
    Checks concept node ID against session's asked_concept_nodes set.
    Allows same concept node only if strategy explicitly requested it (follow-up).
    Cost: zero. Speed: microseconds.

Fallback chain when generation blocked:
    Attempt 1 blocked → regenerate with explicit diversity instruction
    Attempt 2 blocked → switch topic, use second-highest priority
    Attempt 3 blocked → use curated fallback question bank
    All fallbacks logged as events in observability.

Integration:
    from repetition_guard import RepetitionGuard, CheckResult
    guard = RepetitionGuard(session)
    result = guard.check(generated_question, intent)
    if result.allowed:
        ask_question(generated_question)
    else:
        # use result.reason and result.layer to decide next action
"""

from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────

# Cosine similarity thresholds
SIMILARITY_HARD_BLOCK   = 0.92   # always block — same question rephrased
SIMILARITY_SOFT_REVIEW  = 0.75   # block unless strategy explicitly triggered follow-up
SIMILARITY_ALLOW        = 0.74   # always allow — genuinely different question

# Question types that explicitly allow revisiting the same concept
VALID_FOLLOWUP_TYPES = {
    "contradiction",      # angle_2 is designed to revisit angle_1 concept
    "recovery_probe",     # designed to retry a concept the candidate failed
    "why_probe",          # explicitly probes deeper on same concept
    "numerical",          # may probe same concept at numerical depth
    "practical_example",  # same concept, different framing
}

# Curated fallback question bank — guaranteed diverse, expert-written
# Format: domain → difficulty → list of question strings
# These are NEVER generated — written once and locked.
FALLBACK_QUESTIONS: dict[str, dict[str, list[str]]] = {
    "physical_design": {
        "foundational": [
            "Can you explain what a netlist is and its role in physical design?",
            "What is the difference between synthesis and place-and-route?",
            "What does utilization mean in the context of chip design?",
        ],
        "basic": [
            "Walk me through the main stages of a standard physical design flow.",
            "What is clock skew and why does it matter during CTS?",
            "What is the purpose of a power ring in chip layout?",
        ],
        "intermediate": [
            "You are seeing timing violations after routing that were not present after placement. What are the first three things you investigate?",
            "Explain the difference between setup and hold violations and how the fix for each differs.",
            "What is IR drop and how does it affect circuit timing?",
        ],
        "advanced": [
            "Describe your approach to closing timing on a path with simultaneous setup and hold violations.",
            "How do you handle routing congestion when it is blocking a critical net?",
            "What is OCV and how does derating affect your signoff margins?",
        ],
        "expert": [
            "Walk me through how you would set up a multi-corner multi-mode timing analysis flow for signoff.",
            "Explain the trade-offs between useful skew and traditional zero-skew CTS.",
            "How do you balance power, performance, and area trade-offs during floorplanning for a complex SoC?",
        ],
    },
    "analog_layout": {
        "foundational": [
            "What are the main layers used in analog layout and what does each represent?",
            "What is LVS and what does it verify?",
            "Why is device matching important in analog circuits?",
        ],
        "basic": [
            "What techniques do you use to achieve good device matching in layout?",
            "How do you minimize parasitic capacitance in a layout?",
            "What is latch-up and what causes it in CMOS?",
        ],
        "intermediate": [
            "A matched differential pair shows 2% mismatch after fabrication. What layout issues would you check first?",
            "Explain the purpose of guard rings and when you would use them.",
            "How do parasitics extracted from layout differ from schematic parasitics and how do you account for this?",
        ],
        "advanced": [
            "How would you lay out a high-speed comparator to minimize input-referred offset?",
            "Walk me through your approach to EM-safe metal sizing for a high-current power path.",
            "How do process gradients affect matched devices and how do you design around them?",
        ],
        "expert": [
            "Describe your approach to full-custom layout of a 10-bit DAC requiring better than 0.5 LSB INL.",
            "How do you verify shielding effectiveness for a sensitive analog net in a mixed-signal design?",
            "Walk me through Pelgrom's model and how you use it to set area targets for matched devices.",
        ],
    },
    "design_verification": {
        "foundational": [
            "What is the difference between simulation and formal verification?",
            "What does functional coverage measure and why is it important?",
            "What is a testbench and what are its main components?",
        ],
        "basic": [
            "What is the difference between an immediate assertion and a concurrent assertion in SVA?",
            "What is a covergroup and how is it different from code coverage?",
            "Walk me through a basic UVM agent structure.",
        ],
        "intermediate": [
            "Your simulation shows 95% code coverage but you still have functional holes. How do you identify them?",
            "An assertion fires unexpectedly. Walk me through how you debug it.",
            "How does a UVM scoreboard work and what makes a good scoreboard design?",
        ],
        "advanced": [
            "How do you decide between simulation and formal verification for a given design block?",
            "Describe how you would build coverage closure for a complex protocol like AXI.",
            "What is vacuous passing in formal verification and how do you detect and handle it?",
        ],
        "expert": [
            "Walk me through how you would verify a complex cache coherence protocol end-to-end.",
            "How do you manage verification reuse across a product family with multiple chip configurations?",
            "Describe how you would approach full-chip verification sign-off for a mixed-signal SoC.",
        ],
    },
}

MAX_REGENERATION_ATTEMPTS = 3


# ── Data structures ────────────────────────────────────────────────────────

class BlockReason(str, Enum):
    EXACT_DUPLICATE       = "exact_duplicate"
    SEMANTIC_DUPLICATE    = "semantic_duplicate"
    CONCEPT_DUPLICATE     = "concept_duplicate"
    EMBEDDING_UNAVAILABLE = "embedding_unavailable_degraded"


@dataclass
class CheckResult:
    """
    Result of a RepetitionGuard check.

    allowed:          True = question can be asked
    layer:            Which layer made the decision (1, 2, or 3)
    similarity_score: Cosine similarity if Layer 2 was run, else None
    block_reason:     Populated when allowed=False
    most_similar_q:   The previous question most similar to this one
    is_valid_followup: True when blocked by Layer 2 but overridden by strategy
    fallback_used:    True when curated fallback bank was used
    """
    allowed:           bool
    layer:             int                  # 1, 2, or 3
    similarity_score:  Optional[float]      = None
    block_reason:      Optional[BlockReason] = None
    most_similar_q:    Optional[str]        = None
    is_valid_followup: bool                 = False
    fallback_used:     bool                 = False
    log:               list[str]            = field(default_factory=list)


@dataclass
class RegenerationResult:
    """
    Result after all regeneration attempts are exhausted.
    Contains either the approved question or the curated fallback.
    """
    question:          str
    fallback_used:     bool
    attempts_made:     int
    final_check:       CheckResult


# ── Main class ─────────────────────────────────────────────────────────────

class RepetitionGuard:
    """
    Three-layer repetition prevention for the VLSI interview platform.

    Usage (per question generation cycle):

        guard = RepetitionGuard(session)

        # After LLM generates a question:
        result = guard.check(generated_question, intent)

        if result.allowed:
            # proceed — question is safe to ask
            guard.record(generated_question, intent)
        else:
            # regenerate or use fallback
    """

    def __init__(self, session: dict):
        self.session      = session
        self.domain       = session.get("resume", {}).get("domain", "physical_design")
        self.level        = session.get("resume", {}).get("level", "trained_fresher")
        self._embeddings_available = True   # assume available, set False on failure

        # Ensure storage exists in session
        if "asked_question_hashes" not in session:
            session["asked_question_hashes"] = set()
        if "asked_question_embeddings" not in session:
            session["asked_question_embeddings"] = []   # list of (text, vector)
        if "repetition_guard_log" not in session:
            session["repetition_guard_log"] = []

    # ── Public interface ─────────────────────────────────────────────────

    def check(self, question: str, intent) -> CheckResult:
        """
        Runs all three layers in order. Returns on first block.
        If embeddings unavailable, runs Layer 1 + 3 only (degraded mode).

        Args:
            question: The generated question text to check
            intent:   QuestionIntent from StrategyEngine (provides reason code)
        """
        log: list[str] = []

        # Layer 1 — Exact hash
        result = self._layer1_hash(question, log)
        if not result.allowed:
            self._log_event("layer1_block", question, result)
            return result

        # Layer 2 — Semantic embedding (skip in degraded mode)
        if self._embeddings_available:
            result = self._layer2_embedding(question, intent, log)
            if not result.allowed:
                self._log_event("layer2_block", question, result)
                return result
        else:
            log.append("Layer 2 skipped — embedding service unavailable (degraded mode)")

        # Layer 3 — Concept node
        result = self._layer3_concept(question, intent, log)
        if not result.allowed:
            self._log_event("layer3_block", question, result)
            return result

        log.append("All layers passed — question approved")
        return CheckResult(allowed=True, layer=0, log=log)

    def record(self, question: str, intent) -> None:
        """
        Records an approved question into all tracking stores.
        MUST be called after every approved question is asked.

        Args:
            question: The question text that was asked
            intent:   QuestionIntent (used for concept node recording)
        """
        # Layer 1 store
        self.session["asked_question_hashes"].add(self._hash(question))

        # Layer 2 store — embed asynchronously if possible
        # For now: store text only, embed lazily
        # In production: fire async embedding and append when complete
        self.session["asked_question_embeddings"].append({
            "text":      question,
            "vector":    None,   # populated by _ensure_embedded()
            "concept":   getattr(intent, "concept_node_id", "unknown"),
            "q_type":    getattr(intent, "question_type", "unknown"),
            "turn":      self.session.get("turn", 0),
        })

    def get_fallback(self, difficulty: str) -> str:
        """
        Returns a curated fallback question for the given difficulty.
        Rotates through the bank to avoid repeating the same fallback.
        """
        bank = FALLBACK_QUESTIONS.get(self.domain, {}).get(difficulty, [])
        if not bank:
            # Cross-domain fallback
            for d in ["basic", "intermediate", "foundational"]:
                bank = FALLBACK_QUESTIONS.get(self.domain, {}).get(d, [])
                if bank:
                    break

        if not bank:
            return f"Can you describe your experience working with {self.domain.replace('_', ' ')} concepts?"

        # Rotate: track which fallbacks have been used this session
        used = self.session.setdefault("used_fallback_indices", {})
        domain_used = used.setdefault(self.domain, {}).setdefault(difficulty, 0)
        idx = domain_used % len(bank)
        used[self.domain][difficulty] = idx + 1
        return bank[idx]

    def attempt_regeneration(self, first_blocked_question: str,
                              intent, generate_fn) -> RegenerationResult:
        """
        Manages the full regeneration cycle when initial generation is blocked.

        Args:
            first_blocked_question: The question that was blocked
            intent:                 QuestionIntent from StrategyEngine
            generate_fn:            Callable(intent, diversity_hint) → str
                                    The question generator function

        Returns:
            RegenerationResult with approved question or curated fallback
        """
        difficulty = getattr(intent, "difficulty", "basic")
        if hasattr(difficulty, "value"):
            difficulty = difficulty.value

        for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 1):
            if attempt == 1:
                # First retry: ask for a different aspect of the same concept
                hint = (
                    "The previous attempt was too similar to a question already asked. "
                    "Approach this concept from a completely different angle. "
                    f"Do not use similar wording to: '{first_blocked_question[:80]}'"
                )
            elif attempt == 2:
                # Second retry: switch to second-best topic
                hint = (
                    "The previous attempts were blocked for similarity. "
                    "Choose a completely different concept within this domain. "
                    "Do not ask about the same topic as before."
                )
            else:
                break  # use fallback

            try:
                new_question = generate_fn(intent, hint)
                result = self.check(new_question, intent)
                if result.allowed:
                    self.record(new_question, intent)
                    self._log_event("regeneration_success", new_question, result)
                    return RegenerationResult(
                        question=new_question,
                        fallback_used=False,
                        attempts_made=attempt,
                        final_check=result,
                    )
                self._log_event(f"regeneration_attempt_{attempt}_blocked",
                                new_question, result)
            except Exception as e:
                self._log_event(f"regeneration_attempt_{attempt}_error",
                                "", CheckResult(allowed=False, layer=0,
                                               log=[str(e)]))

        # All attempts exhausted — use curated fallback
        fallback_q = self.get_fallback(difficulty)
        fallback_result = CheckResult(
            allowed=True, layer=0,
            fallback_used=True,
            log=["Curated fallback used after all regeneration attempts blocked"],
        )
        self._log_event("fallback_used", fallback_q, fallback_result)

        return RegenerationResult(
            question=fallback_q,
            fallback_used=True,
            attempts_made=MAX_REGENERATION_ATTEMPTS,
            final_check=fallback_result,
        )

    # ── Layer 1: Exact hash ───────────────────────────────────────────────

    def _layer1_hash(self, question: str, log: list[str]) -> CheckResult:
        h = self._hash(question)
        if h in self.session["asked_question_hashes"]:
            log.append(f"Layer 1 BLOCK — exact duplicate hash {h[:8]}")
            return CheckResult(
                allowed=False, layer=1,
                block_reason=BlockReason.EXACT_DUPLICATE,
                most_similar_q=self._find_by_hash(h),
                log=log,
            )
        log.append("Layer 1 pass — no exact duplicate")
        return CheckResult(allowed=True, layer=1, log=log)

    # ── Layer 2: Semantic embedding ───────────────────────────────────────

    def _layer2_embedding(self, question: str, intent,
                           log: list[str]) -> CheckResult:
        """
        Computes cosine similarity against all previous questions.
        Uses the strategy reason code to distinguish valid follow-ups.
        """
        all_stored = self.session["asked_question_embeddings"]
        if not all_stored:
            log.append("Layer 2 pass — no previous questions to compare")
            return CheckResult(allowed=True, layer=2, log=log)

        # Ensure previous questions are embedded
        self._ensure_embedded(all_stored)

        # Embed current question
        try:
            t0 = time.time()
            current_vec = self._get_embedding(question)
            latency = (time.time() - t0) * 1000
            log.append(f"Layer 2 embedding: {latency:.0f}ms")
        except Exception as e:
            self._embeddings_available = False
            log.append(f"Layer 2 SKIP — embedding failed: {e}")
            return CheckResult(
                allowed=True, layer=2,
                block_reason=BlockReason.EMBEDDING_UNAVAILABLE,
                log=log,
            )

        # Find most similar previous question
        max_sim   = 0.0
        most_sim  = ""
        for stored in all_stored:
            if stored.get("vector") is None:
                continue
            sim = self._cosine(current_vec, stored["vector"])
            if sim > max_sim:
                max_sim  = sim
                most_sim = stored["text"]

        log.append(f"Layer 2 max similarity: {max_sim:.3f}")

        # Hard block — always
        if max_sim >= SIMILARITY_HARD_BLOCK:
            log.append(f"Layer 2 BLOCK — similarity {max_sim:.3f} >= {SIMILARITY_HARD_BLOCK}")
            return CheckResult(
                allowed=False, layer=2,
                similarity_score=max_sim,
                block_reason=BlockReason.SEMANTIC_DUPLICATE,
                most_similar_q=most_sim,
                log=log,
            )

        # Soft zone — check if this is a valid follow-up
        if max_sim >= SIMILARITY_SOFT_REVIEW:
            q_type = str(getattr(intent, "question_type", "")).replace(
                "QuestionType.", ""
            )
            if q_type in VALID_FOLLOWUP_TYPES:
                log.append(
                    f"Layer 2 ALLOW (valid follow-up) — similarity {max_sim:.3f}, "
                    f"type={q_type}"
                )
                return CheckResult(
                    allowed=True, layer=2,
                    similarity_score=max_sim,
                    is_valid_followup=True,
                    log=log,
                )
            log.append(
                f"Layer 2 BLOCK — similarity {max_sim:.3f} in soft zone, "
                f"not a valid follow-up type ({q_type})"
            )
            return CheckResult(
                allowed=False, layer=2,
                similarity_score=max_sim,
                block_reason=BlockReason.SEMANTIC_DUPLICATE,
                most_similar_q=most_sim,
                log=log,
            )

        log.append(f"Layer 2 pass — similarity {max_sim:.3f} below threshold")
        return CheckResult(
            allowed=True, layer=2,
            similarity_score=max_sim,
            log=log,
        )

    # ── Layer 3: Concept node ─────────────────────────────────────────────

    def _layer3_concept(self, question: str, intent,
                         log: list[str]) -> CheckResult:
        """
        Checks concept node ID against session's asked_concept_nodes.
        Allows revisit only for explicitly designed follow-up types.
        """
        concept_id = getattr(intent, "concept_node_id", None)
        asked_nodes = getattr(intent, "asked_concept_nodes", set())

        if concept_id is None:
            log.append("Layer 3 pass — no concept node specified")
            return CheckResult(allowed=True, layer=3, log=log)

        if concept_id not in asked_nodes:
            log.append(f"Layer 3 pass — concept '{concept_id}' not yet asked")
            return CheckResult(allowed=True, layer=3, log=log)

        # Concept already asked — check if intentional
        q_type = str(getattr(intent, "question_type", "")).replace(
            "QuestionType.", ""
        )
        if q_type in VALID_FOLLOWUP_TYPES:
            log.append(
                f"Layer 3 ALLOW (valid follow-up) — concept '{concept_id}' "
                f"revisited intentionally, type={q_type}"
            )
            return CheckResult(allowed=True, layer=3, log=log)

        log.append(
            f"Layer 3 BLOCK — concept '{concept_id}' already asked "
            f"and question type '{q_type}' is not a valid follow-up"
        )
        return CheckResult(
            allowed=False, layer=3,
            block_reason=BlockReason.CONCEPT_DUPLICATE,
            log=log,
        )

    # ── Embedding helpers ─────────────────────────────────────────────────

    def _get_embedding(self, text: str) -> list[float]:
        """Calls OpenAI embeddings API. Raises on failure."""
        from config import openai_client
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:512],   # truncate to control cost
        )
        return resp.data[0].embedding

    def _ensure_embedded(self, stored_list: list[dict]) -> None:
        """
        Lazily embeds any stored questions that have vector=None.
        In production this runs async — here it runs inline.
        If embedding fails, leaves vector=None and continues in degraded mode.
        """
        for item in stored_list:
            if item.get("vector") is not None:
                continue
            try:
                item["vector"] = self._get_embedding(item["text"])
            except Exception:
                self._embeddings_available = False
                break   # stop trying if service is unavailable

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(
            text.lower().strip().encode("utf-8")
        ).hexdigest()

    def _find_by_hash(self, h: str) -> Optional[str]:
        for item in self.session.get("asked_question_embeddings", []):
            if self._hash(item["text"]) == h:
                return item["text"]
        return None

    def _log_event(self, event_type: str, question: str,
                   result: CheckResult) -> None:
        self.session["repetition_guard_log"].append({
            "ts":         time.time(),
            "event_type": event_type,
            "question":   question[:100],
            "layer":      result.layer,
            "similarity": result.similarity_score,
            "reason":     result.block_reason,
        })


# ── Session integration ────────────────────────────────────────────────────

def initialize_guard(session: dict) -> RepetitionGuard:
    """
    Returns a RepetitionGuard for the session.
    Call this at the start of each question generation cycle.
    """
    return RepetitionGuard(session)


def get_repetition_stats(session: dict) -> dict:
    """
    Returns repetition guard statistics for observability.
    """
    log   = session.get("repetition_guard_log", [])
    total = len(session.get("asked_question_embeddings", []))
    return {
        "total_questions":        total,
        "layer1_blocks":          sum(1 for e in log if e["event_type"] == "layer1_block"),
        "layer2_blocks":          sum(1 for e in log if e["event_type"] == "layer2_block"),
        "layer3_blocks":          sum(1 for e in log if e["event_type"] == "layer3_block"),
        "regeneration_successes": sum(1 for e in log if e["event_type"] == "regeneration_success"),
        "fallbacks_used":         sum(1 for e in log if e["event_type"] == "fallback_used"),
        "repetition_rate":        round(
            sum(1 for e in log if "block" in e["event_type"]) / max(total, 1), 3
        ),
    }