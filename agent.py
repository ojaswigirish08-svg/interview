"""
agent.py — Realistic Interview Agent
─────────────────────────────────────
Upgraded for genuine conversational feel:
  1. Candidate memory — extracts specific claims from every answer,
     builds a growing profile the interviewer references
  2. Specific reactions — quotes actual phrases the candidate said,
     never generic "Good answer"
  3. Adaptive warmup — genuinely builds on what candidate reveals
  4. Bridge questions — connects topics naturally ("You mentioned X earlier...")
  5. Push-back on vague answers — demands numbers when candidate is imprecise
  6. Honest admission handling — warm, specific, teaches something
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config import (
    DIFFICULTY_LABELS, DOMAIN_TOPICS, openai_client,
    cerebras_client,
)
from strategy_engine import (
    StrategyDecision, QuestionIntent, Phase,
    get_or_create_engine, strategy_decide, strategy_update,
)
from repetition_guard import RepetitionGuard
from evaluation_validator import (
    validate_evaluation, ValidationResult, ValidationStatus,
    should_defer_report,
)


# ── Observability helpers ──────────────────────────────────────────────────
def _track_llm(sid, step, model, latency, **kw):
    try:
        from main import track_llm_call
        track_llm_call(sid, step, model, latency, **kw)
    except Exception:
        pass

def _call_llm_json(messages, temperature=0.5, max_tokens=600,
                   session_id="unknown", step="LLM_question") -> Optional[dict]:
    try:
        from main import call_llm_json
        return call_llm_json(messages, temperature=temperature,
                              max_tokens=max_tokens,
                              _session_id=session_id, _step=step)
    except ImportError:
        import json, re as _re
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        text = _re.sub(r"```json|```", "", text).strip()
        try: return json.loads(text)
        except Exception:
            m = _re.search(r'\{.*\}', text, _re.DOTALL)
            if m:
                try: return json.loads(m.group())
                except Exception: return None
        return None

def _call_cerebras_json(messages, temperature=0.5, max_tokens=300,
                         session_id="unknown", step="resume_parsing") -> Optional[dict]:
    try:
        from main import call_cerebras_json
        return call_cerebras_json(messages, temperature=temperature,
                                   max_tokens=max_tokens,
                                   _session_id=session_id, _step=step)
    except ImportError:
        return _call_llm_json(messages, temperature=temperature,
                               max_tokens=max_tokens,
                               session_id=session_id, step=step)


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE MEMORY — the engine of conversational feel
# ══════════════════════════════════════════════════════════════════════════════

def update_candidate_profile(session: dict, answer: str,
                               eval_data: dict, topic: str) -> None:
    """
    Builds a growing profile of what the candidate has revealed.
    Called after every evaluated answer.
    The profile is injected into every subsequent question prompt.
    """
    if not answer or not eval_data:
        return

    profile = session.setdefault("candidate_profile", {
        "mentioned_tools":     [],
        "claimed_numbers":     [],
        "claimed_experiences": [],
        "strong_topics":       [],
        "weak_topics":         [],
        "honest_admissions":   [],
        "key_phrases":         [],   # exact short phrases to reference back
        "vague_answers":       [],   # topics where they were imprecise
    })

    quality   = eval_data.get("quality", "adequate")
    score     = eval_data.get("score", 5)
    try: score = int(score or 5)
    except: score = 5

    # Extract tool mentions
    known_tools = ["ICC2", "PrimeTime", "Calibre", "Virtuoso", "Genus",
                   "Innovus", "Xcelium", "VCS", "Questa", "JasperGold",
                   "Formality", "VC Formal", "StarRC", "Quantus"]
    for tool in known_tools:
        if tool.lower() in answer.lower() and tool not in profile["mentioned_tools"]:
            profile["mentioned_tools"].append(tool)

    # Extract numbers (ns, ps, MHz, %, um, mV etc.)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:ps|ns|us|MHz|GHz|mV|V|%|um|nm|mA|mW)\b',
                         answer, re.IGNORECASE)
    for n in numbers[:3]:
        if n not in profile["claimed_numbers"]:
            profile["claimed_numbers"].append(n)

    # Store a short key phrase from strong answers for future reference
    if quality == "strong" and score >= 8:
        # Extract first meaningful sentence (not too long)
        sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
        if sentences:
            phrase = sentences[0][:120]
            if phrase not in profile["key_phrases"]:
                profile["key_phrases"].append(phrase)
        if topic and topic not in profile["strong_topics"]:
            profile["strong_topics"].append(topic)

    if quality in ("weak",) and topic and topic not in profile["weak_topics"]:
        profile["weak_topics"].append(topic)

    if quality == "honest_admission":
        profile["honest_admissions"].append(topic)

    # Track vague answers (adequate + no numbers on technical topic)
    if quality == "adequate" and not numbers and score < 7:
        if topic and topic not in profile["vague_answers"]:
            profile["vague_answers"].append(topic)

    # Store claimed experience phrases
    exp_patterns = [r"I (?:worked|implemented|designed|built|ran|used|handled) (.{10,60}?)(?:\.|,|$)"]
    for pat in exp_patterns:
        matches = re.findall(pat, answer, re.IGNORECASE)
        for m in matches[:2]:
            claim = m.strip()
            if claim and claim not in profile["claimed_experiences"]:
                profile["claimed_experiences"].append(claim)


def get_memory_context(session: dict) -> str:
    """
    Builds the memory injection block for the question prompt.
    This is what makes the interviewer feel like they were listening.
    """
    profile  = session.get("candidate_profile", {})
    history  = session.get("history", [])

    lines = []

    # Last 2 actual answers with key content
    recent_answered = [h for h in history if h.get("answer") and h.get("phase") != "warmup"]
    if recent_answered:
        lines.append("WHAT THE CANDIDATE SAID (last 2 answers — reference these specifically):")
        for h in recent_answered[-2:]:
            ans_excerpt = (h.get("answer") or "")[:200].strip()
            q_excerpt   = (h.get("question") or "")[:80].strip()
            qual        = (h.get("evaluation") or {}).get("quality", "")
            lines.append(f"  T{h['turn']} [{h.get('topic','')}]: Q: \"{q_excerpt}\"")
            lines.append(f"  Their answer: \"{ans_excerpt}\"  [quality: {qual}]")

    # Tools and numbers they have claimed
    if profile.get("mentioned_tools"):
        lines.append(f"Tools they've mentioned by name: {', '.join(profile['mentioned_tools'][:5])}")
    if profile.get("claimed_numbers"):
        lines.append(f"Numbers they've used: {', '.join(profile['claimed_numbers'][:5])}")
    if profile.get("claimed_experiences"):
        lines.append(f"Experience they've claimed: {'; '.join(profile['claimed_experiences'][:3])}")

    # What to probe
    if profile.get("vague_answers"):
        lines.append(f"Was vague on (push for specifics): {', '.join(profile['vague_answers'][:3])}")
    if profile.get("strong_topics"):
        lines.append(f"Demonstrated strength on: {', '.join(profile['strong_topics'][:3])}")

    return "\n".join(lines) if lines else ""


def _get_reaction_instruction(session: dict, q_type_str: str) -> str:
    """
    Generates a specific reaction instruction based on the last answer.
    References actual phrases — never generic praise.
    """
    history = session.get("history", [])
    if not history:
        return ""

    last = history[-1]
    last_answer  = (last.get("answer") or "").strip()
    last_eval    = last.get("evaluation") or {}
    last_quality = last_eval.get("quality", "adequate")
    last_notes   = last_eval.get("notes", "")
    last_topic   = last.get("topic", "")
    last_q       = (last.get("question") or "")[:80]

    if not last_answer:
        return ""

    # Get a specific excerpt from their answer
    sentences = [s.strip() for s in last_answer.replace('\n', ' ').split('.')
                 if len(s.strip()) > 15]
    excerpt = sentences[0][:100] if sentences else last_answer[:100]

    if last_quality == "strong":
        return (
            f"START with a specific 1-sentence reaction to what they just said. "
            f"They said: \"{excerpt}\". "
            f"Pick ONE specific thing they got right and name it. "
            f"Example: 'Good, you identified the CTS buffer chain correctly.' "
            f"NOT 'Great answer' or 'That's correct'. Be specific."
        )

    elif last_quality == "honest_admission":
        missing = last_eval.get("missing_points", [])
        missing_text = missing[0] if missing else "the concept"
        return (
            f"They admitted they don't know. They said: \"{excerpt}\". "
            f"React with genuine warmth — say something like 'That's fine, "
            f"knowing your limits is valuable in engineering.' "
            f"Then give a BRIEF 1-sentence teaching hint about {missing_text} "
            f"before asking your next question. Keep the hint under 20 words."
        )

    elif last_quality == "weak":
        return (
            f"They struggled. They said: \"{excerpt}\". "
            f"DO NOT say 'wrong' or 'incorrect'. "
            f"Acknowledge what they got right first: '{last_notes[:80]}'. "
            f"Then gently probe the gap with your question."
        )

    elif last_quality == "poor_articulation":
        return (
            f"They know it but couldn't explain it clearly. They said: \"{excerpt}\". "
            f"Say 'I think you know this — let me ask it differently.' "
            f"Then rephrase as a concrete scenario."
        )

    elif last_quality == "adequate":
        # Check if they were vague (no numbers on a technical answer)
        has_numbers = bool(re.search(r'\b\d+(?:\.\d+)?\s*(?:ps|ns|MHz|%|um|mV)\b',
                                      last_answer, re.IGNORECASE))
        if not has_numbers and last_topic:
            return (
                f"They answered but were vague. They said: \"{excerpt}\". "
                f"Push for specifics: start with something like "
                f"'You're on the right track — can you give me a number? "
                f"What would you actually expect for [specific value]?'"
            )
        return (
            f"They gave an adequate answer. React briefly and specifically, "
            f"then ask your next question. Reference: \"{excerpt[:60]}\""
        )

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

_LEVEL_QUESTION_RULES = {
    "fresh_graduate": """
QUESTION RULES — FRESH GRADUATE:
- Ask only definitions and purpose. "What is X and what problem does it solve?"
- No tool commands, no debugging scenarios, no numerical targets.
- Accept conceptual answers. Do not probe for tool experience.
- Maximum difficulty: basic. Do not go above it.
""",
    "trained_fresher": """
QUESTION RULES — TRAINED FRESHER:
- Ask concepts and simple workflow steps. "How would you approach X?"
- Tool flows at high level are acceptable — they had training.
- Simple scenarios are fine. "If you saw X, what would you check first?"
- Maximum difficulty: intermediate.
""",
    "experienced_junior": """
QUESTION RULES — EXPERIENCED JUNIOR (1–2 years):
- Ask real scenarios and tool-level detail. Expect standard flows.
- Require numbers. "What target skew did you use and why?"
- Difficulty up to advanced. Push on scenarios they should have run.
""",
    "experienced_senior": """
QUESTION RULES — EXPERIENCED SENIOR (3+ years):
- Ask edge cases, trade-offs, architecture decisions, tool internals.
- Demand numerical precision. "What derating factor and why?"
- Push on WHY, not just what. Difficulty should be advanced or expert.
- Basic questions are a waste of their time.
""",
}

_DOMAIN_PROMPTS = {
    "physical_design": (
        "You are Ranjitha, a senior Physical Design engineer at a top semiconductor company, "
        "conducting a mock interview. You are warm but technically rigorous. "
        "Domain: STA, synthesis, floorplanning, placement, CTS, routing, "
        "congestion, IR drop, EM, ECO, timing closure."
    ),
    "analog_layout": (
        "You are Ranjitha, a senior Analog Layout engineer, "
        "conducting a mock interview. You are warm but technically rigorous. "
        "Domain: MOSFET behavior, matching, parasitics, LDE, EMIR, "
        "layout techniques, symmetry, DRC/LVS."
    ),
    "design_verification": (
        "You are Ranjitha, a senior Design Verification engineer, "
        "conducting a mock interview. You are warm but technically rigorous. "
        "Domain: SystemVerilog, UVM, assertions, coverage, testbench, "
        "simulation, formal verification."
    ),
}


def _build_question_prompt(session: dict, intent: QuestionIntent) -> str:
    r        = session.get("resume", {})
    domain   = r.get("domain", "physical_design")
    level    = r.get("level", "trained_fresher")
    name     = strip_initials(r.get("candidate_name", "Candidate")).split()[0]  # first name only
    tools    = ", ".join(r.get("tools", []))
    projects = ", ".join(r.get("key_projects", []))

    domain_prompt  = _DOMAIN_PROMPTS.get(domain, "You are a senior VLSI interviewer.")
    level_rules    = _LEVEL_QUESTION_RULES.get(level, "Calibrate to candidate level.")
    difficulty_str = intent.difficulty.value if hasattr(intent.difficulty, "value") else str(intent.difficulty)
    q_type_str     = intent.question_type.value if hasattr(intent.question_type, "value") else str(intent.question_type)
    phase_str      = intent.phase.value if hasattr(intent.phase, "value") else str(intent.phase)

    # ── Memory context — what the candidate has said so far ────────────
    memory_ctx = get_memory_context(session)

    # ── Specific reaction instruction ──────────────────────────────────
    reaction_instruction = _get_reaction_instruction(session, q_type_str)

    # ── Full question history — no truncation ──────────────────────────
    prev_qs = intent.all_asked_questions
    prev_qs_text = "\n".join(f"- {q}" for q in prev_qs) if prev_qs else "None"

    # ── Hint for recovery probes ───────────────────────────────────────
    hint_instruction = ""
    history = session.get("history", [])
    if history and q_type_str == "recovery_probe":
        last_entry = history[-1]
        missing = (last_entry.get("evaluation") or {}).get("missing_points", [])
        if missing:
            hint_instruction = (
                f"\nGive a small helpful hint. Missing concepts: "
                f"{', '.join(missing[:3])}. Set hint_given=true in your response."
            )

    # ── Contradiction instruction ──────────────────────────────────────
    contradiction_instruction = ""
    if intent.contradiction_extra:
        pair  = intent.contradiction_extra.get("pair", {})
        angle = intent.contradiction_extra.get("angle", "angle_2")
        q_key = "angle_2" if angle == "angle_2" else "angle_1"
        specific_q = pair.get(q_key, "")
        if specific_q:
            contradiction_instruction = f'\nUSE THIS EXACT QUESTION:\n"{specific_q}"'

    # ── Bridge instruction — connect to earlier answers ────────────────
    bridge_instruction = ""
    profile = session.get("candidate_profile", {})
    strong_topics = profile.get("strong_topics", [])
    claimed_exp   = profile.get("claimed_experiences", [])
    # If we know something specific they said, create a bridge
    if strong_topics and intent.topic.replace("_"," ") not in strong_topics:
        if claimed_exp:
            bridge_instruction = (
                f"\nNATURAL BRIDGE (optional): If it flows naturally, "
                f"you can connect this question to something they mentioned earlier. "
                f"They claimed: '{claimed_exp[0][:80]}'. "
                f"Use their own words to make it feel personal."
            )

    return f"""{domain_prompt}
{level_rules}

YOU ARE INTERVIEWING: {name} | {domain.replace('_',' ')} | {level.replace('_',' ')}
Resume tools: {tools} | Projects: {projects}
Phase: {phase_str}

{f"═══ WHAT {name.upper()} HAS SAID SO FAR ═══" if memory_ctx else ""}
{memory_ctx}

═══ YOUR TASK THIS TURN ═══
Topic:           {intent.topic.replace('_',' ')}
Concept to test: {intent.concept_description}
Question type:   {q_type_str}
Difficulty:      {difficulty_str}
{hint_instruction}{contradiction_instruction}{bridge_instruction}

═══ HOW TO REACT BEFORE ASKING ═══
{reaction_instruction if reaction_instruction else "This is the first question — no reaction needed."}

═══ PREVIOUSLY ASKED — DO NOT repeat ═══
{prev_qs_text}

═══ RULES ═══
1. React first (1 sentence, specific to what they said), then ask your question.
2. ONE question only. Max 2 sentences. Conversational, speakable — this is TTS.
3. Use {name}'s first name occasionally — makes it personal.
4. If they gave a number, challenge it or build on it.
5. If they were vague, your question should demand specifics.
6. No preambles like "Moving on to..." or "Next topic...".
7. Sound like a human who was listening, not a test system.

Return ONLY valid JSON, no markdown:
{{
  "question":      "Your reaction + question — natural, conversational",
  "question_type": "{q_type_str}",
  "topic":         "{intent.topic}",
  "difficulty":    "{difficulty_str}",
  "hint_given":    false,
  "hint_text":     null
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_eval_prompt(session: dict, question: str, answer: str,
                        difficulty: str, question_type: str) -> str:
    try:
        from main import build_evaluation_prompt
        return build_evaluation_prompt(session, question, answer,
                                        difficulty, question_type)
    except ImportError:
        r     = session.get("resume", {})
        level = r.get("level", "trained_fresher")
        return (
            f"Evaluate this VLSI interview answer.\n"
            f"Level: {level} | Difficulty: {difficulty} | Type: {question_type}\n"
            f"Question: {question}\nAnswer: {answer}\n"
            "Return JSON: quality, accuracy, confidence_level, quadrant, "
            "expected_points, missing_points, score (0-10), level_gap (-3 to 3), "
            "score_reasoning, notes"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CORE: GENERATE QUESTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_question(session: dict,
                      candidate_answer: Optional[str] = None) -> dict:
    sid     = session.get("id", "unknown")
    t_start = time.time()

    # ── Step 1: Strategy engine decides what to ask ──────────────────────
    decision: StrategyDecision = strategy_decide(session)
    intent:   QuestionIntent   = decision.intent

    print(
        f"[Strategy] phase={decision.current_phase.value} "
        f"topic={intent.topic} type={intent.question_type.value} "
        f"concept={intent.concept_node_id} score={intent.priority_score:.2f}"
    )

    # ── Step 2: Build prompts ─────────────────────────────────────────────
    q_prompt   = _build_question_prompt(session, intent)
    q_messages = [{"role": "system", "content": q_prompt}]

    if candidate_answer:
        q_messages.append({"role": "user", "content": candidate_answer})
    else:
        q_messages.append({"role": "user", "content": "[START INTERVIEW]"})

    # ── Step 3: Parallel — question generation + evaluation ─────────────
    raw_eval:   Optional[dict] = None
    raw_result: Optional[dict] = None

    def _do_eval():
        if not candidate_answer or not session.get("history"):
            return None
        last_entry = session["history"][-1]
        prompt = _build_eval_prompt(
            session,
            last_entry.get("question", ""),
            candidate_answer,
            last_entry.get("difficulty", "basic"),
            last_entry.get("question_type", "definition"),
        )
        return _call_llm_json(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=600,
            session_id=sid, step="LLM_evaluation",
        )

    def _do_qgen():
        return _call_llm_json(
            q_messages, temperature=0.65, max_tokens=450,
            session_id=sid, step="LLM_question",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if candidate_answer and session.get("history"):
            futures["eval"] = executor.submit(_do_eval)
        futures["qgen"] = executor.submit(_do_qgen)
        for name, future in futures.items():
            try:
                val = future.result()
                if name == "eval":   raw_eval   = val
                elif name == "qgen": raw_result = val
            except Exception as e:
                print(f"[Agent] parallel {name} failed: {e}")

    print(f"[Timing] parallel: {time.time()-t_start:.2f}s")

    # ── Step 4: Repetition guard ──────────────────────────────────────────
    guard = RepetitionGuard(session)

    def _regenerate_fn(_, hint: str) -> Optional[str]:
        regen_prompt = q_prompt + f"\n\nDIVERSITY INSTRUCTION: {hint}"
        regen_msgs   = [{"role": "system", "content": regen_prompt},
                        {"role": "user",   "content": candidate_answer or "[START]"}]
        r = _call_llm_json(regen_msgs, temperature=0.75, max_tokens=450,
                            session_id=sid, step="LLM_question")
        return r.get("question", "") if r else ""

    if raw_result and raw_result.get("question"):
        generated_q = raw_result["question"]
        check = guard.check(generated_q, intent)
        if not check.allowed:
            print(f"[Guard] blocked layer={check.layer} — regenerating")
            regen = guard.attempt_regeneration(generated_q, intent, _regenerate_fn)
            raw_result["question"] = regen.question
            if regen.fallback_used:
                raw_result["_fallback_used"] = True
        guard.record(raw_result["question"], intent)
    else:
        difficulty_str = (intent.difficulty.value
                          if hasattr(intent.difficulty, "value")
                          else DIFFICULTY_LABELS[session.get("difficulty_level", 1)])
        fallback_q = guard.get_fallback(difficulty_str)
        raw_result = {
            "question":       fallback_q,
            "question_type":  intent.question_type.value,
            "topic":          intent.topic,
            "difficulty":     difficulty_str,
            "_fallback_used": True,
        }
        guard.record(fallback_q, intent)
        print("[Agent] LLM generation failed — using curated fallback")

    # ── Step 5: Validate evaluation ───────────────────────────────────────
    validation:     Optional[ValidationResult] = None
    validated_eval: Optional[dict]             = None

    if raw_eval is not None and candidate_answer and session.get("history"):
        last_entry = session["history"][-1]

        def _reeval_fn(failure_desc: str) -> Optional[dict]:
            prompt = _build_eval_prompt(
                session,
                last_entry.get("question", ""),
                candidate_answer,
                last_entry.get("difficulty", "basic"),
                last_entry.get("question_type", "definition"),
            )
            return _call_llm_json(
                [{"role": "user", "content": prompt + "\n\n" + failure_desc}],
                temperature=0.2, max_tokens=600,
                session_id=sid, step="LLM_evaluation",
            )

        validation = validate_evaluation(
            session, raw_eval,
            last_entry.get("question", ""),
            candidate_answer,
            last_entry.get("difficulty", "basic"),
            last_entry.get("question_type", "definition"),
            reeval_fn=_reeval_fn,
        )
        validated_eval = validation.eval

        # ── Update candidate memory after validation ──────────────────
        if validated_eval and candidate_answer:
            update_candidate_profile(
                session, candidate_answer, validated_eval,
                last_entry.get("topic", "")
            )

        if validation.deferred:
            print(f"[Validator] DEFERRED at turn {session.get('turn')}")
        elif validation.validation_flag:
            print(f"[Validator] flag={validation.validation_flag}")

    # ── Step 6: Update strategy engine ───────────────────────────────────
    if validated_eval and candidate_answer:
        strategy_update(
            session, validated_eval,
            topic=session.get("last_topic", intent.topic),
            concept_node_id=session.get("last_concept_node_id", intent.concept_node_id),
        )

    # ── Step 7: Build result ──────────────────────────────────────────────
    difficulty_val = (raw_result.get("difficulty")
                      or (intent.difficulty.value
                          if hasattr(intent.difficulty, "value")
                          else DIFFICULTY_LABELS[session.get("difficulty_level", 1)]))

    result = {
        "question":            raw_result.get("question", ""),
        "question_type":       raw_result.get("question_type", intent.question_type.value),
        "topic":               raw_result.get("topic", intent.topic),
        "difficulty":          difficulty_val,
        "hint_given":          raw_result.get("hint_given", False),
        "hint_text":           raw_result.get("hint_text"),
        "evaluation":          validated_eval,
        "_strategy_decision":  decision,
        "_validation_result":  validation,
        "_concept_node_id":    intent.concept_node_id,
        "_fallback_used":      raw_result.get("_fallback_used", False),
        "_extra":              intent.contradiction_extra if intent.contradiction_extra else None,
    }

    session["last_concept_node_id"] = intent.concept_node_id
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def process_evaluation(session: dict,
                        eval_result: dict,
                        topic: str,
                        concept_node_id: str) -> dict:
    if not eval_result:
        return eval_result

    quality    = eval_result.get("quality", "adequate")
    score      = 5
    try: score = int(eval_result.get("score") or 5)
    except Exception: pass
    confidence = eval_result.get("confidence_level", "medium")

    session["last_eval_quality"] = quality
    session["last_confidence"]   = confidence

    if quality == "strong":
        session["consecutive_strong"] = session.get("consecutive_strong", 0) + 1
        session["consecutive_weak"]   = 0
    elif quality == "weak":
        session["consecutive_weak"]   = session.get("consecutive_weak", 0) + 1
        session["consecutive_strong"] = 0
    else:
        session["consecutive_strong"] = 0
        session["consecutive_weak"]   = 0

    if session.get("consecutive_strong", 0) >= 3 and session.get("difficulty_level", 1) < 4:
        session["difficulty_level"]   = session.get("difficulty_level", 1) + 1
        session["consecutive_strong"] = 0
    elif session.get("consecutive_weak", 0) >= 3 and session.get("difficulty_level", 1) > 0:
        session["difficulty_level"]   = session.get("difficulty_level", 1) - 1
        session["consecutive_weak"]   = 0

    strategy_update(session, eval_result, topic, concept_node_id)

    scored_history = [
        h for h in session.get("history", [])
        if h.get("evaluation")
        and (h.get("evaluation") or {}).get("quality") not in ("warmup", None)
        and h.get("phase") != "warmup"
    ]
    scores = []
    for h in scored_history:
        try: scores.append(int(h["evaluation"].get("score") or 5))
        except Exception: scores.append(5)
    session["trajectory_type"] = compute_trajectory(scores)

    return eval_result


# ══════════════════════════════════════════════════════════════════════════════
# WARMUP — genuinely adaptive
# ══════════════════════════════════════════════════════════════════════════════

def strip_initials(name: str) -> str:
    parts  = name.split()
    actual = [p for p in parts if not re.match(r'^[A-Z]\.$', p)]
    return " ".join(actual) if actual else name


def generate_greeting(session: dict) -> dict:
    r     = session.get("resume", {})
    name  = strip_initials(r.get("candidate_name", "Candidate")).split()[0]
    level = r.get("level", "trained_fresher").replace("_", " ")
    domain = r.get("domain", "physical_design").replace("_", " ")

    # Returning candidate — more personal greeting
    if session.get("is_returning") and session.get("previous_sessions"):
        last = session["previous_sessions"][-1]
        weak = ", ".join(last.get("weak_topics", [])[:2]) or "some areas"
        score = last.get("overall_score", "?")
        return {
            "question": (
                f"Welcome back, {name}! Good to see you again. "
                f"Last time you scored {score}/100 and had some room to grow in {weak}. "
                f"Let's see how you've improved. Go ahead — tell me what you've been "
                f"working on or studying since we last spoke."
            ),
            "question_type": "greeting", "topic": "greeting", "difficulty": "basic",
        }

    return {
        "question": (
            f"Hi {name}! Welcome. I'm going to interview you today for a "
            f"{level} {domain} role. "
            f"Let's start simple — tell me about yourself, your background, "
            f"and what you've been working on most recently."
        ),
        "question_type": "greeting", "topic": "greeting", "difficulty": "basic",
    }


def generate_warmup_question(session: dict,
                              candidate_answer: Optional[str] = None) -> dict:
    """
    Adaptive warmup — genuinely builds on what the candidate reveals.
    First question: pick a skill from resume.
    Second question: dig into something specific from their last answer.
    Third question: calibrate — light scenario to set difficulty baseline.
    """
    import random
    resume       = session.get("resume", {})
    sid          = session.get("id", "unknown")
    name         = strip_initials(resume.get("candidate_name", "Candidate")).split()[0]
    skills       = resume.get("skills", []) + resume.get("tools", [])
    domain       = resume.get("domain", "physical_design")
    warmup_turn  = session.get("warmup_turns", 0)  # 0, 1, 2

    warmup_asked = session.setdefault("warmup_skills_asked", [])
    prev_qs      = [h["question"] for h in session.get("history", []) if h.get("question")]
    prev_qs_text = "\n".join(f"- {q}" for q in prev_qs) if prev_qs else "None"

    # ── Warmup turn 0: pick a skill from resume ──────────────────────────
    if warmup_turn == 0 or not candidate_answer:
        remaining = [s for s in skills if s not in warmup_asked] or skills
        random.shuffle(remaining)
        chosen_skill = remaining[0] if remaining else "VLSI"

        prompt = (
            f"You are a friendly senior {domain.replace('_',' ')} interviewer "
            f"starting a warmup conversation with {name}.\n"
            f"Ask ONE simple, conversational question about their experience with: {chosen_skill}\n"
            f"Keep it friendly and open-ended. 1 sentence only. No technical depth yet.\n"
            f"Previously asked (do not repeat):\n{prev_qs_text}\n\n"
            f"Return ONLY JSON: {{\"question\": \"your question\", \"skill_asked\": \"{chosen_skill}\"}}"
        )
        result = _call_cerebras_json(
            [{"role": "user", "content": prompt}],
            temperature=0.8, max_tokens=200, session_id=sid, step="resume_parsing"
        )
        if not result or "question" not in result:
            result = {"question": f"Can you tell me a bit about your experience with {chosen_skill}?",
                      "skill_asked": chosen_skill}
        warmup_asked.append(result.get("skill_asked", chosen_skill))
        result["question_type"] = "warmup"

    # ── Warmup turn 1: dig into something specific from their answer ─────
    elif warmup_turn == 1 and candidate_answer:
        # Extract a specific claim from their last answer
        last_answer = candidate_answer[:300]
        prompt = (
            f"You are a senior {domain.replace('_',' ')} interviewer in a warmup conversation with {name}.\n"
            f"They just said: \"{last_answer}\"\n\n"
            f"Ask ONE follow-up question that picks ONE specific thing they mentioned and "
            f"asks them to tell you more about it. Make it feel natural — like you were "
            f"listening. 1-2 sentences max.\n"
            f"Previously asked (do not repeat):\n{prev_qs_text}\n\n"
            f"Return ONLY JSON: {{\"question\": \"your follow-up question\", \"skill_asked\": \"inferred skill\"}}"
        )
        result = _call_cerebras_json(
            [{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=200, session_id=sid, step="resume_parsing"
        )
        if not result or "question" not in result:
            result = {"question": f"Interesting — can you tell me more about that?",
                      "skill_asked": "general"}
        result["question_type"] = "warmup"

    # ── Warmup turn 2: light scenario to calibrate ───────────────────────
    elif warmup_turn == 2 and candidate_answer:
        level = resume.get("level", "trained_fresher")
        remaining = [s for s in skills if s not in warmup_asked] or skills

        prompt = (
            f"You are a senior {domain.replace('_',' ')} interviewer finishing warmup with {name}.\n"
            f"Candidate level: {level.replace('_',' ')}\n"
            f"Their skills: {', '.join(skills[:5])}\n"
            f"Their last answer: \"{candidate_answer[:200]}\"\n\n"
            f"Ask ONE light scenario question to calibrate their level before the technical round. "
            f"Should feel like a natural conversation-ender, not a test question. "
            f"Something like 'Before we dive in — what's been the most challenging thing "
            f"you've worked on in [domain]?' 1-2 sentences.\n"
            f"Previously asked:\n{prev_qs_text}\n\n"
            f"Return ONLY JSON: {{\"question\": \"your calibration question\", \"skill_asked\": \"calibration\"}}"
        )
        result = _call_cerebras_json(
            [{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=200, session_id=sid, step="resume_parsing"
        )
        if not result or "question" not in result:
            result = {
                "question": f"Before we get into the technical questions — what's the most complex {domain.replace('_',' ')} problem you've tackled so far?",
                "skill_asked": "calibration"
            }
        result["question_type"] = "warmup"

    else:
        # Fallback
        skill = skills[0] if skills else "VLSI"
        result = {"question": f"Tell me about your experience with {skill}.",
                  "question_type": "warmup", "skill_asked": skill}

    # Add to repetition guard so technical round never repeats warmup questions
    guard = RepetitionGuard(session)
    class _WarmupIntent:
        question_type       = "warmup"
        concept_node_id     = f"warmup_{result.get('skill_asked', 'general')}"
        asked_concept_nodes = set()
    guard.record(result.get("question", ""), _WarmupIntent())

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def compute_trajectory(scores: list) -> str:
    if len(scores) < 4:
        return "insufficient_data"
    third      = max(1, len(scores) // 3)
    first      = sum(scores[:third]) / third
    last_chunk = scores[2*third:] or scores[-1:]
    last       = sum(last_chunk) / len(last_chunk)
    variance   = max(scores) - min(scores)
    if variance > 4:       return "spiky"
    if last > first + 1.5: return "rising"
    if first > last + 1.5: return "falling"
    if first >= 7 and last >= 7: return "flat_strong"
    return "flat_weak"


def get_trajectory_interpretation(t: str) -> str:
    return {
        "rising":            "Started nervous, improved significantly.",
        "falling":           "Started strong, performance dropped.",
        "spiky":             "Inconsistent — deep on known topics, suspicious on others.",
        "flat_strong":       "Consistently strong throughout.",
        "flat_weak":         "Consistently struggled — needs more preparation.",
        "insufficient_data": "Too few answers to determine pattern.",
    }.get(t, "Pattern not determined.")


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATS
# ══════════════════════════════════════════════════════════════════════════════

def get_agent_stats(session: dict) -> dict:
    from repetition_guard import get_repetition_stats
    from evaluation_validator import EvaluationValidator, get_deferred_summary

    engine  = get_or_create_engine(session)
    ev      = EvaluationValidator(session)
    rep     = get_repetition_stats(session)
    defer   = get_deferred_summary(session)
    val     = ev.get_validation_stats()

    return {
        "phase":               engine.get_phase().value,
        "tech_turn":           engine.tech_turn,
        "turns_remaining":     engine.turns_remaining,
        "coverage":            engine.get_coverage_summary(),
        "repetition":          rep,
        "validation":          val,
        "deferred":            defer,
        "should_defer_report": should_defer_report(session),
        "candidate_profile":   session.get("candidate_profile", {}),
        "topics_untested":     [t for t, c in engine.coverage.items()
                                 if c.confidence.value == "untested"],
        "topics_confirmed":    [t for t, c in engine.coverage.items()
                                 if c.confidence.value == "confirmed"],
        "suspected_weak":      [t for t, c in engine.coverage.items()
                                 if c.weak_flag.value == "suspected"],
    }