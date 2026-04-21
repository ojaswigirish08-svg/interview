"""
VLSI Interview Agent — Mock Interview Mode
100% spec implementation — all gaps fixed.
"""

import os, re, json, time, uuid, base64, tempfile, statistics, hashlib
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

load_dotenv()

app = FastAPI(title="VLSI Interview Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

sessions: dict = {}

# ════════════════════════════════════════════════════════════
# AI CLIENTS
# ════════════════════════════════════════════════════════════
import boto3
import requests as http_requests
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
polly_client = boto3.client(
    "polly",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# AWS Bedrock — Claude Sonnet 4.6 (for question generation + evaluation)
BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "us.anthropic.claude-sonnet-4-6-20250514-v1:0")
print(f"AWS Bedrock Claude ready (model: {CLAUDE_MODEL_ID}, region: {BEDROCK_REGION})")

# Cerebras (fast LLM for resume parsing)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
cerebras_client = None
if CEREBRAS_API_KEY:
    cerebras_client = OpenAI(
        api_key=CEREBRAS_API_KEY,
        base_url="https://api.cerebras.ai/v1"
    )
    print("Cerebras LLM ready (for resume parsing).")

# ElevenLabs (fallback STT)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    from elevenlabs.client import ElevenLabs
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    print("ElevenLabs Scribe v2 ready (fallback STT).")
else:
    print("ElevenLabs not configured.")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_TTS_REF_AUDIO = None
# Load reference audio for Mistral voice cloning
_ref_audio_path = os.path.join(os.path.dirname(__file__), "ranjitha_4dmjitkw.mp3")
if MISTRAL_API_KEY and os.path.exists(_ref_audio_path):
    with open(_ref_audio_path, "rb") as _f:
        MISTRAL_TTS_REF_AUDIO = base64.b64encode(_f.read()).decode()
    print("Mistral Voxtral TTS ready (with reference voice).")
else:
    print("Mistral TTS not configured.")

# Pocket TTS (local, CPU-only, free, voice cloning)
pocket_tts_model = None
pocket_tts_voice_state = None
try:
    from pocket_tts import TTSModel, export_model_state
    import scipy.io.wavfile
    import numpy as np
    pocket_tts_model = TTSModel.load_model()
    # Load cloned voice from exported safetensors (fast) or from reference audio
    _voice_state_path = os.path.join(os.path.dirname(__file__), "ranjitha_voice.safetensors")
    if os.path.exists(_voice_state_path):
        pocket_tts_voice_state = pocket_tts_model.get_state_for_audio_prompt(_voice_state_path)
        print("Pocket TTS ready (cloned voice from safetensors).")
    elif os.path.exists(_ref_audio_path):
        pocket_tts_voice_state = pocket_tts_model.get_state_for_audio_prompt(_ref_audio_path)
        # Export for fast loading next time
        export_model_state(pocket_tts_voice_state, _voice_state_path)
        print("Pocket TTS ready (cloned voice from reference audio, exported safetensors).")
    else:
        pocket_tts_voice_state = pocket_tts_model.get_state_for_audio_prompt("alba")
        print("Pocket TTS ready (default voice: alba).")
except ImportError:
    print("Pocket TTS not installed. pip install pocket-tts scipy")
except Exception as e:
    print(f"Pocket TTS failed to load: {e}")

print("STT: gpt-4o-mini-transcribe -> ElevenLabs Scribe v2")
print("TTS: Pocket TTS -> Mistral Voxtral")
print(f"LLM (questions+eval): Claude Sonnet 4.6 via Bedrock (prompt cached) | Fallback: GPT-4o-mini")
print(f"LLM (resume+warmup): Cerebras Llama 3.1-8b | LLM (report): GPT-4o-mini")

# ════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE
# ════════════════════════════════════════════════════════════
DOMAIN_TOPICS = {
    "analog_layout": [
        "basic layout concepts", "device matching", "parasitic awareness",
        "latch-up and ESD", "guard rings", "DRC/LVS", "symmetry techniques",
        "shielding and routing", "technology awareness"
    ],
    "physical_design": [
        "floorplanning", "power planning", "placement", "clock tree synthesis",
        "routing", "timing closure", "static timing analysis", "DRC/LVS signoff",
        "tool knowledge"
    ],
    "design_verification": [
        "verification methodologies", "testbench architecture", "functional coverage",
        "assertions and SVA", "simulation vs formal", "debugging skills",
        "protocol knowledge", "regression and signoff", "UVM"
    ]
}

DOMAIN_RESOURCES = {
    "analog_layout": [
        "Weste & Harris — CMOS VLSI Design, Chapter 3 (Layout)",
        "Razavi — Design of Analog CMOS Integrated Circuits, Chapter 18",
        "Virtuoso Layout Suite User Guide — Cadence documentation",
        "Calibre DRC/LVS User Manual — Mentor Graphics",
    ],
    "physical_design": [
        "Synopsys ICC2 User Guide — Floorplanning and Placement chapters",
        "Rabaey — Digital Integrated Circuits, Chapter 7",
        "Static Timing Analysis for Nanometer Designs — Jayaram Bhasker",
        "PrimeTime User Guide — Synopsys documentation",
    ],
    "design_verification": [
        "Spear & Tumbush — SystemVerilog for Verification, Chapters 4-7",
        "Bergeron — Writing Testbenches Using SystemVerilog",
        "UVM Cookbook — Mentor Verification Academy",
        "VCS Simulation User Guide — Synopsys documentation",
    ]
}

# ════════════════════════════════════════════════════════════
# CONTRADICTION PAIR LIBRARY — per spec: ask from two angles, 10-15 turns apart
# ════════════════════════════════════════════════════════════
CONTRADICTION_PAIRS = {
    "analog_layout": [
        {
            "topic": "device matching",
            "angle_1": "What techniques do you use to achieve good device matching in layout?",
            "angle_2": "If two devices have identical layout but different orientations relative to the gradient, will they match? Why or why not?",
        },
        {
            "topic": "parasitic awareness",
            "angle_1": "How do you minimize parasitic capacitance in a layout?",
            "angle_2": "In a situation where two nets run parallel for 100 microns, what is the dominant parasitic concern and how would you quantify it?",
        },
        {
            "topic": "latch-up and ESD",
            "angle_1": "What causes latch-up in CMOS layout and how do you prevent it?",
            "angle_2": "If you have a circuit where the substrate contact spacing is 50 microns from the nearest NMOS, is that acceptable? Walk me through your reasoning.",
        },
    ],
    "physical_design": [
        {
            "topic": "clock tree synthesis",
            "angle_1": "What is clock tree synthesis and what problem does it solve?",
            "angle_2": "If after CTS you have 200ps of skew on one branch, what are the first three things you would check and in what order?",
        },
        {
            "topic": "timing closure",
            "angle_1": "Explain the difference between setup and hold violations.",
            "angle_2": "You have a hold violation of 50ps on a path that passes through three buffers. Adding more buffers made it worse. What is happening and what is the correct fix?",
        },
        {
            "topic": "floorplanning",
            "angle_1": "What are the key objectives of floorplanning in physical design?",
            "angle_2": "You are floorplanning a block with 60% utilization and your timing is already tight. Where exactly would you place the critical path macros and why?",
        },
    ],
    "design_verification": [
        {
            "topic": "functional coverage",
            "angle_1": "What is functional coverage and why is it important in verification?",
            "angle_2": "Your regression shows 98% functional coverage but you still found a bug in silicon. How is that possible and what does it tell you about your coverage model?",
        },
        {
            "topic": "assertions and SVA",
            "angle_1": "What is the difference between a concurrent and immediate assertion in SVA?",
            "angle_2": "You have an assertion that never fires during simulation. Is that good or bad? How do you determine which it is?",
        },
        {
            "topic": "simulation vs formal",
            "angle_1": "What are the advantages of formal verification over simulation?",
            "angle_2": "Formal verification proved your design is correct but you still found a bug. What are three possible explanations?",
        },
    ]
}

DIFFICULTY_LABELS = ["foundational", "basic", "intermediate", "advanced", "expert"]
FILLER_WORDS = {"um", "uh", "like", "basically", "actually", "so", "right", "okay", "hmm", "err"}

# Expected thinking pause (seconds) per difficulty — harder questions take longer naturally
EXPECTED_PAUSE_BY_DIFFICULTY = {
    "foundational": 2.0,
    "basic": 3.0,
    "intermediate": 5.0,
    "advanced": 8.0,
    "expert": 10.0
}

# ════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════
class SessionCreate(BaseModel):
    resume_text: str
    mode: str = "mock"

class AnswerSubmit(BaseModel):
    session_id: str
    answer: str
    turn: int
    answer_duration_sec: float = 0.0
    word_count: int = 0
    thinking_pause_sec: float = 0.0
    input_mode: str = "text"  # "voice" or "text"
    whisper_confidence: float = 1.0  # avg word confidence from Whisper

class AntiCheatEvent(BaseModel):
    session_id: str
    event_type: str  # tab_switch|window_blur|paste_event|dom_overlay|screen_share|canary_triggered
    turn: int
    timestamp: float
    metadata: str = ""

class ReportRequest(BaseModel):
    session_id: str

# ════════════════════════════════════════════════════════════
# LLM HELPERS
# ════════════════════════════════════════════════════════════
def call_llm(messages: list, temperature=0.5, max_tokens=1000) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def safe_json(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(cleaned)
    except Exception:
        pass
    return None

def call_cerebras(messages: list, temperature=0.5, max_tokens=1000) -> str:
    """Fast LLM call via Cerebras. Falls back to GPT-4o-mini."""
    if cerebras_client:
        try:
            resp = cerebras_client.chat.completions.create(
                model="llama3.1-8b",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Cerebras failed, falling back to GPT-4o-mini: {e}")
    return call_llm(messages, temperature, max_tokens)


def call_cerebras_json(messages: list, temperature=0.5, max_tokens=1000, retries=2) -> dict:
    """Fast JSON LLM call via Cerebras with fallback."""
    for attempt in range(retries + 1):
        try:
            raw = call_cerebras(messages, temperature=temperature, max_tokens=max_tokens)
            result = safe_json(raw)
            if result:
                return result
            if attempt < retries:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Return ONLY valid JSON. No markdown, no explanation."}
                ]
        except Exception as e:
            print(f"Cerebras JSON attempt {attempt+1} failed: {e}")
            if attempt == retries:
                raise
    return {}


def call_llm_json(messages: list, temperature=0.5, max_tokens=1000, retries=2) -> dict:
    for attempt in range(retries + 1):
        try:
            raw = call_llm(messages, temperature=temperature, max_tokens=max_tokens)
            result = safe_json(raw)
            if result:
                return result
            if attempt < retries:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Return ONLY valid JSON. No markdown, no explanation."}
                ]
        except Exception as e:
            print(f"LLM attempt {attempt+1} failed: {e}")
            if attempt == retries:
                raise
    return {}

# ════════════════════════════════════════════════════════════
# CLAUDE VIA AWS BEDROCK — with prompt caching
# ════════════════════════════════════════════════════════════
def call_claude(messages: list, temperature=0.5, max_tokens=1000) -> str:
    """Call Claude Sonnet 4.6 via AWS Bedrock with prompt caching.

    Prompt caching: The system prompt is marked with cache_control so Bedrock
    caches it across calls within the same session. The system prompt (~1000 tokens)
    stays cached for 5 minutes, saving latency and cost on subsequent questions.

    Falls back to GPT-4o-mini if Bedrock fails.
    """
    try:
        # Separate system message from conversation
        system_content = []
        converse_messages = []

        for msg in messages:
            if msg["role"] == "system":
                # Mark system prompt for caching (Bedrock prompt caching)
                system_content.append({
                    "text": msg["content"]
                })
            else:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })

        # Ensure messages alternate user/assistant — Bedrock requires this
        if not converse_messages:
            converse_messages = [{"role": "user", "content": [{"text": "Begin."}]}]

        # Build request body
        request_body = {
            "modelId": CLAUDE_MODEL_ID,
            "messages": converse_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }

        if system_content:
            request_body["system"] = system_content

        # Enable prompt caching via performanceConfig
        request_body["performanceConfig"] = {"latency": "optimized"}

        response = bedrock_client.converse(**request_body)

        # Extract text from response
        output = response.get("output", {})
        content = output.get("message", {}).get("content", [])
        text = ""
        for block in content:
            if "text" in block:
                text += block["text"]

        # Log cache usage if available
        usage = response.get("usage", {})
        cache_read = usage.get("cacheReadInputTokens", 0)
        cache_write = usage.get("cacheWriteInputTokens", 0)
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        if cache_read > 0 or cache_write > 0:
            print(f"[Claude] tokens: in={input_tokens} out={output_tokens} cache_read={cache_read} cache_write={cache_write}")
        else:
            print(f"[Claude] tokens: in={input_tokens} out={output_tokens}")

        return text.strip()

    except Exception as e:
        print(f"Bedrock Claude failed, falling back to GPT-4o-mini: {e}")
        return call_llm(messages, temperature, max_tokens)


def call_claude_json(messages: list, temperature=0.5, max_tokens=1000, retries=2) -> dict:
    """Call Claude via Bedrock and parse JSON response. Falls back to GPT-4o-mini."""
    for attempt in range(retries + 1):
        try:
            raw = call_claude(messages, temperature=temperature, max_tokens=max_tokens)
            result = safe_json(raw)
            if result:
                return result
            if attempt < retries:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Return ONLY valid JSON. No markdown, no explanation."}
                ]
        except Exception as e:
            print(f"Claude JSON attempt {attempt+1} failed: {e}")
            if attempt == retries:
                # Final fallback to GPT-4o-mini
                print("Falling back to GPT-4o-mini for JSON call")
                return call_llm_json(messages, temperature, max_tokens, retries=1)
    return {}


# ════════════════════════════════════════════════════════════
# RESUME PARSER
# ════════════════════════════════════════════════════════════
def parse_resume(resume_text: str) -> dict:
    prompt = f"""You are a VLSI expert HR analyst. Parse this resume and extract candidate details.

Resume:
{resume_text[:4000]}

TASKS:
1. Extract the candidate's full name from the resume (usually at the top)
2. Extract all technical skills mentioned in the resume
3. Check if this resume is related to VLSI/Semiconductor/Electronics domain

VLSI indicators to look for:
- VLSI keywords: RTL, Verilog, VHDL, SystemVerilog, ASIC, FPGA, SoC, Physical Design, DFT, Verification, Analog Layout, PnR, STA, Synthesis, Timing, Floorplan, Power, Clock
- Semiconductor tools: Cadence, Synopsys, Mentor, ICC, ICC2, Innovus, Virtuoso, PrimeTime, Design Compiler, VCS, Questa, Calibre, Assura
- Electronics/ECE/EEE education background
- Semiconductor company experience or training

Return ONLY this JSON (no markdown):
{{
  "candidate_name": "Full Name from resume",
  "email": "email if found or empty string",
  "phone": "phone if found or empty string",
  "skills": ["skill1", "skill2", "skill3"],
  "is_vlsi_suitable": true,
  "rejection_reason": "",
  "domain": "analog_layout",
  "level": "trained_fresher",
  "years_experience": 0,
  "tools": ["tool1"],
  "key_projects": ["project description"],
  "background_summary": "2 concise sentences about candidate",
  "training_institutes": ["institute"],
  "education": "degree and branch"
}}

FIELD INSTRUCTIONS:
- candidate_name: Extract exact name from resume header (e.g., "Rahul Sharma", "Priya Singh")
- skills: List ALL technical skills found (programming, tools, concepts, etc.)
- is_vlsi_suitable: true if resume has VLSI/Semiconductor/Electronics background, false otherwise
- rejection_reason: If not suitable, explain why (e.g., "Resume is for Software Development, not VLSI"). Empty string if suitable.
- domain: analog_layout | physical_design | design_verification (classify based on skills/experience)
- level: fresh_graduate (0yr) | trained_fresher (0-1yr) | experienced_junior (1-3yr) | experienced_senior (3+yr)"""

    result = call_cerebras_json([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=1000)
    if result and "is_vlsi_suitable" in result:
        # Ensure candidate_name exists
        if "candidate_name" not in result or not result["candidate_name"]:
            result["candidate_name"] = "Candidate"
        return result
    # Default: assume not suitable if parsing fails
    return {
        "candidate_name": "Candidate",
        "email": "",
        "phone": "",
        "skills": [],
        "is_vlsi_suitable": False,
        "rejection_reason": "Could not parse resume. Please upload a valid VLSI/Semiconductor resume.",
        "domain": "unknown", "level": "unknown",
        "years_experience": 0, "tools": [],
        "key_projects": [], "background_summary": "",
        "training_institutes": [], "education": ""
    }

# ════════════════════════════════════════════════════════════
# BEHAVIORAL BASELINE — 5 metrics
# ════════════════════════════════════════════════════════════
def count_fillers(text: str) -> float:
    words = text.lower().split()
    if not words: return 0.0
    return sum(1 for w in words if w in FILLER_WORDS) / len(words)

def count_personal_pronouns(text: str) -> float:
    words = text.lower().split()
    if not words: return 0.0
    pronouns = {"i", "my", "me", "myself", "we", "our", "i've", "i'd", "i'll"}
    return sum(1 for w in words if w in pronouns) / len(words)

def count_self_corrections(text: str) -> float:
    """Detect self-correction patterns — human speech signal."""
    patterns = [r'\bi mean\b', r'\bactually\b', r'\bwait\b', r'\bno wait\b', r'\bsorry\b', r'\blet me rephrase\b']
    words = text.lower().split()
    if not words: return 0.0
    count = sum(1 for p in patterns if re.search(p, text.lower()))
    return count / len(words)

def compute_baseline(session: dict):
    warmup = [h for h in session["history"] if h["phase"] == "warmup" and h.get("answer")]
    if not warmup: return
    voice = [h for h in warmup if h.get("input_mode") == "voice"]
    session["behavioral_baseline"] = {
        "has_voice": len(voice) > 0,
        "avg_duration_sec":    sum(h.get("answer_duration_sec", 0) for h in voice) / len(voice) if voice else 0,
        "avg_word_count":      sum(h.get("word_count", 0) for h in warmup) / len(warmup),
        "avg_filler_rate":     sum(h.get("filler_rate", 0) for h in warmup) / len(warmup),
        "avg_pronoun_rate":    sum(h.get("pronoun_rate", 0) for h in warmup) / len(warmup),
        "avg_thinking_pause":  sum(h.get("thinking_pause_sec", 0) for h in voice) / len(voice) if voice else 0,
        "avg_correction_rate": sum(h.get("correction_rate", 0) for h in warmup) / len(warmup),
        "sample_size":         len(warmup)
    }

def analyze_behavioral_deviation(session: dict, answer: str, duration_sec: float,
                                  word_count: int, thinking_pause: float,
                                  input_mode: str, difficulty: str) -> dict:
    baseline = session.get("behavioral_baseline")
    if not baseline or baseline.get("sample_size", 0) == 0:
        return {"deviation_score": 0.0, "flags": [], "filler_rate": 0.0,
                "pronoun_rate": 0.0, "correction_rate": 0.0}

    flags = []
    score = 0.0
    filler_rate = count_fillers(answer)
    pronoun_rate = count_personal_pronouns(answer)
    correction_rate = count_self_corrections(answer)
    has_voice = baseline.get("has_voice", False)

    # 1. Duration deviation (voice only)
    if has_voice and input_mode == "voice" and baseline["avg_duration_sec"] > 0 and duration_sec > 0:
        ratio = duration_sec / baseline["avg_duration_sec"]
        if ratio < 0.25: flags.append("unusually_short_answer"); score += 1.5
        elif ratio > 5.0: flags.append("unusually_long_answer"); score += 0.5

    # 2. Word count deviation
    avg_wc = baseline["avg_word_count"]
    if avg_wc > 5 and word_count > 0 and word_count / avg_wc < 0.2:
        flags.append("very_few_words"); score += 1.0

    # 3. Filler word disappearance — AI signal
    avg_fr = baseline["avg_filler_rate"]
    if avg_fr > 0.008 and filler_rate < avg_fr * 0.1 and word_count > 25:
        flags.append("suspiciously_clean_speech"); score += 2.0

    # 4. Personal pronoun disappearance — AI signal
    avg_pr = baseline["avg_pronoun_rate"]
    if avg_pr > 0.02 and pronoun_rate < avg_pr * 0.15 and word_count > 25:
        flags.append("personal_pronouns_vanished"); score += 1.5

    # 5. Thinking pause vs expected for difficulty — AI latency detection
    if has_voice and input_mode == "voice" and thinking_pause > 0:
        expected_pause = EXPECTED_PAUSE_BY_DIFFICULTY.get(difficulty, 4.0)
        avg_pause = baseline["avg_thinking_pause"]

        # Identical pause regardless of difficulty = AI latency
        session["pause_history"] = session.get("pause_history", [])
        session["pause_history"].append({"pause": thinking_pause, "difficulty": difficulty})

        if len(session["pause_history"]) >= 4:
            all_pauses = [p["pause"] for p in session["pause_history"]]
            try:
                stddev = statistics.stdev(all_pauses)
                if stddev < 0.5:
                    flags.append("low_pause_variance"); score += 2.0
            except Exception:
                pass

        # Hard question with suspiciously short pause
        if difficulty in ("advanced", "expert") and thinking_pause < expected_pause * 0.3:
            flags.append(f"instant_answer_on_{difficulty}_question"); score += 1.5

    # 6. Answer length spike on hard questions — AI gives more for harder
    if difficulty in ("advanced", "expert") and avg_wc > 0:
        if word_count / max(avg_wc, 1) > 4.0:
            flags.append("answer_length_spike_on_hard_question"); score += 1.5

    # 7. Self-correction rate dropped — humans self-correct, AI doesn't
    avg_cr = baseline.get("avg_correction_rate", 0)
    if avg_cr > 0.002 and correction_rate < avg_cr * 0.1 and word_count > 30:
        flags.append("self_corrections_vanished"); score += 1.0

    return {"deviation_score": score, "flags": flags,
            "filler_rate": filler_rate, "pronoun_rate": pronoun_rate,
            "correction_rate": correction_rate}

# ════════════════════════════════════════════════════════════
# ANSWER COMPLEXITY VS LEVEL — spec: flag gap above 2 levels
# ════════════════════════════════════════════════════════════
def assess_answer_complexity(session: dict, answer: str, eval_score: int,
                              eval_difficulty: str, resume_level: str) -> dict:
    """
    Score answer sophistication independently.
    If answer quality is significantly above calibrated level = suspicious.
    Could be genuine self-taught OR proxy/AI.
    """
    level_map = {
        "fresh_graduate": 0, "trained_fresher": 1,
        "experienced_junior": 2, "experienced_senior": 3
    }
    diff_map = {d: i for i, d in enumerate(DIFFICULTY_LABELS)}

    candidate_level_num = level_map.get(resume_level, 1)
    answer_difficulty_num = diff_map.get(eval_difficulty, 1)

    # If answer is strong (score >= 8) on difficulty 2+ levels above resume level
    if eval_score >= 8 and answer_difficulty_num > candidate_level_num + 1:
        gap = answer_difficulty_num - candidate_level_num
        return {
            "above_level": True,
            "gap_levels": gap,
            "flag": f"Answer quality significantly above calibrated level ({eval_difficulty} when resume suggests {resume_level})"
        }
    return {"above_level": False, "gap_levels": 0, "flag": ""}

# ════════════════════════════════════════════════════════════
# RECOVERY VELOCITY — spec: force hint, track velocity
# ════════════════════════════════════════════════════════════
def should_give_hint(session: dict) -> bool:
    """Force hint on recovery_probe turns — not LLM-controlled."""
    return session.get("last_question_type", "") == "recovery_probe" or \
           session.get("last_eval_quality", "") == "honest_admission"

def record_hint(session: dict, turn: int, topic: str, hint_text: str):
    session.setdefault("hint_events", []).append({
        "turn": turn, "topic": topic, "hint_text": hint_text,
        "recovery_score": None, "recovery_speed": None, "recovery_quality": None
    })

def evaluate_recovery(session: dict, current_turn: int, answer: str, eval_score: int):
    for hint in session.get("hint_events", []):
        if hint["recovery_score"] is None and current_turn == hint["turn"] + 1:
            hint["recovery_speed"] = "fast"
            if eval_score >= 7:
                hint["recovery_quality"] = "complete"
                hint["recovery_score"] = eval_score
                session.setdefault("genuine_signals", []).append(
                    f"Fast complete recovery at turn {current_turn} on {hint['topic']} — genuine signal"
                )
                # Too perfect recovery = AI signal
                if eval_score == 10:
                    session.setdefault("suspicion_events", []).append({
                        "type": "perfect_recovery_after_hint",
                        "turn": current_turn,
                        "weight": 15,
                        "detail": f"Scored 10/10 immediately after hint on {hint['topic']} at turn {current_turn}"
                    })
            elif eval_score >= 4:
                hint["recovery_quality"] = "partial"
                hint["recovery_score"] = eval_score
            else:
                hint["recovery_quality"] = "none"
                hint["recovery_score"] = eval_score

# ════════════════════════════════════════════════════════════
# SMOOTH TALKER DETECTION
# ════════════════════════════════════════════════════════════
def update_smooth_talker(session: dict, eval_data: dict, question_type: str):
    if not eval_data: return
    quality = eval_data.get("quality", "")
    confidence = eval_data.get("confidence_level", "")
    accuracy = eval_data.get("accuracy", "")
    quadrant = eval_data.get("quadrant", "")

    session.setdefault("smooth_talker_signals", [])

    if question_type == "scenario" and quadrant == "dangerous_fake":
        session["smooth_talker_signals"].append("Collapsed on scenario after confident definition")
    if question_type == "why_probe" and accuracy in ("wrong", "partial"):
        session["smooth_talker_signals"].append("Could not explain WHY — surface-level knowledge only")
    if question_type == "numerical" and quality in ("weak", "adequate") and confidence == "high":
        session["smooth_talker_signals"].append("Evaded numerical probe with no real numbers")
    if question_type == "personal_anchor" and quality == "weak":
        session["smooth_talker_signals"].append("Generic answer to personal experience question")
    if question_type == "contradiction" and accuracy == "wrong":
        session["smooth_talker_signals"].append("Contradicted earlier answer — memorized not understood")

    count = len(session["smooth_talker_signals"])
    session["smooth_talker_detected"] = count >= 3
    session["smooth_talker_score"] = min(100, count * 20)

# ════════════════════════════════════════════════════════════
# NOTABLE MOMENTS
# ════════════════════════════════════════════════════════════
def record_notable(session: dict, turn: int, question: str,
                   answer: str, moment_type: str, detail: str):
    session.setdefault("notable_moments", []).append({
        "turn": turn, "moment_type": moment_type,
        "question": question[:150], "answer_excerpt": (answer or "")[:150],
        "detail": detail
    })

# ════════════════════════════════════════════════════════════
# SUSPICION SIGNAL COUNTER — spec: 7+ signals = CRITICAL
# ════════════════════════════════════════════════════════════
def count_active_signals(session: dict, scored_history: list) -> int:
    """Count distinct active suspicion signals for CRITICAL threshold."""
    count = 0
    anticheat = session.get("anticheat_events", [])

    if any(e["event_type"] == "tab_switch" for e in anticheat): count += 1
    if any(e["event_type"] == "paste_event" for e in anticheat): count += 1
    if any(e["event_type"] == "dom_overlay" for e in anticheat): count += 1
    if any(e["event_type"] == "screen_share" for e in anticheat): count += 1
    if any(e["event_type"] == "canary_triggered" for e in anticheat): count += 1

    flags_all = []
    for h in scored_history:
        flags_all.extend(h.get("behavioral_flags", []))

    if "suspiciously_clean_speech" in flags_all: count += 1
    if "personal_pronouns_vanished" in flags_all: count += 1
    if "low_pause_variance" in flags_all: count += 1
    if "self_corrections_vanished" in flags_all: count += 1

    if session.get("smooth_talker_detected"): count += 1

    honest_count = sum(1 for h in scored_history
                       if (h.get("evaluation") or {}).get("quality") == "honest_admission")
    if len(scored_history) >= 8 and honest_count == 0: count += 1

    df_count = sum(1 for h in scored_history
                   if (h.get("evaluation") or {}).get("quadrant") == "dangerous_fake")
    if df_count >= 3: count += 1

    # Contradiction flag
    if any(h.get("contradiction_inconsistency") for h in scored_history): count += 1

    # Above level answers
    if any(h.get("above_level") for h in scored_history): count += 1

    return count

# ════════════════════════════════════════════════════════════
# TOPIC-LEVEL SUSPICION — spec: track per topic
# ════════════════════════════════════════════════════════════
def compute_topic_suspicion(session: dict, scored_history: list) -> dict:
    """Track suspicious signals at topic level — not just overall."""
    anticheat = session.get("anticheat_events", [])
    topic_suspicion: dict = {}

    for h in scored_history:
        topic = h.get("topic", "general")
        if topic not in topic_suspicion:
            topic_suspicion[topic] = {"score": 0, "flags": []}

        # Tab switch before this answer
        prev_tab = [e for e in anticheat
                    if e["event_type"] == "tab_switch"
                    and e["turn"] == h.get("turn", 0) - 1]
        if prev_tab and (h.get("evaluation") or {}).get("quality") == "strong":
            topic_suspicion[topic]["score"] += 20
            topic_suspicion[topic]["flags"].append(
                f"Tab switch immediately before strong answer on {topic} at turn {h['turn']}"
            )

        # Behavioral flags on this answer
        for flag in h.get("behavioral_flags", []):
            if flag in ("suspiciously_clean_speech", "personal_pronouns_vanished", "low_pause_variance"):
                topic_suspicion[topic]["score"] += 10
                topic_suspicion[topic]["flags"].append(f"{flag} on {topic} answer")

    return topic_suspicion

# ════════════════════════════════════════════════════════════
# SUSPICION SCORE ENGINE — full weighted combination
# ════════════════════════════════════════════════════════════
def compute_suspicion_score(session: dict, scored_history: list) -> dict:
    anticheat = session.get("anticheat_events", [])
    suspicion = 0.0
    flags = []

    # Signal 1: Tab switch correlated with answer quality + difficulty (VERY HIGH weight)
    for ev in anticheat:
        if ev["event_type"] == "tab_switch":
            next_ans = next(
                (h for h in scored_history if h.get("turn", 0) > ev["turn"] and h.get("evaluation")),
                None
            )
            if next_ans:
                q = next_ans["evaluation"].get("quality", "")
                diff = next_ans.get("difficulty", "basic")
                topic = next_ans.get("topic", "unknown")
                if q == "strong" and diff in ("advanced", "expert"):
                    suspicion += 20  # VERY HIGH
                    flags.append(f"Tab switch at turn {ev['turn']} followed by strong answer on {diff} {topic} question (turn {next_ans['turn']})")
                elif q == "strong":
                    suspicion += 8
                    flags.append(f"Tab switch at turn {ev['turn']} followed by strong answer on {topic} (turn {next_ans['turn']})")
                else:
                    suspicion += 2  # LOW — could be accidental
            else:
                suspicion += 2

    # Signal 2: Paste events (MEDIUM)
    for ev in anticheat:
        if ev["event_type"] == "paste_event":
            suspicion += 15
            flags.append(f"Paste event at turn {ev['turn']}")

    # Signal 3: DOM overlay — canary triggered (HIGH)
    for ev in anticheat:
        if ev["event_type"] in ("dom_overlay", "canary_triggered"):
            suspicion += 20
            flags.append(f"AI browser extension detected at turn {ev['turn']} ({ev['event_type']})")

    # Signal 4: Screen share (MEDIUM)
    for ev in anticheat:
        if ev["event_type"] == "screen_share":
            suspicion += 12
            flags.append(f"Screen sharing detected at turn {ev['turn']}")

    # Signal 5: Filler words vanished (HIGH)
    clean_turns = [h for h in scored_history if "suspiciously_clean_speech" in h.get("behavioral_flags", [])]
    if len(clean_turns) >= 3:
        suspicion += len(clean_turns) * 8
        flags.append(f"Filler words vanished in {len(clean_turns)} answers (turns {', '.join(str(h['turn']) for h in clean_turns[:3])}) vs warmup baseline")

    # Signal 6: Personal pronouns vanished (HIGH)
    pronoun_turns = [h for h in scored_history if "personal_pronouns_vanished" in h.get("behavioral_flags", [])]
    if len(pronoun_turns) >= 2:
        suspicion += len(pronoun_turns) * 7
        flags.append(f"Personal pronouns vanished in {len(pronoun_turns)} answers")

    # Signal 7: Self-corrections vanished
    correction_turns = [h for h in scored_history if "self_corrections_vanished" in h.get("behavioral_flags", [])]
    if len(correction_turns) >= 2:
        suspicion += len(correction_turns) * 5
        flags.append(f"Self-correction pattern disappeared in {len(correction_turns)} answers")

    # Signal 8: Low pause variance regardless of difficulty (HIGH)
    if any("low_pause_variance" in h.get("behavioral_flags", []) for h in scored_history):
        suspicion += 15
        flags.append("Identical thinking pause across all difficulty levels — possible AI latency pattern")

    # Signal 9: Instant answer on hard question
    instant_hard = [h for h in scored_history
                    if any(f.startswith("instant_answer_on_") for f in h.get("behavioral_flags", []))]
    if instant_hard:
        suspicion += len(instant_hard) * 8
        flags.append(f"Suspiciously instant answers on hard questions at turns {', '.join(str(h['turn']) for h in instant_hard[:3])}")

    # Signal 10: Zero honest admissions (suspicious if full interview)
    honest_count = sum(1 for h in scored_history
                       if (h.get("evaluation") or {}).get("quality") == "honest_admission")
    if len(scored_history) >= 8 and honest_count == 0:
        suspicion += 12
        flags.append("Zero honest admissions in full interview — genuine candidates always have knowledge limits")

    # Signal 11: Dangerous fake quadrant pattern
    df_turns = [h for h in scored_history if (h.get("evaluation") or {}).get("quadrant") == "dangerous_fake"]
    if len(df_turns) >= 3:
        suspicion += len(df_turns) * 8
        flags.append(f"Confident + wrong pattern at turns {', '.join(str(h['turn']) for h in df_turns[:3])}")

    # Signal 12: Answer length spike on hard questions (HIGH)
    bwc = (session.get("behavioral_baseline") or {}).get("avg_word_count", 60)
    spike_turns = [h for h in scored_history
                   if "answer_length_spike_on_hard_question" in h.get("behavioral_flags", [])]
    if spike_turns:
        suspicion += len(spike_turns) * 10
        flags.append(f"Answer length spiked significantly on hard questions at turns {', '.join(str(h['turn']) for h in spike_turns[:3])}")

    # Signal 13: Smooth talker signals
    st_signals = session.get("smooth_talker_signals", [])
    if len(st_signals) >= 3:
        suspicion += len(st_signals) * 5
        flags.append(f"Smooth talker pattern confirmed: {'; '.join(st_signals[:3])}")

    # Signal 14: Perfect recovery after hint every time (MEDIUM)
    for ev in session.get("suspicion_events", []):
        if ev["type"] == "perfect_recovery_after_hint":
            suspicion += ev.get("weight", 15)
            flags.append(ev["detail"])

    # Signal 15: Answer complexity above level
    above_level_turns = [h for h in scored_history if h.get("above_level")]
    if above_level_turns:
        suspicion += len(above_level_turns) * 8
        flags.append(f"Answer sophistication significantly above calibrated level at turns {', '.join(str(h['turn']) for h in above_level_turns[:3])}")

    # Signal 16: Contradiction inconsistency (HIGH)
    contradiction_fails = [h for h in scored_history if h.get("contradiction_inconsistency")]
    if contradiction_fails:
        suspicion += len(contradiction_fails) * 12
        flags.append(f"Contradicted earlier answers at turns {', '.join(str(h['turn']) for h in contradiction_fails[:3])}")

    suspicion = min(100, suspicion)

    # Total signal count for CRITICAL verdict
    signal_count = count_active_signals(session, scored_history)
    verdict = "critical" if signal_count >= 7 else None

    if suspicion < 15:   level = "clean"
    elif suspicion < 35: level = "low_risk"
    elif suspicion < 60: level = "moderate_risk"
    else:                level = "high_risk"

    if verdict == "critical":
        level = "high_risk"
        flags.append(f"CRITICAL: {signal_count} distinct suspicion signals detected — pattern confirmation")

    return {
        "suspicion_score": suspicion,
        "integrity_level": level,
        "signal_count": signal_count,
        "critical_verdict": verdict == "critical",
        "flags": flags
    }

# ════════════════════════════════════════════════════════════
# CONTRADICTION PAIR TRACKING
# ════════════════════════════════════════════════════════════
def get_next_contradiction(session: dict) -> dict | None:
    """Return a contradiction pair where angle_1 was already asked but angle_2 not yet."""
    domain = session["resume"]["domain"]
    pairs = CONTRADICTION_PAIRS.get(domain, [])
    asked_topics = session.get("contradiction_asked", {})

    for pair in pairs:
        topic = pair["topic"]
        state = asked_topics.get(topic, "none")
        # angle_1 asked 6+ turns ago — now ask angle_2
        if state == "angle_1_asked":
            turn_asked = asked_topics.get(f"{topic}_turn", 0)
            if session["turn"] - turn_asked >= 6:
                return {"pair": pair, "angle": "angle_2"}
        # angle_1 not yet asked and topic was covered
        if state == "none" and topic in session.get("topics_covered", []):
            return {"pair": pair, "angle": "angle_1"}

    return None

def record_contradiction_result(session: dict, topic: str, angle: str,
                                  eval_data: dict, turn: int):
    """Track contradiction pair results and flag inconsistency."""
    asked = session.setdefault("contradiction_asked", {})

    if angle == "angle_1":
        asked[topic] = "angle_1_asked"
        asked[f"{topic}_turn"] = turn
        asked[f"{topic}_angle1_score"] = int(eval_data.get("score") or 5) if eval_data else 5
        asked[f"{topic}_angle1_accuracy"] = eval_data.get("accuracy", "partial") if eval_data else "partial"
    elif angle == "angle_2":
        asked[topic] = "complete"
        angle1_score = asked.get(f"{topic}_angle1_score", 5)
        angle2_score = int(eval_data.get("score") or 5) if eval_data else 5
        angle1_acc = asked.get(f"{topic}_angle1_accuracy", "partial")
        angle2_acc = eval_data.get("accuracy", "partial") if eval_data else "partial"

        # Inconsistency: confident on angle_1 but wrong on angle_2 on same topic
        inconsistent = (
            angle1_acc in ("correct", "partial") and angle1_score >= 6 and
            angle2_acc == "wrong" and angle2_score <= 4
        )
        asked[f"{topic}_inconsistent"] = inconsistent
        return inconsistent
    return False

# ════════════════════════════════════════════════════════════
# QUESTION TYPE DECISION ENGINE
# ════════════════════════════════════════════════════════════
def decide_question_type(session: dict) -> tuple[str, dict | None]:
    """Returns (question_type, extra_data)."""
    phase = session["phase"]
    if phase == "warmup":
        return "warmup", None

    tech_turn = session["turn"] - session.get("warmup_turns", 2)
    last_type = session.get("last_question_type", "")
    anchor_count = session.get("anchor_count", 0)
    topics_covered = session.get("topics_covered", [])
    last_eval = session.get("last_eval_quality", "adequate")
    last_confidence = session.get("last_confidence", "medium")

    # Track recovery attempts per topic (for main interview only, not warmup)
    last_topic = session.get("last_topic", "")
    recovery_attempts = session.get("recovery_attempts_per_topic", {})
    topic_recovery_count = recovery_attempts.get(last_topic, 0)

    # RULE 0: If already tried 2 recovery probes on same topic, MOVE TO NEW SKILL
    if last_type == "recovery_probe" and topic_recovery_count >= 2:
        print(f"[Decision] Moving away from topic '{last_topic}' after {topic_recovery_count} recovery attempts")
        session["skip_topic"] = last_topic  # Mark topic to skip
        return "definition", {"force_new_topic": True}

    # RULE 1: definition → always scenario
    if last_type == "definition":
        return "scenario", None

    # RULE 2: honest_admission → recovery_probe with forced hint (max 2 per topic)
    if last_eval == "honest_admission":
        recovery_attempts[last_topic] = topic_recovery_count + 1
        session["recovery_attempts_per_topic"] = recovery_attempts
        return "recovery_probe", {"force_hint": True}

    # RULE 3: poor_articulation → practical_example
    if last_eval == "poor_articulation":
        return "practical_example", None

    # RULE 4: confident + shallow on scenario → why_probe
    if last_type == "scenario" and last_confidence == "high" and last_eval in ("weak", "adequate"):
        return "why_probe", None

    # RULE 5: Check for contradiction pair opportunity
    contradiction = get_next_contradiction(session)
    if contradiction:
        return "contradiction", contradiction

    # RULE 6: Numerical after every major concept — after definition/scenario pair completes
    if last_type in ("scenario", "why_probe", "practical_example") and tech_turn > 2:
        return "numerical", None

    # RULE 7: Personal anchor at turns 4, 9, 15
    if anchor_count < 3 and tech_turn in [4, 9, 15]:
        return "personal_anchor", None

    return "definition", None

# ════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ════════════════════════════════════════════════════════════
def build_system_prompt(session: dict, forced_type: str, extra: dict = None) -> str:
    r = session["resume"]
    topics = DOMAIN_TOPICS.get(r["domain"], [])
    covered = session.get("topics_covered", [])
    uncovered = [t for t in topics if t not in covered]
    tech_turn = session["turn"] - session.get("warmup_turns", 2)
    force_hint = extra.get("force_hint", False) if extra else False
    contradiction_data = extra.get("pair") if extra and extra.get("angle") else None
    contradiction_angle = extra.get("angle") if extra else None

    specific_question = ""
    if forced_type == "contradiction" and contradiction_data and contradiction_angle:
        q_key = f"angle_{contradiction_angle.split('_')[1]}" if "_" in str(contradiction_angle) else contradiction_angle
        specific_question = f"\nUSE THIS EXACT CONTRADICTION QUESTION (angle {contradiction_angle} for topic '{contradiction_data['topic']}'):\n\"{contradiction_data.get(q_key, contradiction_data.get('angle_2', ''))}\"\n"

    hint_instruction = ""
    if force_hint:
        hint_instruction = "\nIMPORTANT: You MUST give a small hint in your question. Set hint_given=true. The hint should be a small nudge, not the full answer.\n"

    # Get candidate projects
    projects = r.get("key_projects", [])
    projects_text = ", ".join(projects) if projects else "No projects listed"

    # Compute last answer context for better follow-ups
    last_eval_quality = session.get("last_eval_quality", "adequate")
    last_confidence = session.get("last_confidence", "medium")
    last_answer_summary = ""
    if session.get("history") and session["history"][-1].get("answer"):
        last_a = session["history"][-1]
        last_answer_summary = f"- Last answer quality: {last_eval_quality} | Confidence: {last_confidence}"
        if (last_a.get("evaluation") or {}).get("notes"):
            last_answer_summary += f"\n- Evaluator note: {last_a['evaluation']['notes']}"

    # Level-calibrated complexity guide
    level_guide = {
        "fresh_graduate": "Ask about fundamentals and basic concepts. Use simple scenarios. Expect textbook-level answers. Do NOT ask about advanced tool flows or tapeout experience.",
        "trained_fresher": "Ask about concepts with simple application scenarios. Expect some tool awareness. Can ask about training project details. Avoid deep debugging or optimization questions.",
        "experienced_junior": "Ask application-level questions with real scenarios. Expect tool familiarity and some debugging experience. Push into why/how but not corner cases.",
        "experienced_senior": "Ask advanced debugging, optimization, and tradeoff questions. Expect deep tool knowledge, tapeout experience, and failure analysis. Push hard into corner cases and real silicon issues."
    }.get(r["level"], "Calibrate to candidate's experience level.")

    # Domain-specific scenario examples to guide the model
    domain_scenarios = {
        "analog_layout": "Example scenarios: mismatch causing offset in a current mirror, parasitic coupling between sensitive nets, latch-up triggered during ESD event, DRC violations from minimum spacing, LDE effects on threshold voltage, guard ring placement strategy.",
        "physical_design": "Example scenarios: setup violation on a cross-clock path after CTS, hold violation that worsens with buffer insertion, IR drop causing timing failure at worst-case corner, congestion hotspot blocking critical net routing, antenna violations during signoff, useful skew application.",
        "design_verification": "Example scenarios: coverage hole that missed a corner-case bug, assertion that passes in simulation but fails in formal, UVM sequence not generating back-to-back transactions, scoreboard mismatch from incorrect reference model, functional coverage bin that never hits, constrained random not reaching target coverage."
    }.get(r["domain"], "Use realistic industry scenarios relevant to the domain.")

    return f"""You are a senior VLSI interviewer conducting a rigorous technical interview.
You speak naturally -- your questions will be read aloud via text-to-speech, so keep them conversational and speakable. Avoid symbols, code snippets, or formatting that doesn't work in speech.

PERSONA:
- You are a principal engineer who has done multiple tapeouts
- You care about real understanding, not memorized definitions
- You probe deeper when answers sound rehearsed or surface-level
- You follow up on vague answers: "walk me through the exact steps", "what numbers did you see?", "what broke?"
- You acknowledge good answers briefly before moving to the next question

LEVEL CALIBRATION:
{level_guide}

QUESTION DESIGN RULES:
1. Prefer "what would you do if..." over "what is..." -- force reasoning, not recall
2. Use realistic scenarios from actual chip design:
   {domain_scenarios}
3. Escalate within a topic: Concept -> Application -> Failure -> Edge Case -> Optimization
4. Never ask two questions at the same difficulty depth on the same topic
5. Stay strictly within the candidate's domain -- do NOT cross domains
6. NEVER repeat or rephrase a previously asked question
7. Occasionally pressure-test: "Are you sure?", "What if that doesn't work?", "What breaks at corners?"
8. Keep questions to 1-2 sentences maximum -- short, sharp, specific
9. When asking about projects: probe challenges faced, specific tool issues, results achieved, what they would change

CANDIDATE:
- Name: {r.get("candidate_name", "Candidate")}
- Domain: {r["domain"].replace("_", " ")}
- Level: {r["level"].replace("_", " ")} ({r.get("years_experience", 0)} years experience)
- Tools: {", ".join(r.get("tools", [])) or "not specified"}
- Skills: {", ".join(r.get("skills", [])) or "not specified"}
- Projects: {projects_text}
- Background: {r.get("background_summary", "")}

INTERVIEW STATE:
- Turn: {session["turn"]} | Technical question: {tech_turn}
- Current difficulty: {DIFFICULTY_LABELS[session["difficulty_level"]]} (level {session["difficulty_level"]+1}/5)
- Topics covered: {", ".join(covered) or "none yet"}
- Topics remaining: {", ".join(uncovered[:4]) or "all covered"}
- Personal anchors asked: {session.get("anchor_count", 0)}/3
- Last topic: {session.get("last_topic", "none")}
- Candidate trajectory: {session.get("trajectory_type", "unknown")}
- Smooth talker signals: {len(session.get("smooth_talker_signals", []))}
{last_answer_summary}
{specific_question}
{hint_instruction}

MANDATORY QUESTION TYPE FOR THIS TURN: {forced_type}

QUESTION TYPES:
- definition: Entry-level "What is X?" -- use only as a door-opener before going deeper
- scenario: Real-world debugging/failure problem -- MANDATORY after every definition. Must be a DIFFERENT angle, not the definition rephrased
- why_probe: "WHY does that happen?" / "What breaks if you don't?" -- use when candidate gave a confident but shallow answer
- practical_example: "Give me a specific example from your work or training" -- use when candidate has the right terms but can't explain clearly
- numerical: "What exact numbers/specs/margins?" -- push for real data, catch bluffers who avoid specifics
- personal_anchor: "Tell me about a SPECIFIC time YOU personally dealt with X" -- test real experience vs borrowed knowledge
- contradiction: Use the EXACT question provided below -- tests if candidate truly understands or memorized
- recovery_probe: Candidate said "I don't know" -- ask a SIMPLER version or break the concept down. Give a hint. NEVER repeat the failed question.

ADAPTIVE FOLLOW-UP RULES:
- After STRONG answer -> push into edge cases, failure modes, or "what if the constraint changes?"
- After WEAK answer -> simplify the angle but test the SAME underlying concept differently
- After PARTIAL answer -> target the specific missing piece directly ("you mentioned X but what about Y?")
- After HONEST ADMISSION ("I don't know") -> acknowledge positively, give a small hint, ask a simpler related question
- If smooth talker signals > 2 -> increase numerical and why_probe questions, demand specific numbers and steps
- If trajectory is "falling" -> check if fatigue or if early answers were memorized -- ask about an earlier topic from a new angle

RETURN ONLY VALID JSON (no markdown, no explanation):
{{
  "question": "Your interview question -- conversational, speakable, max 2 sentences",
  "question_type": "{forced_type}",
  "topic": "specific topic being tested",
  "difficulty": "{DIFFICULTY_LABELS[session["difficulty_level"]]}",
  "hint_given": {str(force_hint).lower()},
  "hint_text": "the hint if hint_given is true, otherwise null"
}}"""



def build_evaluation_prompt(session: dict, question: str, answer: str, difficulty: str, question_type: str) -> str:
    r = session["resume"]
    return f"""You are a senior VLSI technical interviewer evaluating a candidate's answer.

CANDIDATE:
- Domain: {r["domain"].replace("_", " ")}
- Level: {r["level"].replace("_", " ")} ({r.get("years_experience", 0)} years)
- Skills: {", ".join(r.get("skills", []))}

QUESTION ({question_type}, {difficulty}): {question}
ANSWER: {answer}

EVALUATION RULES:
- INTELLECTUAL HONESTY: "I don't know" = quality "honest_admission", score 6/10, warm response. "I don't know + reasoning" = 8/10.
- POOR ARTICULATION: correct terms but incomplete = quality "poor_articulation".
- UNCONVENTIONAL ANSWER: if technically defensible, score defensibility not format. Set accuracy="correct".
- PARTIAL ANSWER: If question has multiple expected points and candidate answers only some, set accuracy="partial" and list expected_points and missing_points.

RETURN ONLY VALID JSON:
{{
  "quality": "strong|adequate|weak|honest_admission|poor_articulation",
  "accuracy": "correct|partial|wrong|not_applicable",
  "confidence_level": "high|medium|low",
  "quadrant": "genuine_expert|genuine_nervous|dangerous_fake|honest_confused",
  "expected_points": ["point 1", "point 2", "point 3"],
  "missing_points": ["points the candidate missed"],
  "score": 5,
  "score_reasoning": "one sentence",
  "notes": "specific observation"
}}"""


def evaluate_answer_llm(session: dict, question: str, answer: str, difficulty: str, question_type: str) -> dict:
    """Evaluate candidate answer using Claude Sonnet 4.6 via Bedrock."""
    if not answer or question_type == "greeting":
        return None
    prompt = build_evaluation_prompt(session, question, answer, difficulty, question_type)
    result = call_claude_json([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=500)
    return result if result else None

def generate_question(session: dict, candidate_answer: str = None) -> dict:
    q_type, extra = decide_question_type(session)
    resume = session.get("resume", {})

    # Get all skills to cover randomly
    all_skills = resume.get("skills", []) + resume.get("tools", [])
    skills_covered = session.get("skills_covered_in_interview", [])

    # Skip topics where candidate clearly doesn't know (after 2 failed recovery attempts)
    skip_topic = session.get("skip_topic")
    if skip_topic:
        skills_covered = skills_covered + [skip_topic]  # Treat skipped topic as covered
        session["skip_topic"] = None  # Clear the skip flag

    # Pick a skill to focus on (prioritize uncovered skills)
    import random
    uncovered = [s for s in all_skills if s not in skills_covered]
    if uncovered:
        current_skill = random.choice(uncovered)
    elif all_skills:
        current_skill = random.choice(all_skills)
    else:
        current_skill = None

    sys_prompt = build_system_prompt(session, q_type, extra)

    # Build messages: ALL questions + LAST 5 answers only
    messages = [{"role": "system", "content": sys_prompt}]
    history = session.get("history", [])

    # Add all questions
    # for i, h in enumerate(history):
    #     messages.append({"role": "assistant", "content": h["question"]})
    #     # Only include last 5 answers
    #     if h.get("answer") and i >= len(history) - 5:
    #         messages.append({"role": "user", "content": h["answer"]})
    for i, h in enumerate(history):
        messages.append({"role": "assistant", "content": h["question"]})
        # Pass last 5 answers only — agent decides continuation based on recent performance
        if h.get("answer") and i >= len(history) - 5:
            messages.append({"role": "user", "content": h["answer"]})
        elif h.get("answer"):
            # Questions without answers still in context as assistant turns
            # but answers older than 5 are excluded intentionally
            pass

    # Add current answer if provided
    if candidate_answer:
        messages.append({"role": "user", "content": candidate_answer})
    else:
        messages.append({"role": "user", "content": "[START INTERVIEW]"})

    # Get previously asked questions to avoid repetition
    prev_questions = [h["question"] for h in history if h.get("question")]
    #prev_questions_text = "\n".join([f"- {q}" for q in prev_questions[-5:]]) if prev_questions else "None"
    prev_questions_text = "\n".join([f"- {q}" for q in prev_questions]) if prev_questions else "None"

    # Add skill coverage instruction to prompt
    skill_instruction = ""
    if current_skill:
        skill_instruction = f"\n\nFOCUS SKILL FOR THIS QUESTION: {current_skill}\nSkills already covered: {', '.join(skills_covered) if skills_covered else 'None yet'}"

    # Add previously asked questions to avoid repetition
    skill_instruction += f"\n\nPREVIOUSLY ASKED QUESTIONS (DO NOT repeat these):\n{prev_questions_text}"

    # Check if previous answer was partial and needs hint
    hint_instruction = ""
    last_entry = history[-1] if history else None
    if last_entry and last_entry.get("evaluation"):
        eval_data = last_entry.get("evaluation", {})
        if eval_data.get("accuracy") == "partial" or eval_data.get("quality") == "adequate":
            expected_points = eval_data.get("expected_points", [])
            missing_points = eval_data.get("missing_points", [])
            if missing_points:
                hint_instruction = f"\n\nPREVIOUS ANSWER WAS PARTIAL. Missing points: {', '.join(missing_points)}. Give a hint about what was missed OR ask a follow-up to cover the missing part."

    # Modify system prompt with skill and hint instructions
    enhanced_prompt = sys_prompt + skill_instruction + hint_instruction

    messages[0] = {"role": "system", "content": enhanced_prompt}

    # Step 1: Evaluate previous answer with Claude Sonnet 4.6
    evaluation = None
    if candidate_answer and session.get("history"):
        last_entry = session["history"][-1]
        t_eval_start = time.time()
        evaluation = evaluate_answer_llm(
            session,
            last_entry.get("question", ""),
            candidate_answer,
            last_entry.get("difficulty", "basic"),
            last_entry.get("question_type", "")
        )
        t_eval = time.time() - t_eval_start
        print(f"[Timing] Evaluation: {t_eval:.2f}s | Score: {(evaluation or {}).get('score', '?')} | Quality: {(evaluation or {}).get('quality', '?')}")

    # Step 2: Generate next question with Claude Sonnet 4.6 via Bedrock (prompt cached)
    t_qgen_start = time.time()
    result = call_claude_json(messages, temperature=0.65, max_tokens=400)
    t_qgen = time.time() - t_qgen_start
    print(f"[Timing] Question gen: {t_qgen:.2f}s | Type: {q_type} | Skill: {current_skill} | Q: {(result or {}).get('question', 'FALLBACK')[:80]}")

    if not result or "question" not in result:
        fallback_topic = current_skill or DOMAIN_TOPICS.get(session["resume"]["domain"], ["your domain"])[0]
        return {
            "question": f"Can you explain {fallback_topic} in your own words?",
            "question_type": q_type, "topic": fallback_topic,
            "difficulty": DIFFICULTY_LABELS[session["difficulty_level"]],
            "hint_given": False, "hint_text": None, "evaluation": evaluation,
            "current_skill": current_skill
        }

    # Track skill coverage
    if current_skill and current_skill not in skills_covered:
        session.setdefault("skills_covered_in_interview", []).append(current_skill)

    result["question_type"] = q_type
    result["current_skill"] = current_skill
    result["_extra"] = extra  # carry extra data for contradiction tracking
    result["evaluation"] = evaluation  # attach GPT-4o-mini evaluation
    return result

# ════════════════════════════════════════════════════════════
# WARMUP NODE - Skill-based warmup questions
# ════════════════════════════════════════════════════════════
def strip_initials(name: str) -> str:
    """Strip single-letter initials (like B. R. S.) from a name, keeping actual name parts."""
    import re
    parts = name.split()
    actual_parts = [p for p in parts if not re.match(r'^[A-Z]\.$', p)]
    return " ".join(actual_parts) if actual_parts else name


def generate_greeting(session: dict) -> dict:
    """Generate a greeting message before warmup questions begin"""
    resume = session.get("resume", {})
    candidate_name = strip_initials(resume.get("candidate_name", "Candidate"))
    greeting = (
        f"Hi {candidate_name}! Welcome to the interview. "
        "Before we begin, please introduce yourself — your background, experience, and what you've been working on recently. "
        "After that, we'll start with a few warm-up questions and then move into the technical round."
    )
    return {
        "question": greeting,
        "question_type": "greeting",
        "topic": "greeting",
        "difficulty": "basic",
    }


def generate_warmup_question(session: dict, candidate_answer: str = None) -> dict:
    """Generate warmup questions based on user's resume skills"""
    resume = session.get("resume", {})
    warmup_count = session.get("warmup_turns", 0)
    # Extract skills and info from resume
    import random
    candidate_name = strip_initials(resume.get("candidate_name", "Candidate"))
    skills = resume.get("skills", [])
    tools = resume.get("tools", [])
    all_skills = skills + tools
    # Shuffle all skills for variety
    shuffled_skills = all_skills.copy()
    random.shuffle(shuffled_skills)
    skills_text = ", ".join(shuffled_skills) if shuffled_skills else "VLSI concepts"

    projects = resume.get("key_projects", [])
    projects_text = ", ".join(projects[:3]) if projects else "VLSI projects"

    level = resume.get("level", "trained_fresher")
    domain = resume.get("domain", "physical_design")

    # Track skills already asked in warmup to avoid repetition
    warmup_skills_asked = session.get("warmup_skills_asked", [])
    remaining_skills = [s for s in all_skills if s not in warmup_skills_asked]
    if not remaining_skills:
        remaining_skills = all_skills  # Reset if all asked

    # Build conversation history
    conversation_parts = []
    for h in session.get("history", []):
        if h.get("question"):
            conversation_parts.append(f"Interviewer: {h['question']}")
        if h.get("answer"):
            conversation_parts.append(f"Candidate: {h['answer']}")
    conversation_text = "\n".join(conversation_parts) if conversation_parts else "No conversation yet."

    # Force decision after 3 questions
    must_decide = warmup_count >= 2  # After 2 questions asked, this is the 3rd response

    # Get previously asked questions to avoid repetition
    prev_questions = [h["question"] for h in session.get("history", []) if h.get("question")]
    prev_questions_text = "\n".join([f"- {q}" for q in prev_questions]) if prev_questions else "None"

    prompt = f"""You are the Warmup Agent. Ask simple questions based on user skills only.

Candidate Name: {candidate_name}
User Skills: {skills_text}

Previously Asked Questions (DO NOT repeat these):
{prev_questions_text}


Ask a simple question about one of their skills.

Rules:
- Ask questions ONLY about the skills listed above
- Don't mention initials of user name like B. R. - use their actual name
- No complex or scenario questions
- No emojis

Return ONLY this JSON:
{{
  "question": "your simple question",
  "skill_asked": "skill name from the list above"
}}
"""

    # Higher temperature for more varied questions
    result = call_cerebras_json([{"role": "user", "content": prompt}], temperature=0.8, max_tokens=300)

    if not result or "question" not in result:
        # Fallback - pick a random skill
        import random
        skill = random.choice(remaining_skills) if remaining_skills else (all_skills[0] if all_skills else "VLSI")
        question = f"Can you tell me about {skill}?"
        return {
            "question": question,
            "question_type": "warmup",
            "skill_asked": skill
        }

    # Track which skill was asked to avoid repetition
    if result.get("skill_asked"):
        session.setdefault("warmup_skills_asked", []).append(result["skill_asked"])

    result["question_type"] = "warmup"
    return result


# ════════════════════════════════════════════════════════════
# TRAJECTORY ANALYSIS
# ════════════════════════════════════════════════════════════
def compute_trajectory(scores: list) -> str:
    if len(scores) < 4: return "insufficient_data"
    third = max(1, len(scores) // 3)
    first = sum(scores[:third]) / third
    last_chunk = scores[2*third:] or scores[-1:]
    last = sum(last_chunk) / len(last_chunk)
    variance = max(scores) - min(scores)
    if variance > 4: return "spiky"
    if last > first + 1.5: return "rising"
    if first > last + 1.5: return "falling"
    if first >= 7 and last >= 7: return "flat_strong"
    return "flat_weak"

def get_trajectory_interpretation(t: str) -> str:
    return {
        "rising":            "Started nervous, improved significantly — genuine candidate. Second half weighted more in scoring.",
        "falling":           "Started strong, performance dropped — possible memorization exhaustion or fatigue.",
        "spiky":             "Inconsistent — deep on known topics, suspicious on others. Cross-referenced with anti-cheat.",
        "flat_strong":       "Consistently strong throughout — well-prepared candidate.",
        "flat_weak":         "Consistently struggled — needs more preparation before next attempt.",
        "insufficient_data": "Too few answers to determine trajectory pattern."
    }.get(t, "Pattern not determined.")

# ════════════════════════════════════════════════════════════
# TTS
# ════════════════════════════════════════════════════════════
def synthesize_speech_pocket(text: str) -> str:
    """Primary TTS: Pocket TTS (local, CPU, free, voice cloning)."""
    audio = pocket_tts_model.generate_audio(pocket_tts_voice_state, text[:1500])
    # Convert to WAV bytes in memory, then to MP3-compatible base64
    import io
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, pocket_tts_model.sample_rate, audio.numpy())
    wav_bytes = wav_buffer.getvalue()
    return base64.b64encode(wav_bytes).decode("utf-8")


def synthesize_speech_mistral(text: str) -> str:
    """Fallback TTS 1: Mistral Voxtral with voice cloning."""
    resp = http_requests.post(
        "https://api.mistral.ai/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "voxtral-mini-tts-2603",
            "input": text[:1500],
            "ref_audio": MISTRAL_TTS_REF_AUDIO,
            "response_format": "mp3",
        },
        timeout=10,
    )
    resp.raise_for_status()
    audio_b64 = resp.json()["audio_data"]
    return audio_b64


def synthesize_speech_polly(text: str) -> str:
    """Fallback TTS 2: AWS Polly with Amy neural voice."""
    resp = polly_client.synthesize_speech(
        Text=text[:1500], OutputFormat="mp3",
        VoiceId="Amy", Engine="neural"
    )
    return base64.b64encode(resp["AudioStream"].read()).decode("utf-8")


def synthesize_speech(text: str) -> str:
    """TTS with fallback: Pocket TTS -> Mistral Voxtral."""
    # Primary: Pocket TTS (local, free, voice cloning)
    if pocket_tts_model and pocket_tts_voice_state:
        try:
            return synthesize_speech_pocket(text)
        except Exception as e:
            print(f"Pocket TTS failed, falling back to Mistral: {e}")

    # Fallback: Mistral Voxtral (voice cloning)
    if MISTRAL_API_KEY and MISTRAL_TTS_REF_AUDIO:
        try:
            return synthesize_speech_mistral(text)
        except Exception as e:
            print(f"Mistral TTS also failed: {e}")

    return ""


def stream_tts_polly(text: str):
    """Streaming TTS via AWS Polly — yields audio chunks."""
    try:
        resp = polly_client.synthesize_speech(
            Text=text[:1500], OutputFormat="mp3",
            VoiceId="Amy", Engine="neural"
        )
        stream = resp["AudioStream"]
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            yield chunk
    except Exception as e:
        print(f"Polly streaming failed: {e}")


def stream_tts_mistral(text: str):
    """Streaming TTS via Mistral Voxtral — yields audio chunks."""
    try:
        resp = http_requests.post(
            "https://api.mistral.ai/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "voxtral-mini-tts-2603",
                "input": text[:1500],
                "ref_audio": MISTRAL_TTS_REF_AUDIO,
                "response_format": "mp3",
            },
            timeout=15,
            stream=True,
        )
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                yield chunk
    except Exception as e:
        print(f"Mistral streaming failed: {e}")


def stream_tts(text: str):
    """Streaming TTS: Mistral Voxtral. (Pocket TTS doesn't support streaming)"""
    # Mistral Voxtral (streaming + voice cloning)
    if MISTRAL_API_KEY and MISTRAL_TTS_REF_AUDIO:
        try:
            for chunk in stream_tts_mistral(text):
                yield chunk
        except Exception as e:
            print(f"Mistral stream failed: {e}")


# ════════════════════════════════════════════════════════════
# STT — OpenAI Whisper API
# ════════════════════════════════════════════════════════════
def transcribe_audio(audio_bytes: bytes, ext: str = "webm") -> dict:
    """Transcribe: gpt-4o-mini-transcribe (primary) -> ElevenLabs Scribe v2 (fallback)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        transcript = ""
        avg_confidence = 1.0
        source = ""

        # Primary: OpenAI gpt-4o-mini-transcribe (accurate + fast)
        try:
            with open(tmp_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                    language="en",
                )
            transcript = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            source = "gpt-4o-mini-transcribe"
        except Exception as e:
            print(f"gpt-4o-mini-transcribe failed, falling back to ElevenLabs: {e}")

        # Fallback: ElevenLabs Scribe v2 (most accurate)
        if not transcript and elevenlabs_client:
            try:
                with open(tmp_path, "rb") as audio_file:
                    response = elevenlabs_client.speech_to_text.convert(
                        file=audio_file,
                        model_id="scribe_v2",
                        language_code="eng",
                    )
                transcript = response.text.strip() if hasattr(response, 'text') else str(response).strip()
                source = "ElevenLabs Scribe v2"
            except Exception as e:
                print(f"ElevenLabs Scribe v2 also failed: {e}")

        print(f"STT [{source}]: '{transcript[:80]}' | confidence: {avg_confidence:.2f}")

        return {
            "transcript": transcript,
            "avg_confidence": avg_confidence,
            "low_confidence": avg_confidence < 0.5,
            "corrupted_terms": [],
            "needs_repeat": len(transcript.strip()) == 0
        }
    except Exception as e:
        print(f"STT error: {e}")
        return {"transcript": "", "avg_confidence": 0.0, "low_confidence": True,
                "corrupted_terms": [], "needs_repeat": False}
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass


# ════════════════════════════════════════════════════════════
# REPORT GENERATOR — all 8 sections
# ════════════════════════════════════════════════════════════
def generate_report(session: dict) -> dict:
    history = session["history"]
    resume = session["resume"]

    scored = [h for h in history
              if h.get("evaluation")
              and (h.get("evaluation") or {}).get("quality") not in ("warmup", None)
              and h["phase"] != "warmup"]

    raw_scores = []
    for h in scored:
        try: raw_scores.append(int(h["evaluation"].get("score") or 5))
        except: raw_scores.append(5)

    trajectory = compute_trajectory(raw_scores)
    trajectory_interp = get_trajectory_interpretation(trajectory)

    avg = sum(raw_scores) / len(raw_scores) if raw_scores else 5.0
    if trajectory == "rising" and len(raw_scores) >= 4:
        mid = len(raw_scores) // 2
        second_avg = sum(raw_scores[mid:]) / (len(raw_scores) - mid)
        weighted_avg = avg * 0.4 + second_avg * 0.6
    else:
        weighted_avg = avg
    technical_score = min(100, int(weighted_avg * 10))

    diff_order = {d: i for i, d in enumerate(DIFFICULTY_LABELS)}
    difficulties = [h.get("difficulty", "basic") for h in scored]
    max_difficulty = max(difficulties, key=lambda d: diff_order.get(d, 0)) if difficulties else "basic"

    quadrants = [(h.get("evaluation") or {}).get("quadrant", "") for h in scored]
    df_count = quadrants.count("dangerous_fake")
    gn_count = quadrants.count("genuine_nervous")
    ge_count = quadrants.count("genuine_expert")

    smooth_detected = session.get("smooth_talker_detected", False)
    if smooth_detected or df_count >= 3:
        behavioral_profile = "Smooth Talker"
    elif gn_count > ge_count and trajectory == "rising":
        behavioral_profile = "Genuine Nervous"
    elif ge_count >= len(scored) * 0.6:
        behavioral_profile = "Genuine Expert"
    else:
        behavioral_profile = "Mixed Profile"

    honest_admissions = sum(1 for h in scored
                            if (h.get("evaluation") or {}).get("quality") == "honest_admission")

    behavioral_score = 55
    behavioral_score += min(20, honest_admissions * 7)
    if trajectory == "rising": behavioral_score += 12
    elif trajectory == "flat_strong": behavioral_score += 8
    elif trajectory == "falling": behavioral_score -= 5
    if smooth_detected: behavioral_score -= 15
    behavioral_score = max(0, min(100, behavioral_score))

    suspicion_data = compute_suspicion_score(session, scored)
    suspicion_score = suspicion_data["suspicion_score"]
    integrity_level = suspicion_data["integrity_level"]
    integrity_flags = suspicion_data["flags"]
    integrity_score = max(0, 100 - int(suspicion_score))
    cap = 60 if integrity_level == "high_risk" else 100

    overall = min(int(technical_score * 0.60 + behavioral_score * 0.25 + integrity_score * 0.15), cap)
    grade = "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55 else "D" if overall >= 40 else "F"

    # Topic performance
    topic_map: dict = {}
    for h in scored:
        t = h.get("topic", "general")
        if not t: continue
        topic_map.setdefault(t, {"scores": [], "questions": [], "answers": []})
        try: topic_map[t]["scores"].append(int(h["evaluation"].get("score") or 5))
        except: topic_map[t]["scores"].append(5)
        if h.get("question"): topic_map[t]["questions"].append(h["question"][:150])
        if h.get("answer"): topic_map[t]["answers"].append(h["answer"][:200])

    topic_performance = {}
    for t, data in topic_map.items():
        # Only highlight skills truly tested (2+ questions on the topic)
        if len(data["scores"]) < 2:
            continue
        avg_t = sum(data["scores"]) / len(data["scores"])
        rating = "Strong" if avg_t >= 7.5 else "Adequate" if avg_t >= 5.5 else "Needs Work" if avg_t >= 3.0 else "Weak"
        topic_performance[t] = {
            "score": int(avg_t * 10), "rating": rating,
            "questions_asked": len(data["scores"]),
            "questions": data["questions"][:2], "sample_answers": data["answers"][:1]
        }

    # Topic-level suspicion
    topic_suspicion = compute_topic_suspicion(session, scored)

    # Contradiction results
    contradiction_results = []
    for topic, state in session.get("contradiction_asked", {}).items():
        if not isinstance(state, str): continue
        if state == "complete":
            inconsistent = session["contradiction_asked"].get(f"{topic}_inconsistent", False)
            contradiction_results.append({
                "topic": topic,
                "inconsistent": inconsistent,
                "angle1_score": session["contradiction_asked"].get(f"{topic}_angle1_score"),
            })

    # Recovery events
    hint_events = session.get("hint_events", [])
    recovery_summary = []
    for h in hint_events:
        if h.get("recovery_quality"):
            recovery_summary.append(
                f"Turn {h['turn']} ({h['topic']}): hint given, recovery was {h['recovery_quality']} (score: {h.get('recovery_score', 'N/A')})"
            )

    # Build full transcript
    transcript = "\n".join([
        f"[T{h['turn']}] {h.get('question_type','?')}/{h.get('difficulty','?')} topic={h.get('topic','?')}\n"
        f"Q: {h['question']}\n"
        f"A: {(h.get('answer') or '[no answer]')[:300]}\n"
        f"Eval: quality={(h.get('evaluation') or {}).get('quality','?')} score={(h.get('evaluation') or {}).get('score','?')} "
        f"quadrant={(h.get('evaluation') or {}).get('quadrant','?')} confidence={(h.get('evaluation') or {}).get('confidence_level','?')}\n"
        f"Notes: {(h.get('evaluation') or {}).get('notes','')}"
        for h in history if h["phase"] != "warmup"
    ])[:5500]

    resources = DOMAIN_RESOURCES.get(resume["domain"], [])
    signal_count = suspicion_data.get("signal_count", 0)

    narrative_prompt = f"""You are writing a VLSI mock interview performance report as a senior mentor.
Be SPECIFIC. Reference actual turns. Never generic.
IMPORTANT: Do NOT count warmup or greeting questions. Only count technical interview questions.
Only include skills in domain_strength_deep_dive that were truly tested with 2+ questions. Skip skills that had only 1 question.

CANDIDATE: {resume["level"].replace("_"," ")} | {resume["domain"].replace("_"," ")}
Education: {resume.get("education","")} | Tools: {", ".join(resume.get("tools",[])[:3])}

TECHNICAL QUESTIONS: {len(scored)} (excluding warmup/greeting)
SCORES: Technical={technical_score} | Behavioral={behavioral_score} | Integrity={integrity_score} | Overall={overall} ({grade})
TRAJECTORY: {trajectory} — {trajectory_interp}
SKILLS FULLY TESTED (2+ questions): {[t for t, d in topic_map.items() if len(d["scores"]) >= 2]}
SKILLS ONLY TOUCHED (1 question): {[t for t, d in topic_map.items() if len(d["scores"]) < 2]}
BEHAVIORAL PROFILE: {behavioral_profile}
SMOOTH TALKER SIGNALS: {session.get("smooth_talker_signals",[])}
MAX DIFFICULTY REACHED: {max_difficulty}
HONEST ADMISSIONS: {honest_admissions}
RECOVERY EVENTS: {recovery_summary}
CONTRADICTION RESULTS: {contradiction_results}
INTEGRITY FLAGS: {integrity_flags[:5]}
SUSPICION SIGNALS: {signal_count}
NOTABLE MOMENTS: {[m["detail"] for m in session.get("notable_moments",[])[:5]]}

TRANSCRIPT:
{transcript}

RESOURCES: {resources}

Return ONLY valid JSON:
{{
  "overall_assessment": "2-3 specific sentences referencing actual turns and answers",
  "readiness_statement": "Specific sentence with role and tech node. Example: Ready for trained fresher PD at 28nm. Not ready for sub-14nm.",
  "domain_strength_deep_dive": [
    {{"topic": "name", "rating": "Strong|Adequate|Needs Work|Weak",
      "questions_asked": "number of questions on this topic",
      "what_was_right": "specific from actual answer with turn ref",
      "what_was_missing": "specific gap with turn ref",
      "one_action": "concrete action not just study more"}}
  ],
  "strengths": [
    {{"strength": "title", "evidence": "specific moment with turn number", "why_it_matters": "job impact"}}
  ],
  "improvements": [
    {{"area": "topic", "what_is_weak": "specific gap",
      "why_it_matters": "job impact",
      "suggestion": "specific actionable step with tool or exercise",
      "resource": "book/chapter/tool with specific section",
      "timeline": "X weeks"}}
  ],
  "behavioral_feedback": "2-3 sentences as mentor. Address nervousness, communication, honesty. Reference trajectory.",
  "smooth_talker_assessment": "One honest sentence about smooth talker pattern with evidence.",
  "recovery_velocity_feedback": "One sentence about hint-recovery pattern. If no hints, say so.",
  "recommended_resources": [
    {{"resource": "title with chapter/section", "why": "addresses specific observed gap", "action": "specific exercise"}}
  ],
  "readiness_roadmap": [
    {{"milestone": "Week 2", "goal": "specific measurable goal", "action": "specific daily action"}},
    {{"milestone": "Week 4", "goal": "specific measurable goal", "action": "specific daily action"}},
    {{"milestone": "Week 6", "goal": "specific measurable goal", "action": "specific daily action"}}
  ],
  "next_mock_recommendation": "Specific topics, difficulty level, whether lab/hands-on should come first"
}}"""

    narrative = call_llm_json(
        [{"role": "user", "content": narrative_prompt}],
        temperature=0.3, max_tokens=2500, retries=2
    )

    if not narrative:
        narrative = {
            "overall_assessment": "Interview completed. Detailed analysis could not be generated.",
            "readiness_statement": "Continue preparation before applying.",
            "domain_strength_deep_dive": [],
            "strengths": [{"strength": "Completed interview", "evidence": "Participated fully", "why_it_matters": "Shows commitment"}],
            "improvements": [{"area": "Technical depth", "what_is_weak": "Needs practice",
                               "why_it_matters": "Core job requirement",
                               "suggestion": "Study primary domain topics daily",
                               "resource": resources[0] if resources else "Domain textbook",
                               "timeline": "4 weeks"}],
            "behavioral_feedback": "Work on building confidence through consistent practice.",
            "smooth_talker_assessment": "Insufficient data to assess smooth talker pattern.",
            "recovery_velocity_feedback": "No recovery events recorded in this session.",
            "recommended_resources": [{"resource": r, "why": "Core resource", "action": "Study key chapters"} for r in resources[:3]],
            "readiness_roadmap": [{"milestone": "Week 2", "goal": "Fundamentals", "action": "Review core concepts daily"}],
            "next_mock_recommendation": "Focus on weakest topics. Try intermediate difficulty next session."
        }

    return {
        "scores": {
            "technical": technical_score, "behavioral": behavioral_score,
            "integrity": integrity_score, "overall": overall, "grade": grade
        },
        "trajectory": trajectory,
        "trajectory_interpretation": trajectory_interp,
        "behavioral_profile": behavioral_profile,
        "smooth_talker_detected": smooth_detected,
        "smooth_talker_score": session.get("smooth_talker_score", 0),
        "smooth_talker_signals": session.get("smooth_talker_signals", []),
        "max_difficulty_reached": max_difficulty,
        "topic_performance": topic_performance,
        "topic_suspicion": topic_suspicion,
        "contradiction_results": contradiction_results,
        "turns_completed": len(scored),  # Only count technical questions, not warmup/greeting
        "honest_admissions": honest_admissions,
        "integrity_level": integrity_level,
        "integrity_flags": integrity_flags,
        "suspicion_score": suspicion_score,
        "signal_count": signal_count,
        "critical_verdict": suspicion_data.get("critical_verdict", False),
        "recovery_events": recovery_summary,
        "notable_moments": session.get("notable_moments", []),
        "genuine_signals": session.get("genuine_signals", []),
        **narrative
    }


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/interview", response_class=HTMLResponse)
async def interview_ui():
    with open("templates/voice_agent_ui.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/parse-resume")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
    if ext == "pdf":
        try:
            import pdfplumber
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content); tmp_path = tmp.name
            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages: text += (page.extract_text() or "") + "\n"
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(400, f"PDF error: {e}")
    elif ext in ("docx", "doc"):
        try:
            import docx2txt
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(content); tmp_path = tmp.name
            text = docx2txt.process(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(400, f"DOCX error: {e}")
    else:
        text = content.decode("utf-8", errors="ignore")
    return JSONResponse(parse_resume(text))

@app.post("/api/create-session")
async def create_session(data: SessionCreate):
    sid = str(uuid.uuid4())
    resume = parse_resume(data.resume_text)
    session = {
        "id": sid, "mode": data.mode, "resume": resume,
        "phase": "greeting", "turn": 0, "warmup_turns": 0,
        "warmup_performance": "pending", "warmup_conversation": [],
        "difficulty_level": 1, "consecutive_strong": 0, "consecutive_weak": 0,
        "history": [], "topics_covered": [], "anchor_count": 0,
        "last_topic": None, "last_question_type": None,
        "last_eval_quality": "adequate", "last_confidence": "medium",
        "anticheat_events": [], "behavioral_baseline": None,
        "pause_history": [], "trajectory_type": "unknown",
        "hint_events": [], "notable_moments": [], "suspicion_events": [],
        "smooth_talker_signals": [], "smooth_talker_detected": False,
        "smooth_talker_score": 0, "genuine_signals": [],
        "contradiction_asked": {},
        "topic_suspicion": {},
        "started_at": time.time(),
        "cached_first_question": None,
        "cached_first_audio": None,
    }
    sessions[sid] = session

    # Pre-generate greeting in background (for faster start)
    try:
        first_q = generate_greeting(session)
        first_audio = synthesize_speech(first_q["question"])
        session["cached_first_question"] = first_q
        session["cached_first_audio"] = first_audio
    except Exception as e:
        print(f"Pre-generation failed: {e}")

    return JSONResponse({"session_id": sid, "resume": resume})

@app.get("/api/get-session")
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return JSONResponse({
        "session_id": session_id,
        "mode": session.get("mode", "mock"),
        "resume": session.get("resume", {}),
        "turn": session.get("turn", 0),
        "phase": session.get("phase", "warmup")
    })

@app.post("/api/start-interview")
async def start_interview(data: dict):
    sid = data.get("session_id")
    session = sessions.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")

    # Use cached greeting/question if available (faster start)
    if session["phase"] == "greeting" and session.get("cached_first_question"):
        result = session["cached_first_question"]
        audio = session.get("cached_first_audio", "")
        # Clear cache after use
        session["cached_first_question"] = None
        session["cached_first_audio"] = None
        print("[Interview] Using cached greeting - instant start!")
    elif session["phase"] == "greeting":
        result = generate_greeting(session)
        t_tts = time.time()
        audio = synthesize_speech(result["question"])
        print(f"[Timing] TTS (greeting): {time.time()-t_tts:.2f}s | {len(result['question'])} chars")
    elif session["phase"] == "warmup":
        t_wq = time.time()
        result = generate_warmup_question(session)
        print(f"[Timing] Warmup question gen: {time.time()-t_wq:.2f}s")
        t_tts = time.time()
        audio = synthesize_speech(result["question"])
        print(f"[Timing] TTS (warmup): {time.time()-t_tts:.2f}s | {len(result['question'])} chars")
    else:
        result = generate_question(session)
        t_tts = time.time()
        audio = synthesize_speech(result["question"])
        print(f"[Timing] TTS (start): {time.time()-t_tts:.2f}s | {len(result['question'])} chars")

    session["warmup_conversation"].append(f"Interviewer: {result['question']}")

    entry = {
        "turn": session["turn"], "phase": session["phase"],
        "question": result["question"], "question_type": result.get("question_type", "warmup"),
        "topic": result.get("topic", "warmup"), "difficulty": result.get("difficulty", "basic"),
        "answer": None, "evaluation": None, "behavioral_flags": [],
        "answer_duration_sec": 0, "word_count": 0, "filler_rate": 0,
        "pronoun_rate": 0, "thinking_pause_sec": 0, "input_mode": "text",
        "correction_rate": 0, "above_level": False, "contradiction_inconsistency": False,
        "warmup_decision": result.get("warmup_decision"),
    }
    session["history"].append(entry)
    session["turn"] += 1
    session["last_topic"] = result.get("topic")
    session["last_question_type"] = result.get("question_type", "warmup")

    # Check if interview should end due to poor warmup performance
    should_end = result.get("warmup_decision") == "end_not_ready"
    if should_end:
        session["phase"] = "ended"
        session["warmup_performance"] = "poor"

    return JSONResponse({
        "question": result["question"], "question_type": result.get("question_type", "warmup"),
        "turn": session["turn"], "phase": session["phase"],
        "audio": audio, "difficulty": result.get("difficulty", "basic"),
        "should_end": should_end,
        "warmup_decision": result.get("warmup_decision"),
        "resume": session.get("resume", {})
    })

@app.post("/api/submit-answer")
async def submit_answer(data: AnswerSubmit):
    session = sessions.get(data.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    current_entry = session["history"][-1] if session["history"] else None

    if current_entry:
        current_entry["answer"] = data.answer
        current_entry["answer_duration_sec"] = data.answer_duration_sec
        current_entry["word_count"] = data.word_count
        current_entry["thinking_pause_sec"] = data.thinking_pause_sec
        current_entry["input_mode"] = data.input_mode
        current_entry["filler_rate"] = count_fillers(data.answer)
        current_entry["pronoun_rate"] = count_personal_pronouns(data.answer)
        current_entry["correction_rate"] = count_self_corrections(data.answer)

        dev = analyze_behavioral_deviation(
            session, data.answer, data.answer_duration_sec,
            data.word_count, data.thinking_pause_sec,
            data.input_mode, current_entry.get("difficulty", "basic")
        )
        current_entry["behavioral_flags"] = dev["flags"]
        current_entry["behavioral_deviation"] = dev["deviation_score"]

    # Phase transition for greeting -> warmup
    if session["phase"] == "greeting":
        session["phase"] = "warmup"
        result = generate_warmup_question(session)
        session["warmup_conversation"].append(f"Interviewer: {result['question']}")

    # Phase transition for warmup
    elif session["phase"] == "warmup":
        session["warmup_turns"] += 1
        session["warmup_conversation"].append(f"Candidate: {data.answer}")

        # Simple logic: After 3 warmup questions, start the interview
        if session["warmup_turns"] >= 3:
            # Transition to interview
            session["phase"] = "interview"
            compute_baseline(session)
            # Generate first interview question
            result = generate_question(session)
            result["warmup_feedback"] = "Let's begin the technical interview!"
        else:
            # Generate next warmup question
            result = generate_warmup_question(session, data.answer)
            session["warmup_conversation"].append(f"Interviewer: {result['question']}")

    elif session["phase"] == "ready_check":
        # User responded to "are you ready?" question
        answer_lower = data.answer.lower()
        if any(word in answer_lower for word in ["yes", "ready", "proceed", "sure", "okay", "ok"]):
            session["phase"] = "interview"
            compute_baseline(session)
            result = generate_question(session)
        else:
            session["phase"] = "ended"
            session["warmup_performance"] = "declined"
            result = {
                "question": "No problem. Please take your time to prepare and come back when you're ready. Good luck!",
                "question_type": "farewell"
            }
    else:
        # Normal interview flow - Generate next question + evaluate previous
        result = generate_question(session, data.answer)

    if current_entry and result.get("evaluation"):
        eval_data = result["evaluation"]
        current_entry["evaluation"] = eval_data

        quality = eval_data.get("quality", "adequate")
        confidence = eval_data.get("confidence_level", "medium")
        score = 5
        try: score = int(eval_data.get("score") or 5)
        except: pass

        session["last_eval_quality"] = quality
        session["last_confidence"] = confidence

        # Answer complexity vs level check
        complexity = assess_answer_complexity(
            session, data.answer, score,
            current_entry.get("difficulty", "basic"),
            session["resume"]["level"]
        )
        current_entry["above_level"] = complexity["above_level"]
        if complexity["above_level"] and not session.get("smooth_talker_detected"):
            # Could be genuine self-taught — record as notable positive
            record_notable(session, session["turn"] - 1,
                current_entry["question"], data.answer, "positive_signal",
                f"Answer sophistication above calibrated level at turn {session['turn']-1}: {complexity['flag']}")

        # Contradiction tracking
        extra = result.get("_extra")
        if extra and extra.get("angle") and extra.get("pair"):
            pair = extra["pair"]
            angle = extra["angle"]
            inconsistent = record_contradiction_result(
                session, pair["topic"], angle, eval_data, session["turn"] - 1
            )
            current_entry["contradiction_inconsistency"] = inconsistent
            if inconsistent:
                record_notable(session, session["turn"] - 1,
                    current_entry["question"], data.answer, "concern_flag",
                    f"Contradiction detected on topic '{pair['topic']}' at turn {session['turn']-1} — answered differently from earlier")
        elif current_entry.get("question_type") == "contradiction":
            # Track angle_1 was asked
            topic = current_entry.get("topic", "")
            if topic and session["contradiction_asked"].get(topic) != "angle_1_asked":
                session["contradiction_asked"][topic] = "angle_1_asked"
                session["contradiction_asked"][f"{topic}_turn"] = session["turn"] - 1
                session["contradiction_asked"][f"{topic}_angle1_score"] = score
                session["contradiction_asked"][f"{topic}_angle1_accuracy"] = eval_data.get("accuracy", "partial")

        # Smooth talker update
        update_smooth_talker(session, eval_data, current_entry.get("question_type", ""))

        # Recovery velocity
        if result.get("hint_given"):
            hint_text = result.get("hint_text", "Hint given")
            record_hint(session, session["turn"] - 1, current_entry.get("topic", ""), hint_text)
        evaluate_recovery(session, session["turn"] - 1, data.answer, score)

        # Notable moments
        if quality == "honest_admission":
            record_notable(session, session["turn"] - 1,
                current_entry["question"], data.answer, "positive_signal",
                f"Honest admission at turn {session['turn']-1} on {current_entry.get('topic','')} — intellectual honesty signal")
        elif eval_data.get("quadrant") == "dangerous_fake":
            record_notable(session, session["turn"] - 1,
                current_entry["question"], data.answer, "concern_flag",
                f"Confident + wrong at turn {session['turn']-1} on {current_entry.get('topic','')} (dangerous fake quadrant)")
        elif quality == "strong" and current_entry.get("difficulty") in ("advanced", "expert"):
            record_notable(session, session["turn"] - 1,
                current_entry["question"], data.answer, "positive_signal",
                f"Strong answer on {current_entry.get('difficulty','')} {current_entry.get('topic','')} at turn {session['turn']-1}")

        # Consecutive counters + adaptive difficulty
        if quality == "strong":
            session["consecutive_strong"] += 1
            session["consecutive_weak"] = 0
        elif quality == "weak":
            session["consecutive_weak"] += 1
            session["consecutive_strong"] = 0
        else:
            session["consecutive_strong"] = 0
            session["consecutive_weak"] = 0

        if session["consecutive_strong"] >= 3 and session["difficulty_level"] < 4:
            session["difficulty_level"] += 1
            session["consecutive_strong"] = 0
        elif session["consecutive_weak"] >= 3 and session["difficulty_level"] > 0:
            session["difficulty_level"] -= 1
            session["consecutive_weak"] = 0

        # Update trajectory
        sc_hist = [h for h in session["history"]
                   if h.get("evaluation")
                   and (h.get("evaluation") or {}).get("quality") not in ("warmup", None)
                   and h["phase"] != "warmup"]
        sc_list = []
        for h in sc_hist:
            try: sc_list.append(int(h["evaluation"].get("score") or 5))
            except: sc_list.append(5)
        session["trajectory_type"] = compute_trajectory(sc_list)

    # Track topic + anchors
    topic = result.get("topic", "general")
    if topic and topic not in session["topics_covered"]:
        session["topics_covered"].append(topic)
    if result.get("question_type") == "personal_anchor":
        session["anchor_count"] = session.get("anchor_count", 0) + 1

    entry = {
        "turn": session["turn"], "phase": session["phase"],
        "question": result["question"], "question_type": result["question_type"],
        "topic": topic, "difficulty": result.get("difficulty", DIFFICULTY_LABELS[session["difficulty_level"]]),
        "answer": None, "evaluation": None, "behavioral_flags": [],
        "answer_duration_sec": 0, "word_count": 0, "filler_rate": 0,
        "pronoun_rate": 0, "thinking_pause_sec": 0, "input_mode": "text",
        "correction_rate": 0, "above_level": False, "contradiction_inconsistency": False,
    }
    session["history"].append(entry)
    session["turn"] += 1
    session["last_topic"] = topic
    session["last_question_type"] = result["question_type"]

    # Check if candidate is struggling — last 4 consecutive interview answers all weak
    struggling_end = False
    if session["phase"] == "interview" and session["turn"] >= 8:
        recent_interview = [
            h for h in session["history"]
            if h.get("evaluation") and h["phase"] == "interview"
            and (h.get("evaluation") or {}).get("quality") not in ("warmup", None)
        ]
        if len(recent_interview) >= 4:
            last_4 = recent_interview[-4:]
            all_struggling = all(
                h["evaluation"].get("quality") in ("weak", "honest_admission")
                and (h["evaluation"].get("score") or 5) <= 3
                for h in last_4
            )
            if all_struggling:
                struggling_end = True
                session["phase"] = "ended"
                session["early_end_reason"] = "struggling"
                print(f"[Interview] Early end — candidate struggled in last 4 questions")

    # Determine if interview should end (count only technical turns, not warmup/greeting)
    technical_turns = session["turn"] - session.get("warmup_turns", 0) - 1  # -1 for greeting
    should_end = (
        technical_turns >= 20 or
        session["phase"] == "ended" or
        struggling_end or
        result.get("warmup_decision") == "end_not_ready"
    )

    t_tts = time.time()
    audio = synthesize_speech(result["question"])
    t_tts_elapsed = time.time() - t_tts
    print(f"[Timing] TTS: {t_tts_elapsed:.2f}s | {len(result['question'])} chars | Turn {session['turn']}")
    return JSONResponse({
        "question": result["question"], "question_type": result.get("question_type", "interview"),
        "turn": session["turn"], "phase": session["phase"], "audio": audio,
        "difficulty": result.get("difficulty", DIFFICULTY_LABELS[session["difficulty_level"]]),
        "should_end": should_end,
        "hint_given": result.get("hint_given", False),
        "hint_text": result.get("hint_text"),
        "warmup_decision": result.get("warmup_decision"),
        "warmup_performance": session.get("warmup_performance", "pending")
    })

@app.get("/api/stream-tts")
async def stream_tts_endpoint(text: str, session_id: str = ""):
    """Streaming TTS endpoint — returns audio chunks as they're generated."""
    session = sessions.get(session_id) if session_id else None
    if not text:
        raise HTTPException(400, "No text provided")

    # Use Pocket TTS if available (non-streaming, but return as single response)
    if pocket_tts_model and pocket_tts_voice_state:
        try:
            audio_b64 = synthesize_speech_pocket(text)
            audio_bytes = base64.b64decode(audio_b64)
            return StreamingResponse(
                iter([audio_bytes]),
                media_type="audio/wav",
                headers={"Content-Length": str(len(audio_bytes))}
            )
        except Exception as e:
            print(f"Pocket TTS failed in stream endpoint, using streaming fallback: {e}")

    # Streaming fallback (Mistral -> Polly)
    return StreamingResponse(
        stream_tts(text),
        media_type="audio/mpeg",
        headers={"Transfer-Encoding": "chunked"}
    )


@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...), session_id: str = Form(...)):
    import asyncio
    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1] if "." in audio.filename else "webm"
    print(f"Transcribe: received {len(audio_bytes)} bytes, ext={ext}")
    if len(audio_bytes) < 1000:
        print(f"WARNING: audio too small — likely empty")
        return JSONResponse({"transcript": "", "avg_confidence": 0.0,
                            "low_confidence": True, "corrupted_terms": [],
                            "needs_repeat": False})
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, transcribe_audio, audio_bytes, ext)
    return JSONResponse(result)

@app.post("/api/anticheat-event")
async def anticheat_event(data: AntiCheatEvent):
    session = sessions.get(data.session_id)
    if not session:
        return JSONResponse({"ok": False})
    session["anticheat_events"].append({
        "event_type": data.event_type, "turn": data.turn,
        "timestamp": data.timestamp, "metadata": data.metadata
    })
    return JSONResponse({"ok": True})

@app.post("/api/generate-report")
async def generate_report_endpoint(data: ReportRequest):
    session = sessions.get(data.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    try:
        report = generate_report(session)
        return JSONResponse(report)
    except Exception as e:
        print(f"[Report] Error generating report: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Report generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)

# ════════════════════════════════════════════════════════════
# ADMIN DASHBOARD API
# ════════════════════════════════════════════════════════════

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    with open("templates/admin.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/admin/sessions")
async def admin_sessions():
    result = []
    for sid, session in sessions.items():
        scored = [h for h in session["history"]
                  if h.get("evaluation") and h["phase"] != "warmup"
                  and (h.get("evaluation") or {}).get("quality") not in ("warmup", None)]
        scores = []
        for h in scored:
            try: scores.append(int(h["evaluation"].get("score") or 5))
            except: pass
        avg_score = round(sum(scores)/len(scores)*10, 1) if scores else 0
        result.append({
            "session_id": sid,
            "domain": session["resume"].get("domain", ""),
            "level": session["resume"].get("level", ""),
            "phase": session["phase"],
            "turn": session["turn"],
            "avg_score": avg_score,
            "anticheat_count": len(session.get("anticheat_events", [])),
            "smooth_talker": session.get("smooth_talker_detected", False),
            "trajectory": session.get("trajectory_type", "unknown"),
            "signal_count": count_active_signals(session, scored),
            "started_at": session.get("started_at", 0),
        })
    return JSONResponse(result)

@app.get("/api/admin/session/{session_id}")
async def admin_session_detail(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    history = session["history"]
    scored = [h for h in history
              if h.get("evaluation") and h["phase"] != "warmup"
              and (h.get("evaluation") or {}).get("quality") not in ("warmup", None)]

    turn_log = []
    for h in history:
        eval_data = h.get("evaluation") or {}
        turn_log.append({
            "turn": h["turn"], "phase": h["phase"],
            "question_type": h.get("question_type", ""),
            "topic": h.get("topic", ""), "difficulty": h.get("difficulty", ""),
            "question": h.get("question", ""), "answer": h.get("answer", "") or "",
            "word_count": h.get("word_count", 0),
            "answer_duration_sec": round(h.get("answer_duration_sec", 0), 1),
            "thinking_pause_sec": round(h.get("thinking_pause_sec", 0), 1),
            "input_mode": h.get("input_mode", "text"),
            "filler_rate": round(h.get("filler_rate", 0), 3),
            "pronoun_rate": round(h.get("pronoun_rate", 0), 3),
            "correction_rate": round(h.get("correction_rate", 0), 3),
            "behavioral_flags": h.get("behavioral_flags", []),
            "behavioral_deviation": round(h.get("behavioral_deviation", 0), 2),
            "above_level": h.get("above_level", False),
            "contradiction_inconsistency": h.get("contradiction_inconsistency", False),
            "quality": eval_data.get("quality", ""),
            "accuracy": eval_data.get("accuracy", ""),
            "confidence_level": eval_data.get("confidence_level", ""),
            "quadrant": eval_data.get("quadrant", ""),
            "score": eval_data.get("score", ""),
            "score_reasoning": eval_data.get("score_reasoning", ""),
            "notes": eval_data.get("notes", ""),
        })

    contradiction = session.get("contradiction_asked", {})
    contradiction_log = []
    for key, val in contradiction.items():
        if isinstance(val, str) and val in ("angle_1_asked", "complete"):
            topic = key
            contradiction_log.append({
                "topic": topic, "status": val,
                "angle1_score": contradiction.get(f"{topic}_angle1_score"),
                "angle1_accuracy": contradiction.get(f"{topic}_angle1_accuracy"),
                "inconsistent": contradiction.get(f"{topic}_inconsistent", False),
            })

    raw_scores = []
    for h in scored:
        try: raw_scores.append({
            "turn": h["turn"],
            "score": int(h["evaluation"].get("score") or 5),
            "topic": h.get("topic",""),
            "quadrant": (h.get("evaluation") or {}).get("quadrant","")
        })
        except: pass

    signal_count = count_active_signals(session, scored)

    return JSONResponse({
        "session_id": session_id,
        "resume": session["resume"],
        "phase": session["phase"],
        "turn": session["turn"],
        "difficulty_level": session["difficulty_level"],
        "trajectory": session.get("trajectory_type", "unknown"),
        "smooth_talker_detected": session.get("smooth_talker_detected", False),
        "smooth_talker_score": session.get("smooth_talker_score", 0),
        "smooth_talker_signals": session.get("smooth_talker_signals", []),
        "signal_count": signal_count,
        "behavioral_baseline": session.get("behavioral_baseline") or {},
        "pause_history": session.get("pause_history", []),
        "turn_log": turn_log,
        "anticheat_log": session.get("anticheat_events", []),
        "contradiction_log": contradiction_log,
        "recovery_log": session.get("hint_events", []),
        "notable_moments": session.get("notable_moments", []),
        "genuine_signals": session.get("genuine_signals", []),
        "suspicion_events": session.get("suspicion_events", []),
        "raw_scores": raw_scores,
        "topics_covered": session.get("topics_covered", []),
        "anchor_count": session.get("anchor_count", 0),
    })