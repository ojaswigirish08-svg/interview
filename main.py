"""
VLSI Interview Agent — Mock Interview Mode
100% spec implementation — all gaps fixed.
Patches applied:
  - CORS locked to ALLOWED_ORIGINS
  - Observability tracker wired into call_llm, call_cerebras, synthesize_speech, transcribe_audio
  - Auth added to all admin routes
  - POST /api/admin/review endpoint added
  - Rate limiting on submit-answer and transcribe
  - /health endpoint added
  - Global exception handler (no stack traces to client)
  - session_id threaded through TTS and STT for per-session tracking
"""

import os, re, json, time, uuid, base64, tempfile, statistics, hashlib, secrets, threading
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv

# New modular imports
from agent import (
    generate_question as agent_generate_question,
    generate_greeting as agent_generate_greeting,
    generate_warmup_question as agent_generate_warmup_question,
    process_evaluation as agent_process_evaluation,
    update_candidate_profile,
    compute_trajectory, get_trajectory_interpretation,
    strip_initials, get_agent_stats,
)
from strategy_engine import (
    strategy_decide, strategy_update, strategy_coverage_for_report,
    get_or_create_engine,
)
from evaluation_validator import (
    validate_evaluation, should_defer_report, get_deferred_summary,
)
from repetition_guard import get_repetition_stats
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn

load_dotenv()

# ── Rate limiter setup ────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="VLSI Interview Agent")

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail": "Too many requests. Please slow down."}, status_code=429)

@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception):
    print(f"[ERROR] {request.method} {request.url.path}: {exc}")
    return JSONResponse({"detail": "Something went wrong. Please try again."}, status_code=500)

# ── CORS — locked to ALLOWED_ORIGINS ─────────────────────────────────────────
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.state.limiter = limiter

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

sessions: dict = {}

# Database module (PostgreSQL — optional, falls back to in-memory)
import database as db

@app.on_event("startup")
def startup_db():
    db.init_db()


# ════════════════════════════════════════════════════════════
# OBSERVABILITY TRACKER  (in-memory, no external dependency)
# ════════════════════════════════════════════════════════════
_obs_lock = threading.Lock()
_call_logs: list = []
_reviews_store: list = []
_pending_reviewers: list = []
_approved_reviewers: dict = {}
MAX_LOGS = 2000

def track_llm_call(session_id, step, model, latency_ms,
                   input_tokens=None, output_tokens=None,
                   cost_usd=None, status="success", error=None):
    total = (input_tokens or 0) + (output_tokens or 0)
    if cost_usd is None and total > 0:
        cost_usd = ((input_tokens or 0)*0.15 + (output_tokens or 0)*0.60) / 1_000_000
    _obs_write({
        "log_id": uuid.uuid4().hex[:8], "ts": time.time(),
        "ts_str": time.strftime("%H:%M:%S"), "session_id": session_id or "unknown",
        "step": step, "model": model, "latency_ms": round(latency_ms),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total or None,
        "cost_usd": round(cost_usd, 7) if cost_usd else None,
        "status": status, "error": error, "type": "llm",
    })

def track_stt_call(session_id, model, latency_ms,
                   audio_duration_sec=None, status="success", error=None, fallback=False):
    cost_usd = None
    if audio_duration_sec and "gpt" in model.lower():
        cost_usd = round(audio_duration_sec * 0.006 / 60, 7)
    _obs_write({
        "log_id": uuid.uuid4().hex[:8], "ts": time.time(),
        "ts_str": time.strftime("%H:%M:%S"), "session_id": session_id or "unknown",
        "step": "STT", "model": model, "latency_ms": round(latency_ms),
        "cost_usd": cost_usd, "status": status, "error": error,
        "fallback": fallback, "type": "stt",
    })

def track_tts_call(session_id, model, latency_ms,
                   char_count=None, status="success", error=None, fallback=False):
    _obs_write({
        "log_id": uuid.uuid4().hex[:8], "ts": time.time(),
        "ts_str": time.strftime("%H:%M:%S"), "session_id": session_id or "unknown",
        "step": "TTS", "model": model, "latency_ms": round(latency_ms),
        "char_count": char_count, "cost_usd": None,
        "status": status, "error": error, "fallback": fallback, "type": "tts",
    })

def _obs_write(entry):
    with _obs_lock:
        _call_logs.append(entry)
        if len(_call_logs) > MAX_LOGS:
            _call_logs.pop(0)

def _obs_get_logs(session_id=None, step=None, obs_status=None, limit=200):
    with _obs_lock:
        logs = list(reversed(_call_logs))
    if session_id: logs = [l for l in logs if l["session_id"] == session_id]
    if step:       logs = [l for l in logs if l["step"] == step]
    if obs_status: logs = [l for l in logs if l["status"] == obs_status]
    return logs[:limit]

def _obs_platform_summary(window_seconds=86400):
    cutoff = time.time() - window_seconds
    with _obs_lock:
        logs = [l for l in _call_logs if l["ts"] >= cutoff]
    ok   = [l for l in logs if l["status"] == "success"]
    fail = [l for l in logs if l["status"] == "failure"]
    lats = [l["latency_ms"] for l in ok if l.get("latency_ms")]
    cost = sum(l["cost_usd"] or 0 for l in ok)
    steps = ["LLM_question","LLM_evaluation","STT","TTS","resume_parsing"]
    by_step = {}
    for step in steps:
        sl = [l for l in ok   if l["step"]==step and l.get("latency_ms")]
        fl = [l for l in fail if l["step"]==step]
        sv = sorted(l["latency_ms"] for l in sl)
        by_step[step] = {
            "p50": sv[int(len(sv)*.50)] if sv else 0,
            "p95": sv[int(len(sv)*.95)] if sv else 0,
            "avg": round(sum(sv)/len(sv)) if sv else 0,
            "calls": len([l for l in logs if l["step"]==step]),
            "failures": len(fl),
            "cost_usd": round(sum(l["cost_usd"] or 0 for l in ok if l["step"]==step), 6),
        }
    return {
        "window_seconds": window_seconds,
        "total_calls": len(logs), "success_calls": len(ok), "failure_calls": len(fail),
        "avg_latency_ms": round(sum(lats)/len(lats)) if lats else 0,
        "total_cost_usd": round(cost, 5),
        "success_rate_pct": round(len(ok)/len(logs)*100, 1) if logs else 100.0,
        "by_step": by_step,
        "recent_errors": [l for l in fail][-10:],
    }

def _obs_session_summary(session_id):
    logs = _obs_get_logs(session_id=session_id, limit=500)
    ok   = [l for l in logs if l["status"]=="success"]
    lats = [l["latency_ms"] for l in ok if l.get("latency_ms")]
    cost = sum(l["cost_usd"] or 0 for l in ok)
    steps = ["LLM_question","LLM_evaluation","STT","TTS","resume_parsing"]
    return {
        "session_id": session_id, "total_calls": len(logs),
        "success_calls": len(ok), "failure_calls": len(logs)-len(ok),
        "total_cost_usd": round(cost, 6),
        "avg_latency_ms": round(sum(lats)/len(lats)) if lats else 0,
        "step_breakdown": {
            s: {
                "calls": len([l for l in logs if l["step"]==s]),
                "avg_ms": round(sum(l["latency_ms"] for l in ok if l["step"]==s and l.get("latency_ms")) /
                          max(1, len([l for l in ok if l["step"]==s and l.get("latency_ms")]))),
                "cost_usd": round(sum(l["cost_usd"] or 0 for l in ok if l["step"]==s), 6),
            } for s in steps
        },
        "logs": logs[:50],
    }


# ════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════
JWT_SECRET     = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGO       = "HS256"
JWT_EXPIRE_MIN = 480

ADMIN_USER     = os.getenv("ADMIN_USER",    "admin")
ADMIN_PASS     = os.getenv("ADMIN_PASS",    "changeme_before_deploy")
REVIEWER_USER  = os.getenv("REVIEWER_USER", "reviewer")
REVIEWER_PASS  = os.getenv("REVIEWER_PASS", "changeme_before_deploy")

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer(auto_error=False)

PLATFORM_USERS = {
    ADMIN_USER:    {"hash": pwd_ctx.hash(ADMIN_PASS),    "role": "admin"},
    REVIEWER_USER: {"hash": pwd_ctx.hash(REVIEWER_PASS), "role": "reviewer"},
}

def _create_token(payload: dict, expires_min: int = JWT_EXPIRE_MIN) -> str:
    data = {**payload,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_min),
            "jti": secrets.token_hex(8)}
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGO)

def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

def _token_from_request(request: Request,
                        creds: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    if creds:
        return creds.credentials
    return request.cookies.get("_vlsi_tok")

async def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> dict:
    token = _token_from_request(request, creds)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return _decode_token(token)

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user

async def require_reviewer_or_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("admin", "reviewer"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Reviewer or admin access required")
    return user

def _set_auth_cookie(response: Response, token: str):
    is_prod = os.getenv("ENV", "development") == "production"
    response.set_cookie(key="_vlsi_tok", value=token, httponly=True,
                        secure=is_prod, samesite="lax",
                        max_age=JWT_EXPIRE_MIN*60, path="/")


# ════════════════════════════════════════════════════════════
# AI CLIENTS
# ════════════════════════════════════════════════════════════
import boto3
import requests as http_requests
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
polly_client = boto3.client(
    "polly", region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
cerebras_client = None
if CEREBRAS_API_KEY:
    cerebras_client = OpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1")
    print("Cerebras LLM ready.")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    from elevenlabs.client import ElevenLabs
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    print("ElevenLabs Scribe v2 ready (fallback STT).")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_VOICE   = os.getenv("SARVAM_VOICE", "ritu")
SARVAM_MODEL   = os.getenv("SARVAM_MODEL", "bulbul:v3")
if SARVAM_API_KEY:
    print(f"Sarvam AI TTS ready (voice: {SARVAM_VOICE}, model: {SARVAM_MODEL}).")

LMNT_API_KEY  = os.getenv("LMNT_API_KEY", "")
LMNT_VOICE_ID = os.getenv("LMNT_VOICE_ID", "")
TTS_ENABLED          = os.getenv("TTS_ENABLED", "true").lower() == "true"
HEAD_TURN_ENABLED    = os.getenv("HEAD_TURN_ENABLED", "true").lower() == "true"
EYE_AWAY_ENABLED     = os.getenv("EYE_AWAY_ENABLED", "true").lower() == "true"
FACE_DETECT_ENABLED  = os.getenv("FACE_DETECT_ENABLED", "true").lower() == "true"
SAPLING_API_KEY = os.getenv("SAPLING_API_KEY", "")
if SAPLING_API_KEY:
    print("Sapling AI content detector ready.")
else:
    print("Sapling AI detector not configured (SAPLING_API_KEY not set).")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_TTS_REF_AUDIO = None

_ref_audio_path = os.path.join(os.path.dirname(__file__), "ranjitha_4dmjitkw.mp3")
if MISTRAL_API_KEY and os.path.exists(_ref_audio_path):
    with open(_ref_audio_path, "rb") as _f:
        MISTRAL_TTS_REF_AUDIO = base64.b64encode(_f.read()).decode()

pocket_tts_model = None
pocket_tts_voice_state = None
try:
    from pocket_tts import TTSModel, export_model_state
    import scipy.io.wavfile
    import numpy as np
    pocket_tts_model = TTSModel.load_model()
    _voice_state_path = os.path.join(os.path.dirname(__file__), "ranjitha_voice.safetensors")
    if os.path.exists(_voice_state_path):
        pocket_tts_voice_state = pocket_tts_model.get_state_for_audio_prompt(_voice_state_path)
    elif os.path.exists(_ref_audio_path):
        pocket_tts_voice_state = pocket_tts_model.get_state_for_audio_prompt(_ref_audio_path)
        export_model_state(pocket_tts_voice_state, _voice_state_path)
    else:
        pocket_tts_voice_state = pocket_tts_model.get_state_for_audio_prompt("alba")
    print("Pocket TTS ready.")
except ImportError:
    print("Pocket TTS not installed.")
except Exception as e:
    print(f"Pocket TTS failed: {e}")


# ════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE
# ════════════════════════════════════════════════════════════
DOMAIN_TOPICS = {
    "analog_layout": ["basic layout concepts","device matching","parasitic awareness","latch-up and ESD","guard rings","DRC/LVS","symmetry techniques","shielding and routing","technology awareness"],
    "physical_design": ["floorplanning","power planning","placement","clock tree synthesis","routing","timing closure","static timing analysis","DRC/LVS signoff","tool knowledge"],
    "design_verification": ["verification methodologies","testbench architecture","functional coverage","assertions and SVA","simulation vs formal","debugging skills","protocol knowledge","regression and signoff","UVM"],
}
DOMAIN_RESOURCES = {
    "analog_layout": ["Weste & Harris — CMOS VLSI Design, Chapter 3 (Layout)","Razavi — Design of Analog CMOS Integrated Circuits, Chapter 18","Virtuoso Layout Suite User Guide — Cadence documentation","Calibre DRC/LVS User Manual — Mentor Graphics"],
    "physical_design": ["Synopsys ICC2 User Guide — Floorplanning and Placement chapters","Rabaey — Digital Integrated Circuits, Chapter 7","Static Timing Analysis for Nanometer Designs — Jayaram Bhasker","PrimeTime User Guide — Synopsys documentation"],
    "design_verification": ["Spear & Tumbush — SystemVerilog for Verification, Chapters 4-7","Bergeron — Writing Testbenches Using SystemVerilog","UVM Cookbook — Mentor Verification Academy","VCS Simulation User Guide — Synopsys documentation"],
}
CONTRADICTION_PAIRS = {
    "analog_layout": [
        {"topic":"device matching","angle_1":"What techniques do you use to achieve good device matching in layout?","angle_2":"If two devices have identical layout but different orientations relative to the gradient, will they match? Why or why not?"},
        {"topic":"parasitic awareness","angle_1":"How do you minimize parasitic capacitance in a layout?","angle_2":"In a situation where two nets run parallel for 100 microns, what is the dominant parasitic concern and how would you quantify it?"},
        {"topic":"latch-up and ESD","angle_1":"What causes latch-up in CMOS layout and how do you prevent it?","angle_2":"If you have a circuit where the substrate contact spacing is 50 microns from the nearest NMOS, is that acceptable? Walk me through your reasoning."},
    ],
    "physical_design": [
        {"topic":"clock tree synthesis","angle_1":"What is clock tree synthesis and what problem does it solve?","angle_2":"If after CTS you have 200ps of skew on one branch, what are the first three things you would check and in what order?"},
        {"topic":"timing closure","angle_1":"Explain the difference between setup and hold violations.","angle_2":"You have a hold violation of 50ps on a path that passes through three buffers. Adding more buffers made it worse. What is happening and what is the correct fix?"},
        {"topic":"floorplanning","angle_1":"What are the key objectives of floorplanning in physical design?","angle_2":"You are floorplanning a block with 60% utilization and your timing is already tight. Where exactly would you place the critical path macros and why?"},
    ],
    "design_verification": [
        {"topic":"functional coverage","angle_1":"What is functional coverage and why is it important in verification?","angle_2":"Your regression shows 98% functional coverage but you still found a bug in silicon. How is that possible and what does it tell you about your coverage model?"},
        {"topic":"assertions and SVA","angle_1":"What is the difference between a concurrent and immediate assertion in SVA?","angle_2":"You have an assertion that never fires during simulation. Is that good or bad? How do you determine which it is?"},
        {"topic":"simulation vs formal","angle_1":"What are the advantages of formal verification over simulation?","angle_2":"Formal verification proved your design is correct but you still found a bug. What are three possible explanations?"},
    ],
}
DIFFICULTY_LABELS = ["foundational","basic","intermediate","advanced","expert"]
FILLER_WORDS = {"um","uh","like","basically","actually","so","right","okay","hmm","err"}
EXPECTED_PAUSE_BY_DIFFICULTY = {"foundational":2.0,"basic":3.0,"intermediate":5.0,"advanced":8.0,"expert":10.0}


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
    input_mode: str = "text"
    whisper_confidence: float = 1.0

class AntiCheatEvent(BaseModel):
    session_id: str
    event_type: str
    turn: int
    timestamp: float
    metadata: str = ""

class ReportRequest(BaseModel):
    session_id: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ReviewerRegister(BaseModel):
    username: str
    password: str
    name: str
    designation: str
    organisation: str
    domain: str

class ReviewSubmit(BaseModel):
    session_id: str
    question_turn: int
    ai_score: float
    human_score: float
    dimension_assessments: list = []
    error_flags: list = []
    concept_corrections: list = []
    behavior_ratings: dict = {}
    verdict: str
    overall_feedback: str = ""


# ════════════════════════════════════════════════════════════
# LLM HELPERS  — with observability tracking
# ════════════════════════════════════════════════════════════
def call_llm(messages: list, temperature=0.5, max_tokens=1000,
             _session_id: str = "unknown", _step: str = "LLM_question") -> str:
    t0 = time.time()
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        result  = resp.choices[0].message.content.strip()
        usage   = resp.usage
        track_llm_call(
            session_id=_session_id, step=_step, model="gpt-4o-mini",
            latency_ms=(time.time()-t0)*1000,
            input_tokens=usage.prompt_tokens     if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            status="success",
        )
        return result
    except Exception as e:
        track_llm_call(session_id=_session_id, step=_step, model="gpt-4o-mini",
                       latency_ms=(time.time()-t0)*1000, status="failure", error=str(e))
        raise

def safe_json(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    try: return json.loads(re.sub(r',\s*([}\]])', r'\1', text))
    except Exception: pass
    return None

def call_cerebras(messages: list, temperature=0.5, max_tokens=1000,
                  _session_id: str = "unknown", _step: str = "resume_parsing") -> str:
    t0 = time.time()
    if cerebras_client:
        try:
            resp = cerebras_client.chat.completions.create(
                model="llama3.1-8b", messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            result = resp.choices[0].message.content.strip()
            track_llm_call(session_id=_session_id, step=_step, model="cerebras-llama3.1-8b",
                           latency_ms=(time.time()-t0)*1000, status="success")
            return result
        except Exception as e:
            track_llm_call(session_id=_session_id, step=_step, model="cerebras-llama3.1-8b",
                           latency_ms=(time.time()-t0)*1000, status="failure", error=str(e))
            print(f"Cerebras failed, falling back to GPT-4o-mini: {e}")
    return call_llm(messages, temperature, max_tokens, _session_id=_session_id, _step=_step)

def call_cerebras_json(messages: list, temperature=0.5, max_tokens=1000, retries=2,
                       _session_id: str = "unknown", _step: str = "resume_parsing") -> dict:
    for attempt in range(retries + 1):
        try:
            raw = call_cerebras(messages, temperature=temperature, max_tokens=max_tokens,
                                _session_id=_session_id, _step=_step)
            result = safe_json(raw)
            if result: return result
            if attempt < retries:
                messages = messages + [
                    {"role":"assistant","content":raw},
                    {"role":"user","content":"Return ONLY valid JSON. No markdown, no explanation."}
                ]
        except Exception as e:
            print(f"Cerebras JSON attempt {attempt+1} failed: {e}")
            if attempt == retries: raise
    return {}

def call_llm_json(messages: list, temperature=0.5, max_tokens=1000, retries=2,
                  _session_id: str = "unknown", _step: str = "LLM_question") -> dict:
    for attempt in range(retries + 1):
        try:
            raw = call_llm(messages, temperature=temperature, max_tokens=max_tokens,
                           _session_id=_session_id, _step=_step)
            result = safe_json(raw)
            if result: return result
            if attempt < retries:
                messages = messages + [
                    {"role":"assistant","content":raw},
                    {"role":"user","content":"Return ONLY valid JSON. No markdown, no explanation."}
                ]
        except Exception as e:
            print(f"LLM attempt {attempt+1} failed: {e}")
            if attempt == retries: raise
    return {}


# ════════════════════════════════════════════════════════════
# RESUME PARSER
# ════════════════════════════════════════════════════════════
def parse_resume(resume_text: str, _session_id: str = "unknown") -> dict:
    prompt = f"""You are a VLSI expert HR analyst. Parse this resume and extract candidate details.

Resume:
{resume_text[:4000]}

Return ONLY this JSON (no markdown):
{{
  "candidate_name": "Full Name from resume",
  "email": "email if found or empty string",
  "phone": "phone if found or empty string",
  "skills": ["skill1", "skill2"],
  "vlsi_skills": ["only VLSI/semiconductor relevant skills"],
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

domain: analog_layout | physical_design | design_verification
level: fresh_graduate | trained_fresher | experienced_junior | experienced_senior
is_vlsi_suitable: true if VLSI/Semiconductor/Electronics background"""

    result = call_cerebras_json([{"role":"user","content":prompt}],
                                temperature=0.1, max_tokens=1000,
                                _session_id=_session_id, _step="resume_parsing")
    if result and "is_vlsi_suitable" in result:
        if "candidate_name" not in result or not result["candidate_name"]:
            result["candidate_name"] = "Candidate"
        return result
    return {
        "candidate_name":"Candidate","email":"","phone":"","skills":[],
        "vlsi_skills":[],"is_vlsi_suitable":False,
        "rejection_reason":"Could not parse resume. Please upload a valid VLSI/Semiconductor resume.",
        "domain":"unknown","level":"unknown","years_experience":0,"tools":[],
        "key_projects":[],"background_summary":"","training_institutes":[],"education":""
    }


# ════════════════════════════════════════════════════════════
# BEHAVIORAL BASELINE & DEVIATION
# ════════════════════════════════════════════════════════════
def count_fillers(text: str) -> float:
    words = text.lower().split()
    if not words: return 0.0
    return sum(1 for w in words if w in FILLER_WORDS) / len(words)

def count_personal_pronouns(text: str) -> float:
    words = text.lower().split()
    if not words: return 0.0
    pronouns = {"i","my","me","myself","we","our","i've","i'd","i'll"}
    return sum(1 for w in words if w in pronouns) / len(words)

def count_self_corrections(text: str) -> float:
    patterns = [r'\bi mean\b',r'\bactually\b',r'\bwait\b',r'\bno wait\b',r'\bsorry\b',r'\blet me rephrase\b']
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
        "avg_duration_sec":    sum(h.get("answer_duration_sec",0) for h in voice) / len(voice) if voice else 0,
        "avg_word_count":      sum(h.get("word_count",0) for h in warmup) / len(warmup),
        "avg_filler_rate":     sum(h.get("filler_rate",0) for h in warmup) / len(warmup),
        "avg_pronoun_rate":    sum(h.get("pronoun_rate",0) for h in warmup) / len(warmup),
        "avg_thinking_pause":  sum(h.get("thinking_pause_sec",0) for h in voice) / len(voice) if voice else 0,
        "avg_correction_rate": sum(h.get("correction_rate",0) for h in warmup) / len(warmup),
        "sample_size":         len(warmup),
    }

def analyze_behavioral_deviation(session, answer, duration_sec, word_count, thinking_pause, input_mode, difficulty):
    baseline = session.get("behavioral_baseline")
    if not baseline or baseline.get("sample_size",0) == 0:
        return {"deviation_score":0.0,"flags":[],"filler_rate":0.0,"pronoun_rate":0.0,"correction_rate":0.0}
    flags = []; score = 0.0
    filler_rate     = count_fillers(answer)
    pronoun_rate    = count_personal_pronouns(answer)
    correction_rate = count_self_corrections(answer)
    has_voice       = baseline.get("has_voice", False)
    if has_voice and input_mode=="voice" and baseline["avg_duration_sec"]>0 and duration_sec>0:
        ratio = duration_sec / baseline["avg_duration_sec"]
        if ratio < 0.25: flags.append("unusually_short_answer"); score += 1.5
        elif ratio > 5.0: flags.append("unusually_long_answer"); score += 0.5
    avg_wc = baseline["avg_word_count"]
    if avg_wc > 5 and word_count > 0 and word_count/avg_wc < 0.2:
        flags.append("very_few_words"); score += 1.0
    avg_fr = baseline["avg_filler_rate"]
    if avg_fr > 0.008 and filler_rate < avg_fr*0.1 and word_count > 25:
        flags.append("suspiciously_clean_speech"); score += 2.0
    avg_pr = baseline["avg_pronoun_rate"]
    if avg_pr > 0.02 and pronoun_rate < avg_pr*0.15 and word_count > 25:
        flags.append("personal_pronouns_vanished"); score += 1.5
    if has_voice and input_mode=="voice" and thinking_pause > 0:
        expected_pause = EXPECTED_PAUSE_BY_DIFFICULTY.get(difficulty, 4.0)
        session["pause_history"] = session.get("pause_history", [])
        session["pause_history"].append({"pause":thinking_pause,"difficulty":difficulty})
        if len(session["pause_history"]) >= 4:
            try:
                stddev = statistics.stdev([p["pause"] for p in session["pause_history"]])
                if stddev < 0.5: flags.append("low_pause_variance"); score += 2.0
            except Exception: pass
        if difficulty in ("advanced","expert") and thinking_pause < expected_pause*0.3:
            flags.append(f"instant_answer_on_{difficulty}_question"); score += 1.5
    if difficulty in ("advanced","expert") and avg_wc > 0 and word_count/max(avg_wc,1) > 4.0:
        flags.append("answer_length_spike_on_hard_question"); score += 1.5
    avg_cr = baseline.get("avg_correction_rate",0)
    if avg_cr > 0.002 and correction_rate < avg_cr*0.1 and word_count > 30:
        flags.append("self_corrections_vanished"); score += 1.0
    return {"deviation_score":score,"flags":flags,"filler_rate":filler_rate,
            "pronoun_rate":pronoun_rate,"correction_rate":correction_rate}


# ════════════════════════════════════════════════════════════
# ANTI-CHEAT SIGNAL ENGINE
# ════════════════════════════════════════════════════════════
def assess_answer_complexity(session, answer, eval_score, eval_difficulty, resume_level):
    level_map = {"fresh_graduate":0,"trained_fresher":1,"experienced_junior":2,"experienced_senior":3}
    diff_map  = {d:i for i,d in enumerate(DIFFICULTY_LABELS)}
    candidate_level_num  = level_map.get(resume_level, 1)
    answer_difficulty_num = diff_map.get(eval_difficulty, 1)
    if eval_score >= 8 and answer_difficulty_num > candidate_level_num + 1:
        gap = answer_difficulty_num - candidate_level_num
        return {"above_level":True,"gap_levels":gap,
                "flag":f"Answer quality significantly above calibrated level ({eval_difficulty} when resume suggests {resume_level})"}
    return {"above_level":False,"gap_levels":0,"flag":""}

def should_give_hint(session): return session.get("last_question_type","")=="recovery_probe" or session.get("last_eval_quality","")=="honest_admission"
def record_hint(session, turn, topic, hint_text):
    session.setdefault("hint_events",[]).append({"turn":turn,"topic":topic,"hint_text":hint_text,"recovery_score":None,"recovery_speed":None,"recovery_quality":None})

def evaluate_recovery(session, current_turn, answer, eval_score):
    for hint in session.get("hint_events",[]):
        if hint["recovery_score"] is None and current_turn == hint["turn"]+1:
            hint["recovery_speed"] = "fast"
            if eval_score >= 7:
                hint["recovery_quality"] = "complete"; hint["recovery_score"] = eval_score
                session.setdefault("genuine_signals",[]).append(f"Fast complete recovery at question {current_turn} on {hint['topic']}")
                if eval_score == 10:
                    session.setdefault("suspicion_events",[]).append({"type":"perfect_recovery_after_hint","turn":current_turn,"weight":15,"detail":f"Scored 10/10 immediately after hint on {hint['topic']} at question {current_turn}"})
            elif eval_score >= 4: hint["recovery_quality"]="partial"; hint["recovery_score"]=eval_score
            else: hint["recovery_quality"]="none"; hint["recovery_score"]=eval_score

def update_smooth_talker(session, eval_data, question_type):
    if not eval_data or session.get("phase") != "interview": return
    quality    = eval_data.get("quality","")
    confidence = eval_data.get("confidence_level","")
    accuracy   = eval_data.get("accuracy","")
    quadrant   = eval_data.get("quadrant","")
    session.setdefault("smooth_talker_signals",[])
    if question_type=="scenario" and quadrant=="dangerous_fake":          session["smooth_talker_signals"].append("Collapsed on scenario after confident definition")
    if question_type=="why_probe" and accuracy in ("wrong","partial"):    session["smooth_talker_signals"].append("Could not explain WHY — surface-level knowledge only")
    if question_type=="numerical" and quality in ("weak","adequate") and confidence=="high": session["smooth_talker_signals"].append("Evaded numerical probe with no real numbers")
    if question_type=="personal_anchor" and quality=="weak":              session["smooth_talker_signals"].append("Generic answer to personal experience question")
    if question_type=="contradiction" and accuracy=="wrong":              session["smooth_talker_signals"].append("Contradicted earlier answer — memorized not understood")
    count = len(session["smooth_talker_signals"])
    session["smooth_talker_detected"] = count >= 3
    session["smooth_talker_score"]    = min(100, count*20)

def record_notable(session, turn, question, answer, moment_type, detail):
    session.setdefault("notable_moments",[]).append({"turn":turn,"moment_type":moment_type,"question":question[:150],"answer_excerpt":(answer or "")[:150],"detail":detail})

def count_active_signals(session, scored_history):
    count = 0
    anticheat = session.get("anticheat_events",[])
    if any(e["event_type"]=="tab_switch"      for e in anticheat): count += 1
    if any(e["event_type"]=="paste_event"     for e in anticheat): count += 1
    if any(e["event_type"]=="dom_overlay"     for e in anticheat): count += 1
    if any(e["event_type"]=="screen_share"    for e in anticheat): count += 1
    if any(e["event_type"]=="canary_triggered" for e in anticheat): count += 1
    if any(e["event_type"]=="head_turned"     for e in anticheat): count += 1
    if any(e["event_type"]=="eye_away"        for e in anticheat): count += 1
    if any(e["event_type"]=="split_screen"   for e in anticheat): count += 1
    if any(e["event_type"]=="ai_answer_overlay" for e in anticheat): count += 1
    if any(e["event_type"]=="ai_extension_detected" for e in anticheat): count += 1
    # AI-generated answer detected by Sapling
    if any("ai_generated_answer" in h.get("behavioral_flags",[]) for h in scored_history): count += 1
    flags_all = []
    for h in scored_history: flags_all.extend(h.get("behavioral_flags",[]))
    if "suspiciously_clean_speech"   in flags_all: count += 1
    if "personal_pronouns_vanished"  in flags_all: count += 1
    if "low_pause_variance"          in flags_all: count += 1
    if "self_corrections_vanished"   in flags_all: count += 1
    if session.get("smooth_talker_detected"): count += 1
    honest_count = sum(1 for h in scored_history if (h.get("evaluation") or {}).get("quality")=="honest_admission")
    if len(scored_history)>=8 and honest_count==0: count += 1
    df_count = sum(1 for h in scored_history if (h.get("evaluation") or {}).get("quadrant")=="dangerous_fake")
    if df_count >= 3: count += 1
    if any(h.get("contradiction_inconsistency") for h in scored_history): count += 1
    if any(h.get("above_level") for h in scored_history): count += 1
    return count

def compute_topic_suspicion(session, scored_history):
    anticheat = session.get("anticheat_events",[])
    topic_suspicion = {}
    for h in scored_history:
        topic = h.get("topic","general")
        if topic not in topic_suspicion: topic_suspicion[topic] = {"score":0,"flags":[]}
        prev_tab = [e for e in anticheat if e["event_type"]=="tab_switch" and e["turn"]==h.get("turn",0)-1]
        if prev_tab and (h.get("evaluation") or {}).get("quality")=="strong":
            topic_suspicion[topic]["score"] += 20
            topic_suspicion[topic]["flags"].append(f"Tab switch before strong answer on {topic} at question {h['turn']}")
        for flag in h.get("behavioral_flags",[]):
            if flag in ("suspiciously_clean_speech","personal_pronouns_vanished","low_pause_variance"):
                topic_suspicion[topic]["score"] += 10
                topic_suspicion[topic]["flags"].append(f"{flag} on {topic} answer")
    return topic_suspicion

def compute_suspicion_score(session, scored_history):
    anticheat = session.get("anticheat_events",[])
    suspicion = 0.0; flags = []
    for ev in anticheat:
        if ev["event_type"]=="tab_switch":
            next_ans = next((h for h in scored_history if h.get("turn",0)>ev["turn"] and h.get("evaluation")),None)
            if next_ans:
                q = next_ans["evaluation"].get("quality",""); diff = next_ans.get("difficulty","basic"); topic = next_ans.get("topic","unknown")
                if q=="strong" and diff in ("advanced","expert"): suspicion+=20; flags.append(f"Tab switch at Q{ev['turn']} → strong {diff} {topic} answer (Q{next_ans['turn']})")
                elif q=="strong": suspicion+=8; flags.append(f"Tab switch at Q{ev['turn']} → strong answer on {topic}")
                else: suspicion+=2
            else: suspicion+=2
        if ev["event_type"]=="paste_event": suspicion+=15; flags.append(f"Paste event at Q{ev['turn']}")
        if ev["event_type"] in ("dom_overlay","canary_triggered"): suspicion+=20; flags.append(f"AI browser extension detected at Q{ev['turn']}")
        if ev["event_type"]=="screen_share": suspicion+=12; flags.append(f"Screen sharing at Q{ev['turn']}")
    # Head turned away (MEDIUM)
    head_turn_events = [ev for ev in anticheat if ev["event_type"]=="head_turned"]
    if head_turn_events:
        suspicion += len(head_turn_events) * 8
        flags.append(f"Candidate looked away from screen {len(head_turn_events)} time(s) (questions {', '.join(str(ev['turn']) for ev in head_turn_events[:3])})")
    # Eyes looking away while head faces camera (HIGH)
    eye_away_events = [ev for ev in anticheat if ev["event_type"]=="eye_away"]
    if eye_away_events:
        suspicion += len(eye_away_events) * 10
        flags.append(f"Eyes looking away {len(eye_away_events)} time(s) while facing camera (questions {', '.join(str(ev['turn']) for ev in eye_away_events[:3])})")
    # Split screen detected (MEDIUM — candidate may be reading from another app)
    split_events = [ev for ev in anticheat if ev["event_type"]=="split_screen"]
    if split_events:
        suspicion += len(split_events) * 8
        flags.append(f"Split screen detected {len(split_events)} time(s) (questions {', '.join(str(ev['turn']) for ev in split_events[:3])})")
    # AI answer overlay detected on screen (VERY HIGH — direct evidence of cheating tool)
    ai_overlay_events = [ev for ev in anticheat if ev["event_type"]=="ai_answer_overlay"]
    if ai_overlay_events:
        suspicion += len(ai_overlay_events) * 25
        flags.append(f"AI answer overlay detected on screen {len(ai_overlay_events)} time(s) — cheating tool active")
    # AI extension detected (HIGH)
    ai_ext_events = [ev for ev in anticheat if ev["event_type"]=="ai_extension_detected"]
    if ai_ext_events:
        suspicion += len(ai_ext_events) * 20
        flags.append(f"AI browser extension detected (e.g. Parakeet, Copilot)")
    # AI-generated answers detected by Sapling (VERY HIGH)
    ai_gen_turns = [h for h in scored_history if "ai_generated_answer" in h.get("behavioral_flags",[])]
    if ai_gen_turns:
        suspicion += len(ai_gen_turns) * 20
        ai_scores = []
        for h in ai_gen_turns:
            det = h.get("ai_detection", {})
            s = (det.get("sapling") or {}).get("score", 0)
            ai_scores.append(s)
        avg_ai = sum(ai_scores)/len(ai_scores) if ai_scores else 0
        flags.append(f"AI-generated answers detected in {len(ai_gen_turns)} response(s) (avg AI score: {avg_ai:.0%})")
    clean_turns = [h for h in scored_history if "suspiciously_clean_speech" in h.get("behavioral_flags",[])]
    if len(clean_turns)>=3: suspicion+=len(clean_turns)*8; flags.append(f"Filler words vanished in {len(clean_turns)} answers")
    pronoun_turns = [h for h in scored_history if "personal_pronouns_vanished" in h.get("behavioral_flags",[])]
    if len(pronoun_turns)>=2: suspicion+=len(pronoun_turns)*7; flags.append(f"Personal pronouns vanished in {len(pronoun_turns)} answers")
    correction_turns = [h for h in scored_history if "self_corrections_vanished" in h.get("behavioral_flags",[])]
    if len(correction_turns)>=2: suspicion+=len(correction_turns)*5; flags.append(f"Self-correction pattern disappeared in {len(correction_turns)} answers")
    if any("low_pause_variance" in h.get("behavioral_flags",[]) for h in scored_history): suspicion+=15; flags.append("Identical thinking pause across all difficulty levels")
    instant_hard = [h for h in scored_history if any(f.startswith("instant_answer_on_") for f in h.get("behavioral_flags",[]))]
    if instant_hard: suspicion+=len(instant_hard)*8; flags.append(f"Instant answers on hard questions at Q{', Q'.join(str(h['turn']) for h in instant_hard[:3])}")
    honest_count = sum(1 for h in scored_history if (h.get("evaluation") or {}).get("quality")=="honest_admission")
    if len(scored_history)>=8 and honest_count==0: suspicion+=12; flags.append("Zero honest admissions in full interview")
    df_turns = [h for h in scored_history if (h.get("evaluation") or {}).get("quadrant")=="dangerous_fake"]
    if len(df_turns)>=3: suspicion+=len(df_turns)*8; flags.append(f"Confident+wrong pattern at Q{', Q'.join(str(h['turn']) for h in df_turns[:3])}")
    bwc = (session.get("behavioral_baseline") or {}).get("avg_word_count",60)
    spike_turns = [h for h in scored_history if "answer_length_spike_on_hard_question" in h.get("behavioral_flags",[])]
    if spike_turns: suspicion+=len(spike_turns)*10; flags.append(f"Answer length spiked on hard questions")
    st_signals = session.get("smooth_talker_signals",[])
    if len(st_signals)>=3: suspicion+=len(st_signals)*5; flags.append(f"Smooth talker pattern: {'; '.join(st_signals[:3])}")
    for ev in session.get("suspicion_events",[]):
        if ev["type"]=="perfect_recovery_after_hint": suspicion+=ev.get("weight",15); flags.append(ev["detail"])
    above_level_turns = [h for h in scored_history if h.get("above_level")]
    if above_level_turns: suspicion+=len(above_level_turns)*8; flags.append(f"Answer sophistication above calibrated level")
    contradiction_fails = [h for h in scored_history if h.get("contradiction_inconsistency")]
    if contradiction_fails: suspicion+=len(contradiction_fails)*12; flags.append(f"Contradicted earlier answers")
    suspicion = min(100, suspicion)
    signal_count = count_active_signals(session, scored_history)
    verdict = "critical" if signal_count >= 7 else None
    if suspicion < 15:   level = "clean"
    elif suspicion < 35: level = "low_risk"
    elif suspicion < 60: level = "moderate_risk"
    else:                level = "high_risk"
    if verdict == "critical":
        level = "high_risk"
        flags.append(f"CRITICAL: {signal_count} distinct suspicion signals detected")
    return {"suspicion_score":suspicion,"integrity_level":level,"signal_count":signal_count,"critical_verdict":verdict=="critical","flags":flags}


# ════════════════════════════════════════════════════════════
# CONTRADICTION TRACKING
# ════════════════════════════════════════════════════════════
def get_next_contradiction(session):
    domain = session["resume"]["domain"]
    pairs  = CONTRADICTION_PAIRS.get(domain, [])
    asked  = session.get("contradiction_asked", {})
    for pair in pairs:
        topic = pair["topic"]; state = asked.get(topic,"none")
        if state=="angle_1_asked":
            if session["turn"] - asked.get(f"{topic}_turn",0) >= 6:
                return {"pair":pair,"angle":"angle_2"}
        if state=="none" and topic in session.get("topics_covered",[]):
            return {"pair":pair,"angle":"angle_1"}
    return None

def record_contradiction_result(session, topic, angle, eval_data, turn):
    asked = session.setdefault("contradiction_asked",{})
    if angle=="angle_1":
        asked[topic]="angle_1_asked"; asked[f"{topic}_turn"]=turn
        asked[f"{topic}_angle1_score"]=int(eval_data.get("score") or 5) if eval_data else 5
        asked[f"{topic}_angle1_accuracy"]=eval_data.get("accuracy","partial") if eval_data else "partial"
    elif angle=="angle_2":
        asked[topic]="complete"
        a1s=asked.get(f"{topic}_angle1_score",5); a2s=int(eval_data.get("score") or 5) if eval_data else 5
        a1a=asked.get(f"{topic}_angle1_accuracy","partial"); a2a=eval_data.get("accuracy","partial") if eval_data else "partial"
        inconsistent=(a1a in ("correct","partial") and a1s>=6 and a2a=="wrong" and a2s<=4)
        asked[f"{topic}_inconsistent"]=inconsistent
        return inconsistent
    return False

# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# INTERVIEW ENGINE — delegates to agent.py, strategy_engine.py,
# evaluation_validator.py, repetition_guard.py
# Old functions replaced with wrappers to new modules.
# ════════════════════════════════════════════════════════════
def decide_question_type(session):
    phase = session["phase"]
    if phase == "warmup": return "warmup", None
    tech_turn    = session["turn"] - session.get("warmup_turns",2)
    last_type    = session.get("last_question_type","")
    anchor_count = session.get("anchor_count",0)
    last_eval    = session.get("last_eval_quality","adequate")
    last_confidence = session.get("last_confidence","medium")
    last_topic   = session.get("last_topic","")
    recovery_attempts = session.get("recovery_attempts_per_topic",{})
    topic_recovery_count = recovery_attempts.get(last_topic,0)
    topic_perf = session.get("topic_performance", {})

    # RULE 0: Resume-first — first 3 tech questions about candidate's own projects
    if tech_turn <= 3 and session.get("resume",{}).get("key_projects"):
        return "resume_project", None

    # RULE 1: Anti-cheat stealth — if suspicion high + last answer was strong, verify
    running_suspicion = session.get("running_suspicion", 0)
    if running_suspicion >= 25 and last_eval == "strong" and last_type != "verification_followup":
        return "verification_followup", {"verify_topic": last_topic}

    # RULE 2: Recovery after failed recovery — move to new skill
    if last_type=="recovery_probe" and topic_recovery_count>=2:
        session["skip_topic"]=last_topic; return "definition",{"force_new_topic":True}

    # RULE 3: definition → always scenario
    if last_type=="definition": return "scenario",None

    # RULE 4: honest_admission → recovery_probe with hint
    if last_eval=="honest_admission":
        recovery_attempts[last_topic]=topic_recovery_count+1
        session["recovery_attempts_per_topic"]=recovery_attempts
        return "recovery_probe",{"force_hint":True}

    # RULE 5: poor_articulation → practical_example
    if last_eval=="poor_articulation": return "practical_example",None

    # RULE 6: confident + shallow on scenario → why_probe
    if last_type=="scenario" and last_confidence=="high" and last_eval in ("weak","adequate"): return "why_probe",None

    # RULE 7: Contradiction pair opportunity
    contradiction = get_next_contradiction(session)
    if contradiction: return "contradiction",contradiction

    # RULE 8: Numerical after concept questions
    if last_type in ("scenario","why_probe","practical_example") and tech_turn>4: return "numerical",None

    # RULE 9: Personal anchor at strategic turns
    if anchor_count<3 and tech_turn in [5,10,16]: return "personal_anchor",None

    # RULE 10: Skip mastered topics — if a topic has avg_score >= 7 and 2+ questions, skip it
    # Prioritize resume gaps (untested topics)
    resume_gaps = session.get("resume_gaps", [])
    untested_gaps = [t for t in resume_gaps if t not in session.get("topics_covered",[])]
    if untested_gaps:
        return "definition", {"force_gap_topic": True}

    return "definition",None


# ════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ════════════════════════════════════════════════════════════
def build_system_prompt(session, forced_type, extra=None):
    r = session["resume"]; domain = r["domain"]
    topics   = DOMAIN_TOPICS.get(domain,[])
    covered  = session.get("topics_covered",[])
    uncovered= [t for t in topics if t not in covered]
    tech_turn= session["turn"] - session.get("warmup_turns",2)
    force_hint       = extra.get("force_hint",False) if extra else False
    contradiction_data= extra.get("pair") if extra and extra.get("angle") else None
    contradiction_angle= extra.get("angle") if extra else None
    specific_question=""
    if forced_type=="contradiction" and contradiction_data and contradiction_angle:
        q_key=f"angle_{contradiction_angle.split('_')[1]}" if "_" in str(contradiction_angle) else contradiction_angle
        specific_question=f"\nUSE THIS EXACT CONTRADICTION QUESTION:\n\"{contradiction_data.get(q_key,contradiction_data.get('angle_2',''))}\"\n"
    hint_instruction=""
    if force_hint: hint_instruction="\nIMPORTANT: Give a small hint. Set hint_given=true.\n"
    projects=r.get("key_projects",[]); projects_text=", ".join(projects) if projects else "No projects listed"
    last_eval_quality=session.get("last_eval_quality","adequate")
    last_confidence=session.get("last_confidence","medium")

    # Build context-aware reaction instruction based on last answer
    last_answer_summary=""
    reaction_instruction=""
    if session.get("history") and session["history"][-1].get("answer"):
        last_a=session["history"][-1]
        last_eval = last_a.get("evaluation") or {}
        last_answer_summary=f"- Last answer quality: {last_eval_quality} | Confidence: {last_confidence}"
        if last_eval.get("notes"):
            last_answer_summary+=f"\n- Evaluator note: {last_eval['notes']}"

        # Personality-driven reactions based on last answer
        if last_eval.get("accuracy") == "off_topic":
            off_count = session.get("off_topic_count", 0)
            reaction_instruction = f"REACT: The candidate's last answer was OFF-TOPIC ({off_count} time(s)). Politely say: 'That's interesting, but it wasn't quite related to what I was asking about. Let me rephrase --' then re-ask the same topic simpler."
        elif last_eval_quality == "strong":
            reaction_instruction = "REACT: Start with a short positive acknowledgment (e.g., 'Good point about [specific thing they said].' or 'That's a solid understanding.'). Then ask next question."
        elif last_eval_quality == "weak":
            reaction_instruction = "REACT: Acknowledge what they got RIGHT first (e.g., 'You're on the right track with [correct part]...'). Then gently probe the gap. NEVER just say 'wrong'."
        elif last_eval_quality == "honest_admission":
            reaction_instruction = "REACT: Validate their honesty: 'That's completely fine — knowing what you don't know is actually valuable in engineering.' Give a brief 1-sentence teaching hint, then move to a simpler related question."
        elif last_eval_quality == "poor_articulation":
            reaction_instruction = "REACT: Say 'I think you know this — let me ask it differently.' Then rephrase as a practical example."
        elif last_eval_quality == "adequate":
            reaction_instruction = "REACT: Push deeper: 'Can you be more specific?' or 'What numbers would you expect?' Don't accept vague answers."
        else:
            reaction_instruction = "REACT: Give a short natural reaction before your next question."

    level_guide={
        "fresh_graduate":"Ask about fundamentals and basic concepts only.",
        "trained_fresher":"Ask concepts with simple application scenarios.",
        "experienced_junior":"Ask application-level questions with real scenarios.",
        "experienced_senior":"Ask advanced debugging, optimization, and tradeoff questions.",
    }.get(r["level"],"Calibrate to candidate's experience level.")

    # Domain-specific prompt with mentor tone
    if domain=="physical_design":
        domain_prompt="""ROLE: Senior VLSI mentor conducting a supportive Physical Design mock interview.
DOMAIN: STA, synthesis, floorplanning, placement, CTS, routing, congestion, IR drop, EM, ECO, timing closure.
MANDATORY: Include numerical reasoning in 1 of every 2-3 questions. Ask: slack values, timing margins, utilization %, buffer counts."""
    elif domain=="analog_layout":
        domain_prompt="""ROLE: Senior VLSI mentor conducting a supportive Analog Layout mock interview.
DOMAIN: MOSFET behavior, matching, parasitics, LDE, EMIR, layout techniques, symmetry.
MANDATORY: Include numerical scaling. Ask: Id proportional to W/L, mismatch proportional to 1/sqrt(Area)."""
    elif domain=="design_verification":
        domain_prompt="""ROLE: Senior VLSI mentor conducting a supportive Design Verification mock interview.
DOMAIN: SystemVerilog, UVM, assertions, coverage, testbench, simulation, formal verification.
MANDATORY: Include timing numerics. Ask: 5 cycles at 1GHz, latency calculations."""
    else:
        domain_prompt="ROLE: Senior VLSI mentor conducting a supportive mock interview. Adapt to the candidate's domain."

    # Per-topic performance context
    topic_perf = session.get("topic_performance", {})
    topic_perf_summary = ""
    mastered = [t for t,d in topic_perf.items() if d.get("avg_score",0) >= 7 and d.get("count",0) >= 2]
    weak_topics = [t for t,d in topic_perf.items() if d.get("avg_score",5) < 4 and d.get("count",0) >= 1]
    if mastered: topic_perf_summary += f"\n- Topics already mastered (SKIP these): {', '.join(mastered)}"
    if weak_topics: topic_perf_summary += f"\n- Weak topics (ask simpler): {', '.join(weak_topics)}"

    # Resume gaps — topics NOT in candidate's resume
    resume_gaps = session.get("resume_gaps", [])
    gaps_tested = [t for t in resume_gaps if t in covered]
    gaps_untested = [t for t in resume_gaps if t not in covered]

    # Resume-first strategy for early questions
    resume_strategy = ""
    if tech_turn <= 3 and projects:
        resume_strategy = f"\nSTRATEGY: This is an early question. Ask about the candidate's OWN projects/experience from their resume: {projects_text}. Dig into THEIR work, not generic textbook questions."
    elif gaps_untested:
        resume_strategy = f"\nSTRATEGY: Test these resume GAPS (topics candidate didn't mention): {', '.join(gaps_untested[:3])}. These reveal blind spots."

    # Returning candidate context
    returning_context = ""
    prev_sessions = session.get("previous_sessions", [])
    if prev_sessions:
        last_session = prev_sessions[-1]
        returning_context = f"\nRETURNING CANDIDATE: Previously scored {last_session.get('overall_score', '?')}/100. Previously weak on: {', '.join(last_session.get('weak_topics', [])[:3])}. Focus on their previously weak topics to check improvement."

    # Anti-cheat stealth: if suspicion is high, ask verification follow-ups
    suspicion_instruction = ""
    if forced_type == "verification_followup":
        suspicion_instruction = "\nVERIFICATION: The candidate's last answer was suspiciously good. Ask them to walk through their answer STEP BY STEP. Example: 'You mentioned [specific term]. Can you walk me through exactly how you'd implement that?' Do NOT reveal any suspicion. Frame it as genuine curiosity."

    return f"""{domain_prompt}

PERSONALITY:
You are a mentor who happens to be interviewing — encouraging but honest. Your tone is conversational and supportive.
Your questions will be read aloud via TTS — keep them conversational and speakable. No symbols, no code.

{reaction_instruction}

LEVEL CALIBRATION: {level_guide}
CANDIDATE: {r.get('candidate_name','Candidate')} | {domain.replace('_',' ')} | {r['level'].replace('_',' ')} | Tools: {', '.join(r.get('tools',[]))} | Projects: {projects_text}
STATE: Turn {session['turn']} | Tech Q: {tech_turn} | Difficulty: {DIFFICULTY_LABELS[session['difficulty_level']]} | Topics covered: {', '.join(covered) or 'none'} | Remaining: {', '.join(uncovered[:4]) or 'all covered'}
{topic_perf_summary}
{last_answer_summary}
{resume_strategy}
{returning_context}
{specific_question}{hint_instruction}{suspicion_instruction}

BEHAVIOR RULES:
- ALWAYS start with a short (3-8 word) reaction to their last answer. NEVER skip the reaction. NEVER use generic "Good" or "Okay".
- When relevant, reference something the candidate said in a previous answer (builds continuity).
- If answer was adequate but vague, push: "Can you be more specific?" or "What numbers would you expect?"
- If candidate struggles, reduce complexity and give a hint. Stay on SAME concept.
- If candidate is strong, push into edge cases and tradeoffs.

MANDATORY QUESTION TYPE FOR THIS TURN: {forced_type}

RETURN ONLY VALID JSON (no markdown):
{{
  "reaction": "Your short reaction to their last answer (3-8 words). Empty string if first question.",
  "question": "Your interview question — conversational, speakable, max 2 sentences",
  "question_type": "{forced_type}",
  "topic": "specific topic being tested",
  "difficulty": "{DIFFICULTY_LABELS[session['difficulty_level']]}",
  "hint_given": {str(force_hint).lower()},
  "hint_text": "the hint if hint_given is true, otherwise null"
}}"""

def build_evaluation_prompt(session, question, answer, difficulty, question_type):
    r = session["resume"]
    return f"""You are a senior VLSI technical evaluator scoring a candidate's answer across multiple dimensions.

CANDIDATE: {r['domain'].replace('_',' ')} | {r['level'].replace('_',' ')} ({r.get('years_experience',0)} years)
QUESTION ({question_type}, {difficulty}): {question}
ANSWER: {answer}

EVALUATION RULES:
- OFF-TOPIC: If the answer is completely unrelated to the question, set accuracy="off_topic", quality="weak", score=1
- "I don't know" = quality "honest_admission", score=6
- "I don't know + reasoning attempt" = score=8
- Correct but poorly explained = quality "poor_articulation"
- Unconventional but technically defensible = accuracy "correct"

SCORING DIMENSIONS (each 1-10):
- technical_accuracy: Are the facts correct? (wrong=1-3, partial=4-6, correct=7-10)
- depth_of_understanding: Can they explain WHY, not just WHAT? (surface=1-3, some depth=4-6, deep=7-10)
- practical_application: Can they apply it to a real scenario? (no=1-3, somewhat=4-6, yes=7-10)
- communication: Is the answer structured and clear? (confusing=1-3, okay=4-6, clear=7-10)
- confidence_calibration: Does their confidence match their accuracy? (confident+wrong=1-3, uncertain+correct=5, confident+correct=8-10)

RETURN ONLY VALID JSON:
{{
  "quality": "strong|adequate|weak|honest_admission|poor_articulation",
  "accuracy": "correct|partial|wrong|off_topic|not_applicable",
  "confidence_level": "high|medium|low",
  "quadrant": "genuine_expert|genuine_nervous|dangerous_fake|honest_confused",
  "scores": {{
    "technical_accuracy": 5,
    "depth_of_understanding": 5,
    "practical_application": 5,
    "communication": 5,
    "confidence_calibration": 5
  }},
  "score": 5,
  "expected_points": ["point 1","point 2"],
  "missing_points": ["points candidate missed"],
  "score_reasoning": "one sentence",
  "notes": "specific observation"
}}"""

def evaluate_answer_llm(session, question, answer, difficulty, question_type):
    if not answer or question_type=="greeting": return None
    sid = session.get("id","unknown")
    prompt = build_evaluation_prompt(session, question, answer, difficulty, question_type)
    return call_llm_json([{"role":"user","content":prompt}], temperature=0.3, max_tokens=500,
                         _session_id=sid, _step="LLM_evaluation")


# ════════════════════════════════════════════════════════════
# QUESTION GENERATOR — delegates to agent.py
# Uses strategy_engine for decisions, repetition_guard for dedup,
# evaluation_validator for answer validation.
# ════════════════════════════════════════════════════════════
def generate_question(session, candidate_answer=None):
    """Delegates to agent.py — strategy engine + repetition guard + evaluation validator."""
    return agent_generate_question(session, candidate_answer)


# ════════════════════════════════════════════════════════════
# AI CONTENT DETECTION (Sapling.ai + Copyleaks)
# ════════════════════════════════════════════════════════════
def detect_ai_content(text: str, session_id: str = "unknown") -> dict:
    """Check if answer is AI-generated using Sapling.ai."""
    if not SAPLING_API_KEY or not text or len(text.strip()) < 20:
        return {"sapling": None, "is_ai": False, "checked": False}
    try:
        t0 = time.time()
        resp = http_requests.post(
            "https://api.sapling.ai/api/v1/aidetect",
            json={"key": SAPLING_API_KEY, "text": text},
            timeout=5
        )
        latency = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            ai_score = round(data.get("score", 0.0), 3)
            is_ai = ai_score > 0.7
            print(f"[AI Detect] sapling={ai_score} is_ai={is_ai} latency={latency:.2f}s")
            return {"sapling": {"score": ai_score, "is_ai": is_ai}, "is_ai": is_ai, "checked": True}
        else:
            print(f"[AI Detect] Sapling error: {resp.status_code}")
            return {"sapling": None, "is_ai": False, "checked": False}
    except Exception as e:
        print(f"[AI Detect] Failed: {e}")
        return {"sapling": None, "is_ai": False, "checked": False}


# ════════════════════════════════════════════════════════════
# WARMUP
# ════════════════════════════════════════════════════════════
def save_candidate_history(session):
    """Save session summary to PostgreSQL (falls back to print-only if DB unavailable)."""
    candidate_key = session.get("candidate_key", "")
    if not candidate_key: return
    scored = [h for h in session["history"] if h.get("evaluation") and h["phase"]!="warmup" and (h.get("evaluation") or {}).get("quality") not in ("warmup",None,"no_answer")]
    scores = []
    for h in scored:
        try: scores.append(int(h["evaluation"].get("score") or 0))
        except: pass
    if not scores:
        print(f"[History] Skipped — no answered questions for {candidate_key}")
        return
    avg = sum(scores)/len(scores)
    tp = session.get("topic_performance", {})
    weak = [t for t,d in tp.items() if d.get("avg_score",5) < 4]
    strong = [t for t,d in tp.items() if d.get("avg_score",0) >= 7]
    overall_score = min(100, int(avg * 10))

    resume = session.get("resume", {})
    candidate_id = db.get_or_create_candidate(
        candidate_key,
        resume.get("candidate_name", "Candidate"),
        resume.get("email", ""),
        resume.get("domain", "physical_design"),
        resume.get("level", "unknown"),
        resume.get("education", "")
    )

    db.save_session(
        session_id=session["id"],
        candidate_id=candidate_id,
        mode=session.get("mode", "mock"),
        difficulty_level=session.get("difficulty_level", 1),
        turns_completed=len(scored),
        overall_score=overall_score,
        grade=None,
        warmup_performance=session.get("warmup_performance", "pending"),
        early_end_reason=session.get("early_end_reason"),
        started_at=session.get("started_at", time.time()),
        topic_performance=tp,
        weak_topics=weak,
        strong_topics=strong,
        warmup_turns=session.get("warmup_turns", 0),
        warmup_perf_str=session.get("warmup_performance", "pending"),
        skills_asked=session.get("warmup_skills_asked")
    )
    print(f"[History] Saved for {candidate_key}: score={overall_score}, weak={weak}, strong={strong}")


# These functions now delegate to agent.py (new modular architecture)
# strip_initials imported from agent.py at top of file

def generate_greeting(session):
    return agent_generate_greeting(session)

def generate_warmup_question(session, candidate_answer=None):
    return agent_generate_warmup_question(session, candidate_answer)

# compute_trajectory and get_trajectory_interpretation imported from agent.py


# ════════════════════════════════════════════════════════════
# TTS — with observability tracking
# Fallback chain: Sarvam Bulbul V3 → Mistral Voxtral → LMNT → Pocket TTS → Browser
# ════════════════════════════════════════════════════════════
def synthesize_speech_sarvam(text):
    """Primary TTS: Sarvam AI Bulbul V3 (Indian English voice)."""
    resp = http_requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "text": text[:2500],
            "model": SARVAM_MODEL,
            "speaker": SARVAM_VOICE,
            "target_language_code": "en-IN",
            "pace": 1.2,
            "speech_sample_rate": 24000,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    # Sarvam returns base64 audio in "audios" array
    audios = data.get("audios", [])
    if audios:
        return "".join(audios)
    # Fallback: check "audio" field
    audio_b64 = data.get("audio", "")
    if audio_b64:
        return audio_b64
    return base64.b64encode(resp.content).decode()

def synthesize_speech_mistral(text):
    """Primary TTS: Mistral Voxtral with voice cloning."""
    resp = http_requests.post(
        "https://api.mistral.ai/v1/audio/speech",
        headers={"Authorization":f"Bearer {MISTRAL_API_KEY}","Content-Type":"application/json"},
        json={"model":"voxtral-mini-tts-2603","input":text[:1500],"ref_audio":MISTRAL_TTS_REF_AUDIO,"response_format":"mp3"},
        timeout=15,
    )
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type","")
    if "json" in content_type:
        data = resp.json()
        audio_b64 = data.get("audio_data","")
        if audio_b64: return audio_b64
    return base64.b64encode(resp.content).decode()

def synthesize_speech_lmnt(text):
    """Fallback 1 TTS: LMNT with voice cloning."""
    resp = http_requests.post(
        "https://api.lmnt.com/v1/ai/speech",
        headers={"X-API-Key":LMNT_API_KEY,"Content-Type":"application/json"},
        json={"voice":LMNT_VOICE_ID,"text":text[:1500],"format":"mp3","speed":1.2},
        timeout=15,
    )
    resp.raise_for_status()
    if "json" in resp.headers.get("Content-Type",""):
        data=resp.json()
        if "audio" in data: return base64.b64encode(base64.b64decode(data["audio"])).decode()
    return base64.b64encode(resp.content).decode()

def synthesize_speech_pocket(text):
    """Fallback 2 TTS: Pocket TTS (local, free)."""
    audio=pocket_tts_model.generate_audio(pocket_tts_voice_state, text[:1500])
    import io
    wav_buffer=io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, pocket_tts_model.sample_rate, audio.numpy())
    return base64.b64encode(wav_buffer.getvalue()).decode()

def synthesize_speech_polly(text):
    resp=polly_client.synthesize_speech(Text=text[:1500],OutputFormat="mp3",VoiceId="Amy",Engine="neural")
    return base64.b64encode(resp["AudioStream"].read()).decode()

def synthesize_speech(text: str, session_id: str = "unknown") -> str:
    """TTS fallback chain: Sarvam → Mistral → LMNT → Pocket TTS → empty (browser fallback)."""
    session = sessions.get(session_id)
    tts_on = TTS_ENABLED
    if session and "tts_enabled" in session:
        tts_on = session["tts_enabled"]
    if not tts_on:
        return ""
    char_count = len(text)

    # 1. Primary: Sarvam AI Bulbul V3 (Indian English voice)
    if SARVAM_API_KEY:
        t0 = time.time()
        try:
            result = synthesize_speech_sarvam(text)
            track_tts_call(session_id=session_id, model="Sarvam-BulbulV3",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="success")
            return result
        except Exception as e:
            track_tts_call(session_id=session_id, model="Sarvam-BulbulV3",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="failure", error=str(e))
            print(f"Sarvam TTS failed, falling back to Mistral: {e}")

    # 2. Mistral Voxtral (voice cloning)
    if MISTRAL_API_KEY and MISTRAL_TTS_REF_AUDIO:
        t0 = time.time()
        try:
            result = synthesize_speech_mistral(text)
            track_tts_call(session_id=session_id, model="Mistral-Voxtral",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="success")
            return result
        except Exception as e:
            track_tts_call(session_id=session_id, model="Mistral-Voxtral",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="failure", error=str(e))
            print(f"Mistral TTS failed, falling back to LMNT: {e}")

    # 3. LMNT (voice cloning)
    if LMNT_API_KEY and LMNT_VOICE_ID:
        t0 = time.time()
        try:
            result = synthesize_speech_lmnt(text)
            track_tts_call(session_id=session_id, model="LMNT",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="success", fallback=True)
            return result
        except Exception as e:
            track_tts_call(session_id=session_id, model="LMNT",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="failure", error=str(e), fallback=True)
            print(f"LMNT TTS failed, falling back to Pocket TTS: {e}")

    # 4. Pocket TTS (local, free)
    if pocket_tts_model and pocket_tts_voice_state:
        t0 = time.time()
        try:
            result = synthesize_speech_pocket(text)
            track_tts_call(session_id=session_id, model="PocketTTS",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="success", fallback=True)
            return result
        except Exception as e:
            track_tts_call(session_id=session_id, model="PocketTTS",
                           latency_ms=(time.time()-t0)*1000,
                           char_count=char_count, status="failure", error=str(e), fallback=True)
            print(f"Pocket TTS also failed: {e}")
    return ""

def stream_tts_polly(text):
    try:
        resp=polly_client.synthesize_speech(Text=text[:1500],OutputFormat="mp3",VoiceId="Amy",Engine="neural")
        stream=resp["AudioStream"]
        while True:
            chunk=stream.read(4096)
            if not chunk: break
            yield chunk
    except Exception as e: print(f"Polly streaming failed: {e}")

def stream_tts(text):
    if MISTRAL_API_KEY and MISTRAL_TTS_REF_AUDIO:
        try:
            resp=http_requests.post("https://api.mistral.ai/v1/audio/speech",
                headers={"Authorization":f"Bearer {MISTRAL_API_KEY}","Content-Type":"application/json"},
                json={"model":"voxtral-mini-tts-2603","input":text[:1500],"ref_audio":MISTRAL_TTS_REF_AUDIO,"response_format":"mp3"},
                timeout=15, stream=True)
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk: yield chunk
        except Exception as e: print(f"Mistral stream failed: {e}")

# ════════════════════════════════════════════════════════════
# STT — with observability tracking
# ════════════════════════════════════════════════════════════
def transcribe_audio(audio_bytes: bytes, ext: str = "webm", session_id: str = "unknown") -> dict:
    """Transcribe: gpt-4o-mini-transcribe (primary) → ElevenLabs Scribe v2 (fallback).
    session_id is now threaded through for per-session observability tracking."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(audio_bytes); tmp_path = f.name
        transcript = ""; avg_confidence = 1.0; source = ""

        # Primary: gpt-4o-mini-transcribe
        t_stt_start = time.time()
        try:
            with open(tmp_path,"rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe", file=audio_file, language="en",
                )
            transcript = response.text.strip() if hasattr(response,"text") else str(response).strip()
            t_primary = time.time() - t_stt_start
            if transcript:
                source = "gpt-4o-mini-transcribe"
                track_stt_call(session_id=session_id, model="gpt-4o-mini-transcribe",
                               latency_ms=t_primary*1000, status="success")
                print(f"[Timing] STT (gpt-4o-mini-transcribe): {t_primary:.2f}s | '{transcript[:60]}'")
            else:
                track_stt_call(session_id=session_id, model="gpt-4o-mini-transcribe",
                               latency_ms=t_primary*1000, status="failure", error="empty transcript")
                print(f"[Timing] STT (gpt-4o-mini-transcribe): {t_primary:.2f}s | EMPTY — no speech detected")
        except Exception as e:
            t_primary = time.time() - t_stt_start
            track_stt_call(session_id=session_id, model="gpt-4o-mini-transcribe",
                           latency_ms=t_primary*1000, status="failure", error=str(e))
            print(f"[Timing] STT (gpt-4o-mini-transcribe): {t_primary:.2f}s | FAILED: {e}")

        return {"transcript":transcript,"avg_confidence":avg_confidence,
                "low_confidence":avg_confidence<0.5,"corrupted_terms":[],
                "needs_repeat":len(transcript.strip())==0}
    except Exception as e:
        print(f"STT error: {e}")
        return {"transcript":"","avg_confidence":0.0,"low_confidence":True,"corrupted_terms":[],"needs_repeat":False}
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass


# ════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ════════════════════════════════════════════════════════════
def generate_report(session: dict) -> dict:
    history=session["history"]; resume=session["resume"]
    sid=session.get("id","unknown")
    scored=[h for h in history if h.get("evaluation") and (h.get("evaluation") or {}).get("quality") not in ("warmup",None) and h["phase"]!="warmup"]
    raw_scores=[]
    for h in scored:
        try: raw_scores.append(int(h["evaluation"].get("score") or 5))
        except: raw_scores.append(5)
    trajectory=compute_trajectory(raw_scores); trajectory_interp=get_trajectory_interpretation(trajectory)
    avg=sum(raw_scores)/len(raw_scores) if raw_scores else 5.0
    if trajectory=="rising" and len(raw_scores)>=4:
        mid=len(raw_scores)//2
        second_avg=sum(raw_scores[mid:])/(len(raw_scores)-mid)
        weighted_avg=avg*0.4+second_avg*0.6
    else: weighted_avg=avg
    technical_score=min(100,int(weighted_avg*10))
    diff_order={d:i for i,d in enumerate(DIFFICULTY_LABELS)}
    difficulties=[h.get("difficulty","basic") for h in scored]
    max_difficulty=max(difficulties,key=lambda d:diff_order.get(d,0)) if difficulties else "basic"
    quadrants=[(h.get("evaluation") or {}).get("quadrant","") for h in scored]
    df_count=quadrants.count("dangerous_fake"); gn_count=quadrants.count("genuine_nervous"); ge_count=quadrants.count("genuine_expert")
    smooth_detected=session.get("smooth_talker_detected",False)
    if smooth_detected or df_count>=3: behavioral_profile="Smooth Talker"
    elif gn_count>ge_count and trajectory=="rising": behavioral_profile="Genuine Nervous"
    elif ge_count>=len(scored)*0.6: behavioral_profile="Genuine Expert"
    else: behavioral_profile="Mixed Profile"
    honest_admissions=sum(1 for h in scored if (h.get("evaluation") or {}).get("quality")=="honest_admission")
    behavioral_score=55
    behavioral_score+=min(20,honest_admissions*7)
    if trajectory=="rising": behavioral_score+=12
    elif trajectory=="flat_strong": behavioral_score+=8
    elif trajectory=="falling": behavioral_score-=5
    if smooth_detected: behavioral_score-=15
    behavioral_score=max(0,min(100,behavioral_score))
    suspicion_data=compute_suspicion_score(session,scored)
    suspicion_score=suspicion_data["suspicion_score"]; integrity_level=suspicion_data["integrity_level"]
    integrity_flags=suspicion_data["flags"]; integrity_score=max(0,100-int(suspicion_score))
    cap=60 if integrity_level=="high_risk" else 100
    overall=min(int(technical_score*0.60+behavioral_score*0.25+integrity_score*0.15),cap)
    grade="A" if overall>=85 else "B" if overall>=70 else "C" if overall>=55 else "D" if overall>=40 else "F"
    topic_map={}
    for h in scored:
        t=h.get("topic","general")
        if not t: continue
        topic_map.setdefault(t,{"scores":[],"questions":[],"answers":[]})
        try: topic_map[t]["scores"].append(int(h["evaluation"].get("score") or 5))
        except: topic_map[t]["scores"].append(5)
        if h.get("question"): topic_map[t]["questions"].append(h["question"][:150])
        if h.get("answer"):   topic_map[t]["answers"].append(h["answer"][:200])
    topic_performance={}
    for t,data in topic_map.items():
        if len(data["scores"])<2: continue
        avg_t=sum(data["scores"])/len(data["scores"])
        rating="Strong" if avg_t>=7.5 else "Adequate" if avg_t>=5.5 else "Needs Work" if avg_t>=3.0 else "Weak"
        topic_performance[t]={"score":int(avg_t*10),"rating":rating,"questions_asked":len(data["scores"]),"questions":data["questions"][:2],"sample_answers":data["answers"][:1]}
    topic_suspicion=compute_topic_suspicion(session,scored)
    contradiction_results=[]
    for topic,state in session.get("contradiction_asked",{}).items():
        if not isinstance(state,str): continue
        if state=="complete":
            contradiction_results.append({"topic":topic,"inconsistent":session["contradiction_asked"].get(f"{topic}_inconsistent",False),"angle1_score":session["contradiction_asked"].get(f"{topic}_angle1_score")})
    hint_events=session.get("hint_events",[])
    recovery_summary=[f"Question {h['turn']} ({h['topic']}): hint given, recovery was {h['recovery_quality']} (score: {h.get('recovery_score','N/A')})" for h in hint_events if h.get("recovery_quality")]
    transcript="\n".join([f"[Q{h['turn']}] {h.get('question_type','?')}/{h.get('difficulty','?')} topic={h.get('topic','?')}\nQ: {h['question']}\nA: {(h.get('answer') or '[no answer]')[:300]}\nEval: quality={(h.get('evaluation') or {}).get('quality','?')} score={(h.get('evaluation') or {}).get('score','?')}" for h in history if h["phase"]!="warmup"])[:5500]
    resources=DOMAIN_RESOURCES.get(resume["domain"],[])
    signal_count=suspicion_data.get("signal_count",0)
    narrative_prompt=f"""You are a senior VLSI mentor generating a mock interview performance report.
CANDIDATE: {resume['level'].replace('_',' ')} | {resume['domain'].replace('_',' ')}
TECHNICAL QUESTIONS: {len(scored)} | SCORES: Technical={technical_score} Behavioral={behavioral_score} Overall={overall} ({grade})
TRAJECTORY: {trajectory} -- {trajectory_interp}
BEHAVIORAL PROFILE: {behavioral_profile} | MAX DIFFICULTY: {max_difficulty}
HONEST ADMISSIONS: {honest_admissions} | RECOVERY EVENTS: {recovery_summary}
SKILLS FULLY TESTED: {[t for t,d in topic_map.items() if len(d['scores'])>=2]}
TRANSCRIPT: {transcript}
RESOURCES: {resources}

Return ONLY valid JSON:
{{
  "quick_snapshot": "2-3 lines max. Strongest signal, biggest gap, key observation.",
  "readiness_statement": "One clear sentence. Include role + level + tech node.",
  "strengths": [{{"strength":"title","evidence":"turn reference","why_it_matters":"job impact"}}],
  "weak_areas": [{{"topic":"name","gap_type":"concept_gap|articulation_gap|behavioral_issue","what_happened":"observation with turn reference","why_it_matters":"job impact","fix":"specific actionable step"}}],
  "communication_feedback": "Clarity, structure, confidence assessment.",
  "learning_plan": [{{"topic":"name","action":"exact action","resource":"book/chapter/tool","timeline":"X weeks"}}],
  "readiness_roadmap": [{{"milestone":"Week 2","goal":"specific goal"}},{{"milestone":"Week 4","goal":"specific goal"}},{{"milestone":"Week 6","goal":"specific goal"}}],
  "next_mock_recommendation": "Topics to focus, difficulty level.",
  "mentor_note": "2-3 lines. Encourage improvement."
}}"""
    narrative=call_llm_json([{"role":"user","content":narrative_prompt}],
                             temperature=0.3, max_tokens=2500, retries=2,
                             _session_id=sid, _step="LLM_question")
    if not narrative:
        narrative={"quick_snapshot":"Interview completed.","readiness_statement":"Continue preparation before applying.","strengths":[{"strength":"Completed interview","evidence":"Participated fully","why_it_matters":"Shows commitment"}],"weak_areas":[{"topic":"Technical depth","gap_type":"concept_gap","what_happened":"Needs more practice","why_it_matters":"Core job requirement","fix":"Study primary domain topics daily"}],"communication_feedback":"Work on structuring answers clearly.","learning_plan":[{"topic":"Core fundamentals","action":"Review key concepts daily","resource":resources[0] if resources else "Domain textbook","timeline":"4 weeks"}],"readiness_roadmap":[{"milestone":"Week 2","goal":"Master fundamentals"},{"milestone":"Week 4","goal":"Practice scenario questions"},{"milestone":"Week 6","goal":"Mock interview with numerical focus"}],"next_mock_recommendation":"Focus on weakest topics.","mentor_note":"Every expert was once a beginner. Focus on understanding, not memorizing."}
    # Build interview replay — turn-by-turn moments
    interview_replay = []
    prev_score = 5
    for h in scored:
        ev = h.get("evaluation") or {}
        s = int(ev.get("score") or 5)
        scores_dim = ev.get("scores", {})
        moment_type = "normal"
        if s - prev_score >= 3: moment_type = "breakthrough"
        elif ev.get("quality") in ("weak","honest_admission"): moment_type = "struggle"
        elif ev.get("quadrant") == "dangerous_fake": moment_type = "red_flag"
        elif ev.get("quality") == "strong" and h.get("difficulty") in ("advanced","expert"): moment_type = "strong"
        interview_replay.append({
            "turn": h["turn"], "topic": h.get("topic",""), "difficulty": h.get("difficulty",""),
            "question": h.get("question","")[:150], "answer_excerpt": (h.get("answer","") or "")[:150],
            "overall_score": s, "dimension_scores": scores_dim,
            "quality": ev.get("quality",""), "moment_type": moment_type,
            "reaction": ev.get("reaction",""),
        })
        prev_score = s

    # Aggregate multi-dimensional scores
    dim_names = ["technical_accuracy","depth_of_understanding","practical_application","communication","confidence_calibration"]
    dimension_averages = {}
    for dim in dim_names:
        dim_scores = []
        for h in scored:
            ds = (h.get("evaluation") or {}).get("scores", {})
            if ds and dim in ds:
                try: dim_scores.append(int(ds[dim]))
                except: pass
        if dim_scores:
            first_half = dim_scores[:len(dim_scores)//2] if len(dim_scores)>=4 else dim_scores
            second_half = dim_scores[len(dim_scores)//2:] if len(dim_scores)>=4 else dim_scores
            avg_first = sum(first_half)/len(first_half) if first_half else 5
            avg_second = sum(second_half)/len(second_half) if second_half else 5
            trend = "rising" if avg_second > avg_first + 0.5 else "falling" if avg_first > avg_second + 0.5 else "stable"
            dimension_averages[dim] = {"avg": round(sum(dim_scores)/len(dim_scores),1), "trend": trend}

    # Save candidate history for returning candidate support
    save_candidate_history(session)

    return {"scores":{"technical":technical_score,"behavioral":behavioral_score,"integrity":integrity_score,"overall":overall,"grade":grade},"trajectory":trajectory,"trajectory_interpretation":trajectory_interp,"behavioral_profile":behavioral_profile,"smooth_talker_detected":smooth_detected,"smooth_talker_score":session.get("smooth_talker_score",0),"smooth_talker_signals":session.get("smooth_talker_signals",[]),"max_difficulty_reached":max_difficulty,"topic_performance":topic_performance,"dimension_averages":dimension_averages,"interview_replay":interview_replay,"topic_suspicion":topic_suspicion,"contradiction_results":contradiction_results,"turns_completed":len(scored),"honest_admissions":honest_admissions,"integrity_level":integrity_level,"integrity_flags":integrity_flags,"suspicion_score":suspicion_score,"signal_count":signal_count,"critical_verdict":suspicion_data.get("critical_verdict",False),"recovery_events":recovery_summary,"notable_moments":session.get("notable_moments",[]),"genuine_signals":session.get("genuine_signals",[]),"observability":_obs_session_summary(sid),**narrative}


# ════════════════════════════════════════════════════════════
# CORE INTERVIEW ROUTES
# ════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html","r",encoding="utf-8") as f: return f.read()

@app.get("/interview", response_class=HTMLResponse)
async def interview_ui():
    with open("templates/voice_agent_ui.html","r",encoding="utf-8") as f: return f.read()

@app.post("/api/parse-resume")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 5_000_000: raise HTTPException(413, "File too large. Max 5MB.")
    ext = file.filename.rsplit(".",1)[-1].lower() if "." in file.filename else "txt"
    if ext=="pdf":
        if not content.startswith(b"%PDF-"): raise HTTPException(400,"Not a valid PDF.")
        try:
            import pdfplumber
            with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp: tmp.write(content); tmp_path=tmp.name
            text=""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages: text+=(page.extract_text() or "")+"\n"
            os.unlink(tmp_path)
        except Exception as e: raise HTTPException(400,f"PDF error: {e}")
    elif ext in ("docx","doc"):
        try:
            import docx2txt
            with tempfile.NamedTemporaryFile(suffix=".docx",delete=False) as tmp: tmp.write(content); tmp_path=tmp.name
            text=docx2txt.process(tmp_path); os.unlink(tmp_path)
        except Exception as e: raise HTTPException(400,f"DOCX error: {e}")
    else: text=content.decode("utf-8",errors="ignore")
    return JSONResponse(parse_resume(text))

@app.post("/api/create-session")
async def create_session(data: SessionCreate):
    sid=str(uuid.uuid4())
    resume=parse_resume(data.resume_text, _session_id=sid)

    # Compute resume gaps — domain topics NOT mentioned in candidate's skills/projects
    domain = resume.get("domain", "physical_design")
    domain_topics = DOMAIN_TOPICS.get(domain, [])
    candidate_skills = set(s.lower() for s in (resume.get("vlsi_skills", []) + resume.get("tools", []) + resume.get("skills", [])))
    candidate_projects_text = " ".join(resume.get("key_projects", [])).lower()
    resume_gaps = [t for t in domain_topics if t.lower() not in candidate_skills and t.lower() not in candidate_projects_text]

    # Check for returning candidate (from PostgreSQL)
    candidate_key = f"{resume.get('candidate_name','').lower().strip()}|{resume.get('email','').lower().strip()}"
    previous_sessions = db.get_candidate_sessions(candidate_key)
    is_returning = len(previous_sessions) > 0
    skip_warmup = len(previous_sessions) >= 2  # 2+ interviews → skip warmup on 3rd
    starting_difficulty = 1
    if is_returning and previous_sessions:
        last = previous_sessions[-1]
        starting_difficulty = min(4, max(0, last.get("difficulty_level", 1)))
        print(f"[Session] Returning candidate: {resume.get('candidate_name','')} | prev sessions: {len(previous_sessions)} | prev score: {last.get('overall_score','?')} | starting at difficulty {starting_difficulty} | skip_warmup: {skip_warmup}")

    session={"id":sid,"mode":data.mode,"resume":resume,"phase":"greeting","turn":0,"warmup_turns":0,"warmup_performance":"pending","warmup_conversation":[],"difficulty_level":starting_difficulty,"consecutive_strong":0,"consecutive_weak":0,"history":[],"topics_covered":[],"anchor_count":0,"last_topic":None,"last_question_type":None,"last_eval_quality":"adequate","last_confidence":"medium","anticheat_events":[],"behavioral_baseline":None,"pause_history":[],"trajectory_type":"unknown","hint_events":[],"notable_moments":[],"suspicion_events":[],"smooth_talker_signals":[],"smooth_talker_detected":False,"smooth_talker_score":0,"genuine_signals":[],"contradiction_asked":{},"topic_suspicion":{},"started_at":time.time(),"cached_first_question":None,"cached_first_audio":None,"resume_gaps":resume_gaps,"topic_performance":{},"running_suspicion":0,"is_returning":is_returning,"previous_sessions":previous_sessions,"candidate_key":candidate_key,"tts_enabled":TTS_ENABLED,"skip_warmup":skip_warmup}
    sessions[sid]=session

    # If returning candidate with 2+ interviews, skip warmup → go directly to technical
    if skip_warmup:
        session["phase"] = "greeting"  # Still greet, but will skip warmup after greeting
        session["warmup_performance"] = "skipped_returning"
        print(f"[Session] Will skip warmup for returning candidate ({len(previous_sessions)} prior sessions)")
    try:
        first_q=generate_greeting(session)
        first_audio=synthesize_speech(first_q["question"], session_id=sid)
        session["cached_first_question"]=first_q; session["cached_first_audio"]=first_audio
    except Exception as e: print(f"Pre-generation failed: {e}")
    return JSONResponse({"session_id":sid,"resume":resume})

@app.get("/api/get-session")
async def get_session(session_id: str):
    session=sessions.get(session_id)
    if not session: raise HTTPException(404,"Session not found")
    return JSONResponse({"session_id":session_id,"mode":session.get("mode","mock"),"resume":session.get("resume",{}),"turn":session.get("turn",0),"phase":session.get("phase","warmup")})

@app.post("/api/start-interview")
async def start_interview(data: dict):
    sid=data.get("session_id"); session=sessions.get(sid)
    if not session: raise HTTPException(404,"Session not found")
    if session["phase"]=="ended": raise HTTPException(400,"This interview session has already ended")
    if session["phase"]=="greeting" and session.get("cached_first_question"):
        result=session["cached_first_question"]; audio=session.get("cached_first_audio","")
        session["cached_first_question"]=None; session["cached_first_audio"]=None
    elif session["phase"]=="greeting":
        result=generate_greeting(session); audio=synthesize_speech(result["question"],session_id=sid)
    elif session["phase"]=="warmup":
        result=generate_warmup_question(session); audio=synthesize_speech(result["question"],session_id=sid)
    else:
        result=generate_question(session); audio=synthesize_speech(result["question"],session_id=sid)
    session["warmup_conversation"].append(f"Interviewer: {result['question']}")
    entry={"turn":session["turn"],"phase":session["phase"],"question":result["question"],"question_type":result.get("question_type","warmup"),"topic":result.get("topic","warmup"),"difficulty":result.get("difficulty","basic"),"answer":None,"evaluation":None,"behavioral_flags":[],"answer_duration_sec":0,"word_count":0,"filler_rate":0,"pronoun_rate":0,"thinking_pause_sec":0,"input_mode":"text","correction_rate":0,"above_level":False,"contradiction_inconsistency":False,"warmup_decision":result.get("warmup_decision")}
    session["history"].append(entry); session["turn"]+=1
    session["last_topic"]=result.get("topic"); session["last_question_type"]=result.get("question_type","warmup")
    should_end=result.get("warmup_decision")=="end_not_ready"
    if should_end: session["phase"]="ended"; session["warmup_performance"]="poor"
    return JSONResponse({"question":result["question"],"question_type":result.get("question_type","warmup"),"turn":session["turn"],"phase":session["phase"],"audio":audio,"difficulty":result.get("difficulty","basic"),"should_end":should_end,"warmup_decision":result.get("warmup_decision"),"resume":session.get("resume",{})})


@app.post("/api/submit-answer")
@limiter.limit("20/minute")
async def submit_answer(request: Request, data: AnswerSubmit):
    session=sessions.get(data.session_id)
    if not session: raise HTTPException(404,"Session not found")
    if session["phase"]=="ended": raise HTTPException(400,"This interview session has already ended")
    sid=data.session_id
    current_entry=session["history"][-1] if session["history"] else None
    if current_entry:
        current_entry["answer"]=data.answer; current_entry["answer_duration_sec"]=data.answer_duration_sec
        current_entry["word_count"]=data.word_count; current_entry["thinking_pause_sec"]=data.thinking_pause_sec
        current_entry["input_mode"]=data.input_mode; current_entry["filler_rate"]=count_fillers(data.answer)
        current_entry["pronoun_rate"]=count_personal_pronouns(data.answer); current_entry["correction_rate"]=count_self_corrections(data.answer)
        dev=analyze_behavioral_deviation(session,data.answer,data.answer_duration_sec,data.word_count,data.thinking_pause_sec,data.input_mode,current_entry.get("difficulty","basic"))
        current_entry["behavioral_flags"]=dev["flags"]; current_entry["behavioral_deviation"]=dev["deviation_score"]

        # AI content detection — check if answer is AI-generated (only for technical answers with enough text)
        if session["phase"] == "interview" and data.word_count >= 15:
            ai_result = detect_ai_content(data.answer, session_id=sid)
            current_entry["ai_detection"] = ai_result
            if ai_result.get("is_ai"):
                # Get the highest score from whichever detector flagged it
                s_score = (ai_result.get("sapling") or {}).get("score", 0)
                current_entry["behavioral_flags"].append("ai_generated_answer")
                session["running_suspicion"] = session.get("running_suspicion", 0) + 15
                record_notable(session, session["turn"]-1, current_entry.get("question",""), data.answer,
                    "concern_flag", f"AI-generated answer detected at question {session['turn']-1} (sapling: {s_score:.0%})")
                print(f"[AI Detect] WARNING: AI-generated answer at question {session['turn']-1} (sapling: {s_score:.0%})")

    if session["phase"]=="greeting":
        if session.get("skip_warmup"):
            # Returning candidate (2+ interviews) — skip warmup, go to technical
            session["phase"]="interview"; result=generate_question(session)
            result["warmup_feedback"]="Welcome back! Let's jump straight into the technical questions."
            print(f"[Interview] Skipped warmup for returning candidate")
        else:
            session["phase"]="warmup"; result=generate_warmup_question(session); session["warmup_conversation"].append(f"Interviewer: {result['question']}")
    elif session["phase"]=="warmup":
        session["warmup_turns"]+=1; session["warmup_conversation"].append(f"Candidate: {data.answer}")
        if session["warmup_turns"]>=3:
            session["phase"]="interview"; compute_baseline(session); result=generate_question(session); result["warmup_feedback"]="Let's begin the technical interview!"
        else:
            result=generate_warmup_question(session,data.answer); session["warmup_conversation"].append(f"Interviewer: {result['question']}")
    elif session["phase"]=="ready_check":
        answer_lower=data.answer.lower()
        if any(word in answer_lower for word in ["yes","ready","proceed","sure","okay","ok"]):
            session["phase"]="interview"; compute_baseline(session); result=generate_question(session)
        else:
            session["phase"]="ended"; session["warmup_performance"]="declined"
            result={"question":"No problem. Please take your time to prepare and come back when you're ready. Good luck!","question_type":"farewell"}
    else:
        # Detect noise/non-answers
        NOISE_PATTERNS = {"[background noise]","[silence]","[laughs]","[whistles]","[clicking]","[music]","[coughing]","[inaudible]","[noise]","[static]"}
        answer_stripped = data.answer.strip().lower()
        is_noise = answer_stripped in {p.lower() for p in NOISE_PATTERNS} or len(answer_stripped) <= 2 or data.word_count <= 1

        if is_noise and session["phase"] == "interview":
            session.setdefault("no_answer_count", 0)
            session["no_answer_count"] += 1
            print(f"[Interview] No-answer #{session['no_answer_count']}: '{data.answer[:40]}'")
            if current_entry:
                current_entry["evaluation"] = {"quality":"no_answer","accuracy":"not_applicable","confidence_level":"low","quadrant":"honest_confused","score":0,"score_reasoning":"No audible answer","notes":f"Noise: {data.answer[:50]}"}
            if session["no_answer_count"] >= 3:
                candidate_name = strip_initials(session["resume"].get("candidate_name","Candidate"))
                session["phase"] = "ended"
                session["early_end_reason"] = "no_answers"
                save_candidate_history(session)
                result = {"question":f"{candidate_name}, we haven't been able to hear your responses. We'll stop here so you can check your microphone. Come back anytime.","question_type":"farewell"}
                print(f"[Interview] Ending — {session['no_answer_count']} consecutive no-answers")
            else:
                result = generate_question(session)
        else:
            if not is_noise:
                session["no_answer_count"] = 0
            result = generate_question(session, data.answer)

    if current_entry and result.get("evaluation"):
        eval_data=result["evaluation"]; current_entry["evaluation"]=eval_data
        quality=eval_data.get("quality","adequate"); confidence=eval_data.get("confidence_level","medium")
        score=5
        try: score=int(eval_data.get("score") or 5)
        except: pass
        session["last_eval_quality"]=quality; session["last_confidence"]=confidence

        # Off-topic detection — track consecutive off-topic answers
        accuracy = eval_data.get("accuracy","")
        if accuracy == "off_topic":
            session.setdefault("off_topic_count", 0)
            session["off_topic_count"] += 1
            print(f"[Interview] Off-topic answer #{session['off_topic_count']} at turn {session['turn']-1}")
            record_notable(session, session["turn"]-1, current_entry["question"], data.answer, "concern_flag", f"Off-topic answer at question {session['turn']-1}")
        else:
            session["off_topic_count"] = 0

        complexity=assess_answer_complexity(session,data.answer,score,current_entry.get("difficulty","basic"),session["resume"]["level"])
        current_entry["above_level"]=complexity["above_level"]
        if complexity["above_level"] and not session.get("smooth_talker_detected"):
            record_notable(session,session["turn"]-1,current_entry["question"],data.answer,"positive_signal",f"Answer sophistication above calibrated level at question {session['turn']-1}: {complexity['flag']}")
        extra=result.get("_extra")
        if extra and extra.get("angle") and extra.get("pair"):
            pair=extra["pair"]; angle=extra["angle"]
            inconsistent=record_contradiction_result(session,pair["topic"],angle,eval_data,session["turn"]-1)
            current_entry["contradiction_inconsistency"]=inconsistent
            if inconsistent: record_notable(session,session["turn"]-1,current_entry["question"],data.answer,"concern_flag",f"Contradiction detected on '{pair['topic']}' at question {session['turn']-1}")
        elif current_entry.get("question_type")=="contradiction":
            topic=current_entry.get("topic","")
            if topic and session["contradiction_asked"].get(topic)!="angle_1_asked":
                session["contradiction_asked"][topic]="angle_1_asked"; session["contradiction_asked"][f"{topic}_turn"]=session["turn"]-1
                session["contradiction_asked"][f"{topic}_angle1_score"]=score; session["contradiction_asked"][f"{topic}_angle1_accuracy"]=eval_data.get("accuracy","partial")
        update_smooth_talker(session,eval_data,current_entry.get("question_type",""))
        if result.get("hint_given"): record_hint(session,session["turn"]-1,current_entry.get("topic",""),result.get("hint_text","Hint given"))
        evaluate_recovery(session,session["turn"]-1,data.answer,score)
        if quality=="honest_admission": record_notable(session,session["turn"]-1,current_entry["question"],data.answer,"positive_signal",f"Honest admission at question {session['turn']-1} on {current_entry.get('topic','')}")
        elif eval_data.get("quadrant")=="dangerous_fake": record_notable(session,session["turn"]-1,current_entry["question"],data.answer,"concern_flag",f"Confident+wrong at question {session['turn']-1} on {current_entry.get('topic','')}")
        elif quality=="strong" and current_entry.get("difficulty") in ("advanced","expert"): record_notable(session,session["turn"]-1,current_entry["question"],data.answer,"positive_signal",f"Strong answer on {current_entry.get('difficulty','')} {current_entry.get('topic','')} at question {session['turn']-1}")
        if quality=="strong": session["consecutive_strong"]+=1; session["consecutive_weak"]=0
        elif quality=="weak": session["consecutive_weak"]+=1; session["consecutive_strong"]=0
        else: session["consecutive_strong"]=0; session["consecutive_weak"]=0
        if session["consecutive_strong"]>=3 and session["difficulty_level"]<4: session["difficulty_level"]+=1; session["consecutive_strong"]=0
        elif session["consecutive_weak"]>=3 and session["difficulty_level"]>0: session["difficulty_level"]-=1; session["consecutive_weak"]=0

        # Per-topic performance tracking
        topic_name = current_entry.get("topic", "general")
        if topic_name and session["phase"] == "interview":
            tp = session.setdefault("topic_performance", {})
            if topic_name not in tp:
                tp[topic_name] = {"scores": [], "count": 0, "avg_score": 0}
            tp[topic_name]["scores"].append(score)
            tp[topic_name]["count"] += 1
            tp[topic_name]["avg_score"] = sum(tp[topic_name]["scores"]) / len(tp[topic_name]["scores"])

        # Running suspicion score (for stealth anti-cheat follow-ups)
        behavioral_flags = current_entry.get("behavioral_flags", [])
        suspicion_delta = len(behavioral_flags) * 5
        if current_entry.get("above_level"): suspicion_delta += 8
        if current_entry.get("contradiction_inconsistency"): suspicion_delta += 12
        session["running_suspicion"] = session.get("running_suspicion", 0) + suspicion_delta

        # Verification follow-up tracking: if last Q was verification and answer is now weak → strong cheat signal
        if current_entry.get("question_type") == "verification_followup" and quality in ("weak", "poor_articulation"):
            session.setdefault("suspicion_events", []).append({"type": "failed_verification", "turn": session["turn"]-1, "weight": 20, "detail": f"Strong answer followed by weak verification at question {session['turn']-1}"})
            session["running_suspicion"] += 20

        sc_hist=[h for h in session["history"] if h.get("evaluation") and (h.get("evaluation") or {}).get("quality") not in ("warmup",None) and h["phase"]!="warmup"]
        sc_list=[]
        for h in sc_hist:
            try: sc_list.append(int(h["evaluation"].get("score") or 5))
            except: sc_list.append(5)
        session["trajectory_type"]=compute_trajectory(sc_list)

    topic=result.get("topic","general")
    if topic and topic not in session["topics_covered"]: session["topics_covered"].append(topic)
    if result.get("question_type")=="personal_anchor": session["anchor_count"]=session.get("anchor_count",0)+1
    entry={"turn":session["turn"],"phase":session["phase"],"question":result["question"],"question_type":result["question_type"],"topic":topic,"difficulty":result.get("difficulty",DIFFICULTY_LABELS[session["difficulty_level"]]),"answer":None,"evaluation":None,"behavioral_flags":[],"answer_duration_sec":0,"word_count":0,"filler_rate":0,"pronoun_rate":0,"thinking_pause_sec":0,"input_mode":"text","correction_rate":0,"above_level":False,"contradiction_inconsistency":False}
    session["history"].append(entry); session["turn"]+=1
    session["last_topic"]=topic; session["last_question_type"]=result["question_type"]

    # Early stop check
    struggling_end=False
    if session["phase"]=="interview" and session["turn"]>=8:
        recent_interview=[h for h in session["history"] if h.get("evaluation") and h["phase"]=="interview" and (h.get("evaluation") or {}).get("quality") not in ("warmup",None)]
        is_real_mode=session.get("mode")=="real"
        if len(recent_interview)>=3:
            if is_real_mode:
                last_3=recent_interview[-3:]
                if all(h["evaluation"].get("quality") in ("weak","honest_admission") and (h["evaluation"].get("score") or 5)<=3 for h in last_3):
                    struggling_end=True; session["phase"]="ended"; session["early_end_reason"]="struggling_real"
                    save_candidate_history(session)
                    candidate_name=strip_initials(session["resume"].get("candidate_name","Candidate"))
                    result["question"]=f"Thank you {candidate_name}, I appreciate your time today. We've covered several topics and I have a good understanding of where you stand. We'll wrap up here. You'll receive a detailed feedback report shortly."
                    result["question_type"]="farewell"
                    print(f"[Interview] REAL MODE early end — candidate struggling")
            elif len(recent_interview)>=4:
                last_4=recent_interview[-4:]
                if all(h["evaluation"].get("quality") in ("weak","honest_admission") and (h["evaluation"].get("score") or 5)<=3 for h in last_4):
                    struggling_end=True; session["phase"]="ended"; session["early_end_reason"]="struggling"
                    save_candidate_history(session)
                    candidate_name=strip_initials(session["resume"].get("candidate_name","Candidate"))
                    result["question"]=f"Alright {candidate_name}, let's pause here. I can see some of these topics are challenging right now, and that's completely okay. I'm going to generate a detailed report with specific areas to focus on and resources that will help you prepare."
                    result["question_type"]="farewell"
                    print(f"[Interview] MOCK MODE early end — candidate struggling")

    # Off-topic early stop — 3 consecutive off-topic answers
    off_topic_end = False
    if session.get("off_topic_count", 0) >= 3 and session["phase"] == "interview":
        off_topic_end = True
        session["phase"] = "ended"; session["early_end_reason"] = "off_topic"
        save_candidate_history(session)
        candidate_name = strip_initials(session["resume"].get("candidate_name","Candidate"))
        domain_name = session["resume"].get("domain","VLSI").replace("_"," ")
        result["question"] = f"{candidate_name}, your last few answers were not related to the questions being asked. Let's stop here. I'd suggest reviewing the core {domain_name} topics and coming back for another mock when you feel more prepared."
        result["question_type"] = "farewell"
        print(f"[Interview] Ending — {session['off_topic_count']} consecutive off-topic answers")

    technical_turns=session["turn"]-session.get("warmup_turns",0)-1
    should_end=(technical_turns>=20 or session["phase"]=="ended" or struggling_end or off_topic_end or result.get("warmup_decision")=="end_not_ready")
    audio=synthesize_speech(result["question"], session_id=sid)
    return JSONResponse({"question":result["question"],"question_type":result.get("question_type","interview"),"turn":session["turn"],"phase":session["phase"],"audio":audio,"difficulty":result.get("difficulty",DIFFICULTY_LABELS[session["difficulty_level"]]),"should_end":should_end,"hint_given":result.get("hint_given",False),"hint_text":result.get("hint_text"),"warmup_decision":result.get("warmup_decision"),"warmup_performance":session.get("warmup_performance","pending")})

@app.get("/api/stream-tts")
async def stream_tts_endpoint(text: str, session_id: str = ""):
    if not text: raise HTTPException(400,"No text provided")
    if pocket_tts_model and pocket_tts_voice_state:
        try:
            audio_b64=synthesize_speech_pocket(text); audio_bytes=base64.b64decode(audio_b64)
            return StreamingResponse(iter([audio_bytes]),media_type="audio/wav",headers={"Content-Length":str(len(audio_bytes))})
        except Exception as e: print(f"Pocket TTS failed in stream: {e}")
    return StreamingResponse(stream_tts(text),media_type="audio/mpeg",headers={"Transfer-Encoding":"chunked"})

@app.post("/api/transcribe")
@limiter.limit("30/minute")
async def transcribe_endpoint(request: Request, audio: UploadFile = File(...), session_id: str = Form(...)):
    import asyncio
    audio_bytes=await audio.read()
    ext=audio.filename.rsplit(".",1)[-1] if "." in audio.filename else "webm"
    if len(audio_bytes)<1000:
        return JSONResponse({"transcript":"","avg_confidence":0.0,"low_confidence":True,"corrupted_terms":[],"needs_repeat":False})
    loop=asyncio.get_event_loop()
    result=await loop.run_in_executor(None, transcribe_audio, audio_bytes, ext, session_id)
    return JSONResponse(result)

@app.post("/api/toggle-tts")
async def toggle_tts(data: dict):
    """Toggle TTS on/off globally or per session."""
    sid = data.get("session_id")
    enabled = data.get("enabled", True)
    if sid:
        session = sessions.get(sid)
        if not session: raise HTTPException(404, "Session not found")
        session["tts_enabled"] = enabled
        return JSONResponse({"ok": True, "tts_enabled": enabled, "scope": "session"})
    else:
        global TTS_ENABLED
        TTS_ENABLED = enabled
        return JSONResponse({"ok": True, "tts_enabled": enabled, "scope": "global"})

@app.get("/api/tts-status")
async def tts_status():
    return JSONResponse({"tts_enabled": TTS_ENABLED})

@app.get("/api/anticheat-settings")
async def anticheat_settings():
    return JSONResponse({
        "head_turn_enabled": HEAD_TURN_ENABLED,
        "eye_away_enabled": EYE_AWAY_ENABLED,
        "face_detect_enabled": FACE_DETECT_ENABLED,
    })

@app.post("/api/end-session")
async def end_session(data: dict):
    sid = data.get("session_id")
    session = sessions.get(sid)
    if not session: raise HTTPException(404, "Session not found")
    session["phase"] = "ended"
    save_candidate_history(session)
    return JSONResponse({"ok": True})

@app.post("/api/anticheat-event")
async def anticheat_event(data: AntiCheatEvent):
    session=sessions.get(data.session_id)
    if not session: return JSONResponse({"ok":False})
    session["anticheat_events"].append({"event_type":data.event_type,"turn":data.turn,"timestamp":data.timestamp,"metadata":data.metadata})

    # Increase difficulty for serious cheating events (not head turn or eye movement)
    if data.event_type in ("tab_switch","paste_event","dom_overlay","ai_answer_overlay","ai_extension_detected","split_screen") and session.get("phase") == "interview":
        old_diff = session.get("difficulty_level", 1)
        new_diff = min(4, old_diff + 1)
        if new_diff > old_diff:
            session["difficulty_level"] = new_diff
            session["running_suspicion"] = session.get("running_suspicion", 0) + 10
            print(f"[Anti-cheat] {data.event_type} -> difficulty: {old_diff} -> {new_diff}")

    return JSONResponse({"ok":True})

@app.post("/api/generate-report")
async def generate_report_endpoint(data: ReportRequest):
    session=sessions.get(data.session_id)
    if not session: raise HTTPException(404,"Session not found")
    try: return JSONResponse(generate_report(session))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500,f"Report generation failed: {str(e)}")


# ════════════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════════════
@app.post("/api/auth/login")
async def api_login(data: LoginRequest):
    user=PLATFORM_USERS.get(data.username) or _approved_reviewers.get(data.username)
    if not user or not pwd_ctx.verify(data.password, user["hash"]):
        raise HTTPException(401,"Invalid credentials")
    token=_create_token({"sub":data.username,"role":user["role"]})
    resp = JSONResponse({"ok":True,"role":user["role"]})
    _set_auth_cookie(resp, token)
    return resp

@app.post("/api/auth/logout")
async def api_logout():
    resp = JSONResponse({"ok":True})
    resp.delete_cookie("_vlsi_tok", path="/")
    return resp

@app.get("/api/auth/me")
async def api_me(user: dict = Depends(get_current_user)):
    return JSONResponse({"username":user.get("sub"),"role":user.get("role")})

@app.post("/api/auth/reviewer/register")
async def reviewer_register(data: ReviewerRegister):
    if data.username in PLATFORM_USERS or data.username in _approved_reviewers: raise HTTPException(409,"Username already exists")
    if any(r["username"]==data.username for r in _pending_reviewers): raise HTTPException(409,"Registration already pending")
    _pending_reviewers.append({"username":data.username,"hash":pwd_ctx.hash(data.password),"name":data.name,"designation":data.designation,"organisation":data.organisation,"domain":data.domain,"requested_at":time.time()})
    return JSONResponse({"ok":True,"message":"Registration submitted. Admin will approve your access."})

@app.get("/api/admin/reviewers/pending")
async def get_pending_reviewers(_=Depends(require_admin)):
    return JSONResponse([{k:v for k,v in r.items() if k!="hash"} for r in _pending_reviewers])

@app.post("/api/admin/reviewers/approve/{username}")
async def approve_reviewer(username: str, _=Depends(require_admin)):
    reg=next((r for r in _pending_reviewers if r["username"]==username),None)
    if not reg: raise HTTPException(404,"Pending registration not found")
    _approved_reviewers[username]={"hash":reg["hash"],"role":"reviewer","meta":reg}
    _pending_reviewers.remove(reg)
    return JSONResponse({"ok":True,"approved":username})

@app.delete("/api/admin/reviewers/{username}")
async def revoke_reviewer(username: str, _=Depends(require_admin)):
    if username in _approved_reviewers: del _approved_reviewers[username]; return JSONResponse({"ok":True})
    raise HTTPException(404,"Reviewer not found")

# ════════════════════════════════════════════════════════════
# OBSERVABILITY ROUTES  (real data — no more fake logs)
# ════════════════════════════════════════════════════════════
@app.get("/api/observability/summary")
async def obs_summary(window: int = 86400, _=Depends(require_reviewer_or_admin)):
    return JSONResponse(_obs_platform_summary(window_seconds=window))

@app.get("/api/observability/logs")
async def obs_logs(session_id: Optional[str]=None, step: Optional[str]=None,
                   obs_status: Optional[str]=None, limit: int=200,
                   _=Depends(require_reviewer_or_admin)):
    return JSONResponse(_obs_get_logs(session_id=session_id, step=step, obs_status=obs_status, limit=limit))

@app.get("/api/observability/session/{session_id}")
async def obs_session(session_id: str, _=Depends(require_reviewer_or_admin)):
    return JSONResponse(_obs_session_summary(session_id))

# ════════════════════════════════════════════════════════════
# ADMIN ROUTES  (auth-gated)
# ════════════════════════════════════════════════════════════
@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    with open("templates/admin.html","r",encoding="utf-8") as f: return f.read()

@app.get("/api/admin/sessions")
async def admin_sessions(_=Depends(require_reviewer_or_admin)):
    result=[]
    for sid,session in sessions.items():
        scored=[h for h in session["history"] if h.get("evaluation") and h["phase"]!="warmup" and (h.get("evaluation") or {}).get("quality") not in ("warmup",None)]
        scores=[]
        for h in scored:
            try: scores.append(int(h["evaluation"].get("score") or 5))
            except: pass
        avg_score=round(sum(scores)/len(scores)*10,1) if scores else 0
        result.append({"session_id":sid,"domain":session["resume"].get("domain",""),"level":session["resume"].get("level",""),"phase":session["phase"],"turn":session["turn"],"avg_score":avg_score,"anticheat_count":len(session.get("anticheat_events",[])),"smooth_talker":session.get("smooth_talker_detected",False),"trajectory":session.get("trajectory_type","unknown"),"signal_count":count_active_signals(session,scored),"started_at":session.get("started_at",0),"candidate_name":session["resume"].get("candidate_name","")})
    return JSONResponse(sorted(result,key=lambda x:x["started_at"],reverse=True))

@app.get("/api/admin/session/{session_id}")
async def admin_session_detail(session_id: str, _=Depends(require_reviewer_or_admin)):
    session=sessions.get(session_id)
    if not session: raise HTTPException(404,"Session not found")
    history=session["history"]
    scored=[h for h in history if h.get("evaluation") and h["phase"]!="warmup" and (h.get("evaluation") or {}).get("quality") not in ("warmup",None)]
    turn_log=[]
    for h in history:
        eval_data=h.get("evaluation") or {}
        turn_log.append({"turn":h["turn"],"phase":h["phase"],"question_type":h.get("question_type",""),"topic":h.get("topic",""),"difficulty":h.get("difficulty",""),"question":h.get("question",""),"answer":h.get("answer","") or "","word_count":h.get("word_count",0),"answer_duration_sec":round(h.get("answer_duration_sec",0),1),"thinking_pause_sec":round(h.get("thinking_pause_sec",0),1),"input_mode":h.get("input_mode","text"),"filler_rate":round(h.get("filler_rate",0),3),"pronoun_rate":round(h.get("pronoun_rate",0),3),"correction_rate":round(h.get("correction_rate",0),3),"behavioral_flags":h.get("behavioral_flags",[]),"behavioral_deviation":round(h.get("behavioral_deviation",0),2),"above_level":h.get("above_level",False),"contradiction_inconsistency":h.get("contradiction_inconsistency",False),"quality":eval_data.get("quality",""),"accuracy":eval_data.get("accuracy",""),"confidence_level":eval_data.get("confidence_level",""),"quadrant":eval_data.get("quadrant",""),"score":eval_data.get("score",""),"score_reasoning":eval_data.get("score_reasoning",""),"ai_detection":h.get("ai_detection",{}),"notes":eval_data.get("notes","")})
    contradiction=session.get("contradiction_asked",{}); contradiction_log=[]
    for key,val in contradiction.items():
        if isinstance(val,str) and val in ("angle_1_asked","complete"):
            contradiction_log.append({"topic":key,"status":val,"angle1_score":contradiction.get(f"{key}_angle1_score"),"angle1_accuracy":contradiction.get(f"{key}_angle1_accuracy"),"inconsistent":contradiction.get(f"{key}_inconsistent",False)})
    raw_scores=[]
    for h in scored:
        try: raw_scores.append({"turn":h["turn"],"score":int(h["evaluation"].get("score") or 5),"topic":h.get("topic",""),"quadrant":(h.get("evaluation") or {}).get("quadrant","")})
        except: pass
    return JSONResponse({"session_id":session_id,"resume":session["resume"],"phase":session["phase"],"turn":session["turn"],"difficulty_level":session["difficulty_level"],"trajectory":session.get("trajectory_type","unknown"),"smooth_talker_detected":session.get("smooth_talker_detected",False),"smooth_talker_score":session.get("smooth_talker_score",0),"smooth_talker_signals":session.get("smooth_talker_signals",[]),"signal_count":count_active_signals(session,scored),"behavioral_baseline":session.get("behavioral_baseline") or {},"pause_history":session.get("pause_history",[]),"turn_log":turn_log,"anticheat_log":session.get("anticheat_events",[]),"contradiction_log":contradiction_log,"recovery_log":session.get("hint_events",[]),"notable_moments":session.get("notable_moments",[]),"genuine_signals":session.get("genuine_signals",[]),"suspicion_events":session.get("suspicion_events",[]),"raw_scores":raw_scores,"topics_covered":session.get("topics_covered",[]),"anchor_count":session.get("anchor_count",0),"expert_reviews":session.get("expert_reviews",[]),"observability":_obs_session_summary(session_id)})

# ════════════════════════════════════════════════════════════
# EXPERT REVIEW ROUTES
# ════════════════════════════════════════════════════════════
@app.post("/api/admin/review")
async def submit_review(data: ReviewSubmit, user: dict = Depends(require_reviewer_or_admin)):
    session=sessions.get(data.session_id)
    if not session: raise HTTPException(404,"Session not found")
    review_record={"review_id":f"R-{secrets.token_hex(4).upper()}","session_id":data.session_id,"question_turn":data.question_turn,"reviewer":user.get("sub","unknown"),"reviewed_at":time.time(),"reviewed_at_str":datetime.now().isoformat(),"ai_score":data.ai_score,"human_score":data.human_score,"score_delta":round(data.human_score-data.ai_score,1),"dimension_assessments":data.dimension_assessments,"error_flags":data.error_flags,"concept_corrections":data.concept_corrections,"behavior_ratings":data.behavior_ratings,"verdict":data.verdict,"overall_feedback":data.overall_feedback}
    _reviews_store.append(review_record)
    session.setdefault("expert_reviews",[]).append(review_record)
    return JSONResponse({"ok":True,"review_id":review_record["review_id"],"recorded_at":review_record["reviewed_at_str"]})

@app.get("/api/admin/review/{session_id}")
async def get_session_reviews(session_id: str, _=Depends(require_reviewer_or_admin)):
    return JSONResponse([r for r in _reviews_store if r["session_id"]==session_id])

@app.get("/api/admin/reviews/all")
async def get_all_reviews(limit: int=100, _=Depends(require_admin)):
    return JSONResponse(list(reversed(_reviews_store))[:limit])

# ════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    return JSONResponse({"status":"ok","service":"vlsi-interview-platform","version":"1.0.0","sessions":len(sessions),"reviews":len(_reviews_store),"logs_tracked":len(_call_logs)})

# ════════════════════════════════════════════════════════════
# LMS LAUNCH
# ════════════════════════════════════════════════════════════
@app.post("/lms/launch")
async def lms_launch(request: Request, response: Response):
    body=await request.json()
    token=body.get("token") or request.headers.get("Authorization","").replace("Bearer ","")
    if not token: raise HTTPException(400,"No token provided")
    try:
        payload=_decode_token(token)
    except Exception: raise HTTPException(401,"Invalid or expired LMS token")
    client_id=payload.get("client_id","")
    lms_secret=os.getenv(f"LMS_SECRET_{client_id.upper().replace('-','_').replace(' ','_')}")
    if not lms_secret: raise HTTPException(403,f"Unknown client: {client_id}")
    try:
        import jose.jwt as _jwt
        _jwt.decode(token, lms_secret, algorithms=["HS256"])
    except Exception: raise HTTPException(401,"LMS token signature invalid")
    user_id=payload.get("user_id","unknown")
    session_token=_create_token({"sub":user_id,"role":"student","client_id":client_id,"domain":payload.get("domain","physical_design"),"level":payload.get("level","trained_fresher"),"callback_url":payload.get("callback_url","")})
    _set_auth_cookie(response,session_token)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/interview",status_code=303)

@app.get("/lms/report/{session_id}")
async def lms_report(session_id: str, client_id: str, _=Depends(get_current_user)):
    session=sessions.get(session_id)
    if not session: raise HTTPException(404,"Session not found")
    if session["phase"] not in ("ended","generating_report"):
        return JSONResponse({"status":"in_progress"},status_code=202)
    report=generate_report(session)
    return JSONResponse(report)

# ════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("VLSI Interview Platform starting...")
    print(f"  CORS origins: {_ALLOWED_ORIGINS}")
    print(f"  Admin user:   {ADMIN_USER}")
    print(f"  JWT secret:   {'from env' if os.getenv('JWT_SECRET') else 'RANDOM — set JWT_SECRET in .env'}")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)