"""
config.py — single source of truth for env vars, AI clients, and domain constants.

To add a new AI provider: add client init here. Every module imports from here.
To change domain topics: edit DOMAIN_TOPICS / DOMAIN_RESOURCES.
To add a new difficulty level: extend DIFFICULTY_LABELS and update matrices in agent/evaluator.py.
"""

import os, base64
from dotenv import load_dotenv

load_dotenv()

# ── Environment ───────────────────────────────────────────────────────────────
ENV            = os.getenv("ENV", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8001").split(",")

# ── AI keys ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
CEREBRAS_API_KEY  = os.getenv("CEREBRAS_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
LMNT_API_KEY      = os.getenv("LMNT_API_KEY", "")
LMNT_VOICE_ID     = os.getenv("LMNT_VOICE_ID", "")
MISTRAL_API_KEY   = os.getenv("MISTRAL_API_KEY", "")
AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")

# ── Auth ──────────────────────────────────────────────────────────────────────
import secrets
JWT_SECRET     = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGO       = "HS256"
JWT_EXPIRE_MIN = 480

ADMIN_USER    = os.getenv("ADMIN_USER",    "admin")
ADMIN_PASS    = os.getenv("ADMIN_PASS",    "changeme_before_deploy")
REVIEWER_USER = os.getenv("REVIEWER_USER", "reviewer")
REVIEWER_PASS = os.getenv("REVIEWER_PASS", "changeme_before_deploy")

# ── AI Clients ────────────────────────────────────────────────────────────────
import boto3, requests as http_requests
from openai import OpenAI

openai_client = OpenAI(api_key=OPENAI_API_KEY)

polly_client = boto3.client(
    "polly", region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

cerebras_client = None
if CEREBRAS_API_KEY:
    cerebras_client = OpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1")
    print("Cerebras LLM ready.")

elevenlabs_client = None
if ELEVENLABS_API_KEY:
    from elevenlabs.client import ElevenLabs
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    print("ElevenLabs Scribe v2 ready (fallback STT).")

# ── TTS reference audio ───────────────────────────────────────────────────────
_ref_audio_path = os.path.join(os.path.dirname(__file__), "ranjitha_4dmjitkw.mp3")
MISTRAL_TTS_REF_AUDIO = None
if MISTRAL_API_KEY and os.path.exists(_ref_audio_path):
    with open(_ref_audio_path, "rb") as _f:
        MISTRAL_TTS_REF_AUDIO = base64.b64encode(_f.read()).decode()

# Pocket TTS
pocket_tts_model = None
pocket_tts_voice_state = None
try:
    from pocket_tts import TTSModel, export_model_state
    import scipy.io.wavfile, numpy as np
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

# ── TTS toggle ────────────────────────────────────────────────────────────────
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"

# ── AI content detection ──────────────────────────────────────────────────────
SAPLING_API_KEY = os.getenv("SAPLING_API_KEY", "")
if SAPLING_API_KEY:
    print("Sapling AI content detector ready.")

COPYLEAKS_API_KEY = os.getenv("COPYLEAKS_API_KEY", "")
COPYLEAKS_EMAIL = os.getenv("COPYLEAKS_EMAIL", "")

# ── Interview constants ───────────────────────────────────────────────────────
DIFFICULTY_LABELS = ["foundational", "basic", "intermediate", "advanced", "expert"]
FILLER_WORDS      = {"um","uh","like","basically","actually","so","right","okay","hmm","err"}
EXPECTED_PAUSE_BY_DIFFICULTY = {
    "foundational": 2.0, "basic": 3.0, "intermediate": 5.0, "advanced": 8.0, "expert": 10.0
}

# ── Domain knowledge ──────────────────────────────────────────────────────────
# To add a new domain: add an entry to all three dicts below.
DOMAIN_TOPICS = {
    "analog_layout": [
        "basic layout concepts","device matching","parasitic awareness",
        "latch-up and ESD","guard rings","DRC/LVS","symmetry techniques",
        "shielding and routing","technology awareness",
    ],
    "physical_design": [
        "floorplanning","power planning","placement","clock tree synthesis",
        "routing","timing closure","static timing analysis","DRC/LVS signoff","tool knowledge",
    ],
    "design_verification": [
        "verification methodologies","testbench architecture","functional coverage",
        "assertions and SVA","simulation vs formal","debugging skills",
        "protocol knowledge","regression and signoff","UVM",
    ],
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
    ],
}

CONTRADICTION_PAIRS = {
    "analog_layout": [
        {"topic":"device matching",
         "angle_1":"What techniques do you use to achieve good device matching in layout?",
         "angle_2":"If two devices have identical layout but different orientations relative to the gradient, will they match? Why or why not?"},
        {"topic":"parasitic awareness",
         "angle_1":"How do you minimize parasitic capacitance in a layout?",
         "angle_2":"In a situation where two nets run parallel for 100 microns, what is the dominant parasitic concern and how would you quantify it?"},
        {"topic":"latch-up and ESD",
         "angle_1":"What causes latch-up in CMOS layout and how do you prevent it?",
         "angle_2":"If you have a circuit where the substrate contact spacing is 50 microns from the nearest NMOS, is that acceptable? Walk me through your reasoning."},
    ],
    "physical_design": [
        {"topic":"clock tree synthesis",
         "angle_1":"What is clock tree synthesis and what problem does it solve?",
         "angle_2":"If after CTS you have 200ps of skew on one branch, what are the first three things you would check and in what order?"},
        {"topic":"timing closure",
         "angle_1":"Explain the difference between setup and hold violations.",
         "angle_2":"You have a hold violation of 50ps on a path that passes through three buffers. Adding more buffers made it worse. What is happening and what is the correct fix?"},
        {"topic":"floorplanning",
         "angle_1":"What are the key objectives of floorplanning in physical design?",
         "angle_2":"You are floorplanning a block with 60% utilization and your timing is already tight. Where exactly would you place the critical path macros and why?"},
    ],
    "design_verification": [
        {"topic":"functional coverage",
         "angle_1":"What is functional coverage and why is it important in verification?",
         "angle_2":"Your regression shows 98% functional coverage but you still found a bug in silicon. How is that possible and what does it tell you about your coverage model?"},
        {"topic":"assertions and SVA",
         "angle_1":"What is the difference between a concurrent and immediate assertion in SVA?",
         "angle_2":"You have an assertion that never fires during simulation. Is that good or bad? How do you determine which it is?"},
        {"topic":"simulation vs formal",
         "angle_1":"What are the advantages of formal verification over simulation?",
         "angle_2":"Formal verification proved your design is correct but you still found a bug. What are three possible explanations?"},
    ],
}