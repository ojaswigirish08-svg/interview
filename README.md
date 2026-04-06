# VLSI Interview Agent

AI-powered mock interview platform for VLSI engineers. Adaptive difficulty, domain-specific intelligence, behavioral analysis, silent anti-cheat, and detailed performance reports.

---

## Quick Start

### 1. Extract and Setup

```cmd
setup.bat
```

This installs all dependencies and creates a `.env` file.

### 2. Fill in API Keys

Open `.env` and add:

| Key | Where to get it |
|-----|----------------|
| `GROQ_API_KEY` | https://console.groq.com — free tier |
| `AWS_ACCESS_KEY_ID` | AWS IAM console — AmazonPollyReadOnlyAccess |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM console |

### 3. Run

```cmd
venv\Scripts\activate
python main.py
```

### 4. Open Browser

```
http://localhost:8001
```

---

## How It Works

1. **Upload Resume** — Agent parses domain (Analog Layout / Physical Design / Design Verification), experience level, and tools
2. **Camera Preview** — Allow mic and camera access for full experience (text fallback available)
3. **Warmup Phase** — 2 casual questions to build comfort and establish behavioral baseline
4. **Technical Interview** — 18-20 adaptive questions calibrated to your profile
   - Starts at your level, escalates/drops based on performance
   - Definition → Scenario follow-ups
   - Personal anchor questions ("Tell me about a specific time...")
   - Numerical probing ("What numbers did you work with?")
   - Contradiction traps (consistency check across turns)
5. **Report** — Technical score, behavioral score, integrity score, topic performance chart, strengths, improvements, roadmap

---

## Three VLSI Domains

| Domain | Topics |
|--------|--------|
| **Analog Layout** | Matching, parasitics, DRC/LVS, guard rings, ESD, Virtuoso |
| **Physical Design** | Floorplan, CTS, timing closure, STA, ICC2/Innovus |
| **Design Verification** | UVM, SVA, coverage, simulation/formal, VCS/Questa |

---

## Tech Stack

- **Backend**: FastAPI + Python
- **LLM**: Groq (llama-3.3-70b-versatile) — free tier
- **STT**: OpenAI Whisper (runs locally, no API cost)
- **TTS**: AWS Polly Neural (Joanna voice)
- **Frontend**: Pure HTML/CSS/JS, no frameworks

---

## Files

```
vlsi_interview/
├── main.py              # FastAPI backend (LLM, STT, TTS, sessions)
├── templates/
│   └── index.html       # Complete frontend (single file)
├── static/              # Static assets (empty by default)
├── requirements.txt     # Python dependencies
├── setup.bat            # Windows setup script
├── .env.example         # Environment template
└── README.md            # This file
```

---

## Notes

- Whisper model downloads on first run (~150MB for "base"). This is a one-time download.
- AWS Polly neural voices have a small cost per character. ~$16 per 1M characters. A 30-min interview is roughly $0.02.
- Sessions are in-memory only — restarting the server clears all sessions.
- Camera is captured but only periodic analysis is used internally. Full video is NOT uploaded.
