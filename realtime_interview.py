"""
realtime_interview.py — OpenAI Realtime API Interview Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Separate from main.py — uses OpenAI Realtime API for low-latency voice interviews.

Architecture:
  Browser (mic/speaker) ←WebSocket→ This Server ←WebSocket→ OpenAI Realtime API
                                         ↕
                                  Tool calls (evaluate_answer, get_next_topic)

How to run:
  pip install fastapi uvicorn websockets
  python realtime_interview.py

Then open: http://localhost:8002
"""

import os
import json
import time
import asyncio
import base64
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
try:
    import websockets
    from websockets.asyncio.client import connect as ws_connect
except ImportError:
    import websockets
    ws_connect = websockets.connect

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_MODEL = "gpt-4o-mini-realtime-preview"
VOICE = "alloy"  # alloy, echo, fable, onyx, nova, shimmer

app = FastAPI(title="VLSI Interview — Realtime Voice")

# ════════════════════════════════════════════════════════════
# INTERVIEW STATE
# ════════════════════════════════════════════════════════════
sessions = {}

def get_system_prompt(domain="physical_design", level="trained_fresher", candidate_name="Candidate"):
    return f"""You are Ranjitha, a senior {domain.replace('_',' ')} engineer conducting a mock interview.

PERSONALITY:
- Warm but technically rigorous
- React specifically to what the candidate says (never generic "Good answer")
- Push for numbers and specifics when answers are vague
- If candidate says "I don't know" — validate their honesty and give a brief hint

CANDIDATE: {candidate_name} | {domain.replace('_',' ')} | {level.replace('_',' ')}

INTERVIEW FLOW:
1. Start with a greeting and ask them to introduce themselves
2. Ask warmup questions about their resume/experience (2-3 questions)
3. Move to technical questions — start easy, increase difficulty
4. After each answer, call the 'evaluate_answer' tool to score it
5. Use the evaluation result to decide difficulty of next question
6. After 15-20 questions, end with encouragement

RULES:
- ONE question at a time — max 2 sentences
- React to their previous answer before asking next question
- Include numerical questions periodically ("What skew target?", "What slack value?")
- If they struggle 3 times, simplify
- If they're strong, push into edge cases
- Sound natural — you are speaking, not writing

DOMAIN TOPICS ({domain.replace('_',' ')}):
{"STA, synthesis, floorplanning, placement, CTS, routing, congestion, IR drop, timing closure" if domain == "physical_design" else "MOSFET, matching, parasitics, LDE, guard rings, DRC/LVS, symmetry" if domain == "analog_layout" else "SystemVerilog, UVM, assertions, coverage, testbench, simulation, formal verification"}

IMPORTANT: After EVERY candidate answer, you MUST call the 'evaluate_answer' tool before responding. Use the score to adapt your next question's difficulty."""


# ════════════════════════════════════════════════════════════
# TOOLS — called by OpenAI during the conversation
# ════════════════════════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "name": "evaluate_answer",
        "description": "Evaluate the candidate's answer. Call this after every answer before asking the next question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer_summary": {
                    "type": "string",
                    "description": "Brief summary of what the candidate said"
                },
                "topic": {
                    "type": "string",
                    "description": "Topic being tested (e.g., 'clock_tree_synthesis', 'timing_closure')"
                },
                "quality": {
                    "type": "string",
                    "enum": ["strong", "adequate", "weak", "honest_admission"],
                    "description": "Overall quality of the answer"
                },
                "score": {
                    "type": "integer",
                    "description": "Score 1-10"
                },
                "missing_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key points the candidate missed"
                }
            },
            "required": ["answer_summary", "topic", "quality", "score"]
        }
    },
    {
        "type": "function",
        "name": "end_interview",
        "description": "End the interview and generate final scores. Call this after 15-20 questions or when candidate wants to stop.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the interview is ending"
                },
                "questions_asked": {
                    "type": "integer",
                    "description": "Total technical questions asked"
                }
            },
            "required": ["reason", "questions_asked"]
        }
    }
]


def handle_tool_call(session_id, tool_name, arguments):
    """Process tool calls from OpenAI Realtime API."""
    session = sessions.get(session_id, {})

    if tool_name == "evaluate_answer":
        # Store evaluation
        turn = session.get("turn", 0) + 1
        session["turn"] = turn
        evaluations = session.setdefault("evaluations", [])
        evaluations.append({
            "turn": turn,
            "topic": arguments.get("topic", ""),
            "quality": arguments.get("quality", "adequate"),
            "score": arguments.get("score", 5),
            "answer_summary": arguments.get("answer_summary", ""),
            "missing_points": arguments.get("missing_points", []),
            "timestamp": time.time(),
        })

        # Compute running stats
        scores = [e["score"] for e in evaluations]
        avg_score = sum(scores) / len(scores)
        consecutive_weak = 0
        for e in reversed(evaluations):
            if e["quality"] in ("weak", "honest_admission") and e["score"] <= 4:
                consecutive_weak += 1
            else:
                break

        # Determine difficulty adjustment
        difficulty = "intermediate"
        if avg_score >= 8: difficulty = "advanced"
        elif avg_score >= 6: difficulty = "intermediate"
        elif avg_score <= 4: difficulty = "basic"

        result = {
            "status": "recorded",
            "turn": turn,
            "running_avg": round(avg_score, 1),
            "next_difficulty": difficulty,
            "consecutive_weak": consecutive_weak,
            "instruction": ""
        }

        if consecutive_weak >= 3:
            result["instruction"] = "Candidate is struggling. Ask a simpler question or offer encouragement."
        elif avg_score >= 8 and turn >= 5:
            result["instruction"] = "Candidate is strong. Push into advanced edge cases or ask for specific numbers."

        print(f"[Tool] evaluate_answer: Q{turn} topic={arguments.get('topic')} score={arguments.get('score')} quality={arguments.get('quality')} avg={avg_score:.1f}")
        return json.dumps(result)

    elif tool_name == "end_interview":
        evaluations = session.get("evaluations", [])
        scores = [e["score"] for e in evaluations]
        avg = sum(scores) / len(scores) if scores else 5
        session["ended"] = True
        session["end_reason"] = arguments.get("reason", "completed")

        report = {
            "status": "interview_ended",
            "total_questions": len(evaluations),
            "average_score": round(avg, 1),
            "grade": "A" if avg >= 8.5 else "B" if avg >= 7 else "C" if avg >= 5.5 else "D" if avg >= 4 else "F",
            "strong_topics": [e["topic"] for e in evaluations if e["score"] >= 8],
            "weak_topics": [e["topic"] for e in evaluations if e["score"] <= 4],
            "instruction": "Thank the candidate warmly. Mention their strengths and suggest 1-2 areas to improve. Keep it brief and encouraging."
        }
        print(f"[Tool] end_interview: {len(evaluations)} questions, avg={avg:.1f}, grade={report['grade']}")
        return json.dumps(report)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ════════════════════════════════════════════════════════════
# WEBSOCKET — Browser ↔ Server ↔ OpenAI Realtime
# ════════════════════════════════════════════════════════════

@app.websocket("/ws/interview")
async def websocket_interview(ws: WebSocket):
    await ws.accept()

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"turn": 0, "evaluations": [], "started_at": time.time()}

    # Get config from client
    try:
        config_msg = await asyncio.wait_for(ws.receive_text(), timeout=5)
        config = json.loads(config_msg)
    except:
        config = {}

    domain = config.get("domain", "physical_design")
    level = config.get("level", "trained_fresher")
    candidate_name = config.get("candidate_name", "Candidate")

    system_prompt = get_system_prompt(domain, level, candidate_name)

    print(f"[Realtime] Session {session_id[:8]} started — {candidate_name} | {domain} | {level}")

    # Connect to OpenAI Realtime API (model must be in URL)
    openai_ws_url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    try:
        async with ws_connect(openai_ws_url, additional_headers=headers) as openai_ws:
            # Configure the session
            session_config = {
                "type": "session.update",
                "session": {
                    "voice": VOICE,
                    "instructions": system_prompt,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 1500
                    }
                }
            }
            await openai_ws.send(json.dumps(session_config))
            print(f"[Realtime] OpenAI session configured")

            # Send initial greeting trigger
            await openai_ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"]
                }
            }))

            # Relay messages between browser and OpenAI
            async def browser_to_openai():
                """Forward audio from browser to OpenAI."""
                try:
                    while True:
                        data = await ws.receive()
                        if "bytes" in data:
                            # Audio data from browser mic
                            audio_b64 = base64.b64encode(data["bytes"]).decode()
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64
                            }))
                        elif "text" in data:
                            msg = json.loads(data["text"])
                            if msg.get("type") == "end":
                                # Client wants to end
                                await openai_ws.send(json.dumps({
                                    "type": "response.create",
                                    "response": {
                                        "modalities": ["text", "audio"],
                                        "instructions": "The candidate wants to end the interview. Call end_interview tool and say goodbye."
                                    }
                                }))
                except WebSocketDisconnect:
                    pass

            async def openai_to_browser():
                """Forward audio/events from OpenAI to browser."""
                try:
                    async for message in openai_ws:
                        event = json.loads(message)
                        event_type = event.get("type", "")

                        # Audio delta — send to browser for playback
                        if event_type == "response.audio.delta":
                            audio_bytes = base64.b64decode(event["delta"])
                            await ws.send_bytes(audio_bytes)

                        # Text delta — send transcript to browser
                        elif event_type == "response.audio_transcript.delta":
                            await ws.send_text(json.dumps({
                                "type": "transcript",
                                "text": event.get("delta", "")
                            }))

                        # Tool call — process and return result
                        elif event_type == "response.function_call_arguments.done":
                            tool_name = event.get("name", "")
                            call_id = event.get("call_id", "")
                            try:
                                args = json.loads(event.get("arguments", "{}"))
                            except:
                                args = {}

                            result = handle_tool_call(session_id, tool_name, args)

                            # Send tool result back to OpenAI
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result
                                }
                            }))
                            # Trigger next response
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                                "response": {"modalities": ["text", "audio"]}
                            }))

                            # Send evaluation update to browser
                            await ws.send_text(json.dumps({
                                "type": "evaluation",
                                "data": json.loads(result)
                            }))

                        # Speech started — notify browser
                        elif event_type == "input_audio_buffer.speech_started":
                            await ws.send_text(json.dumps({"type": "speech_started"}))

                        # Speech stopped
                        elif event_type == "input_audio_buffer.speech_stopped":
                            await ws.send_text(json.dumps({"type": "speech_stopped"}))

                        # Response done
                        elif event_type == "response.done":
                            await ws.send_text(json.dumps({"type": "response_done"}))

                        # Error
                        elif event_type == "error":
                            print(f"[Realtime] Error: {event.get('error', {})}")
                            await ws.send_text(json.dumps({
                                "type": "error",
                                "message": str(event.get("error", {}).get("message", "Unknown error"))
                            }))

                except websockets.exceptions.ConnectionClosed:
                    pass

            # Run both directions concurrently
            await asyncio.gather(
                browser_to_openai(),
                openai_to_browser(),
                return_exceptions=True
            )

    except Exception as e:
        print(f"[Realtime] Connection error: {e}")
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))

    finally:
        # Session ended
        session = sessions.get(session_id, {})
        evaluations = session.get("evaluations", [])
        print(f"[Realtime] Session {session_id[:8]} ended — {len(evaluations)} questions evaluated")
        await ws.close()


# ════════════════════════════════════════════════════════════
# API — Get session results
# ════════════════════════════════════════════════════════════

@app.get("/api/realtime/session/{session_id}")
async def get_realtime_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    return session


# ════════════════════════════════════════════════════════════
# FRONTEND
# ════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VLSI Interview — Realtime Voice</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.shell { width: 500px; padding: 32px; }
h1 { font-size: 22px; margin-bottom: 4px; color: #fff; }
p.sub { font-size: 12px; color: #555; margin-bottom: 24px; }
.card { background: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.card h2 { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.orb { width: 80px; height: 80px; border-radius: 50%; border: 2px solid #333; margin: 0 auto 16px; display: flex; align-items: center; justify-content: center; transition: all 0.3s; }
.orb.idle { border-color: #333; }
.orb.listening { border-color: #4ade80; box-shadow: 0 0 20px rgba(74,222,128,0.2); }
.orb.speaking { border-color: #60a5fa; box-shadow: 0 0 20px rgba(96,165,250,0.2); }
.orb-label { text-align: center; font-size: 12px; color: #555; margin-bottom: 16px; }
.transcript { font-size: 13px; color: #aaa; min-height: 40px; line-height: 1.6; font-style: italic; }
.stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.stat { text-align: center; padding: 8px; background: #111; border-radius: 8px; }
.stat-val { font-size: 20px; font-weight: 700; font-family: monospace; }
.stat-label { font-size: 9px; color: #555; text-transform: uppercase; }
.btn { width: 100%; padding: 14px; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.btn-start { background: #16a34a; color: #fff; }
.btn-start:hover { background: #15803d; }
.btn-end { background: #dc2626; color: #fff; margin-top: 8px; }
.btn-end:hover { background: #b91c1c; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
select, input { width: 100%; padding: 8px 12px; background: #111; border: 1px solid #333; border-radius: 6px; color: #fff; font-size: 13px; margin-bottom: 8px; }
label { font-size: 11px; color: #666; margin-bottom: 4px; display: block; }
.log { font-family: monospace; font-size: 10px; color: #555; max-height: 120px; overflow-y: auto; margin-top: 12px; background: #0d0d0d; border-radius: 6px; padding: 8px; }
.log div { padding: 1px 0; }
.log .eval { color: #4ade80; }
.log .err { color: #f87171; }
</style>
</head>
<body>
<div class="shell">
    <h1>VLSI Interview — Realtime Voice</h1>
    <p class="sub">OpenAI Realtime API — sub-second latency voice interview</p>

    <div class="card">
        <h2>Setup</h2>
        <label>Domain</label>
        <select id="domain">
            <option value="physical_design">Physical Design</option>
            <option value="analog_layout">Analog Layout</option>
            <option value="design_verification">Design Verification</option>
        </select>
        <label>Level</label>
        <select id="level">
            <option value="trained_fresher">Trained Fresher</option>
            <option value="fresh_graduate">Fresh Graduate</option>
            <option value="experienced_junior">Experienced Junior</option>
            <option value="experienced_senior">Experienced Senior</option>
        </select>
        <label>Your Name</label>
        <input type="text" id="name" value="Candidate" placeholder="Enter your name">
    </div>

    <div class="card" style="text-align:center">
        <div class="orb idle" id="orb">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a4 4 0 0 1 4 4v4a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/>
                <path d="M6 12a6 6 0 0 0 12 0"/>
                <line x1="12" y1="18" x2="12" y2="22"/>
            </svg>
        </div>
        <div class="orb-label" id="status">Press Start to begin</div>
        <div class="transcript" id="transcript">AI responses will appear here...</div>
    </div>

    <div class="card">
        <h2>Live Score</h2>
        <div class="stats">
            <div class="stat"><div class="stat-val" id="s-q">0</div><div class="stat-label">Questions</div></div>
            <div class="stat"><div class="stat-val" id="s-avg">—</div><div class="stat-label">Avg Score</div></div>
            <div class="stat"><div class="stat-val" id="s-diff">—</div><div class="stat-label">Difficulty</div></div>
        </div>
    </div>

    <button class="btn btn-start" id="btn-start" onclick="startInterview()">Start Interview</button>
    <button class="btn btn-end" id="btn-end" onclick="endInterview()" disabled>End Interview</button>

    <div class="log" id="log"></div>
</div>

<script>
let ws = null;
let audioCtx = null;
let mediaStream = null;
let scriptNode = null;
let source = null;

function log(msg, cls='') {
    const el = document.getElementById('log');
    const d = document.createElement('div');
    d.className = cls;
    d.textContent = new Date().toLocaleTimeString() + ' ' + msg;
    el.prepend(d);
    if (el.children.length > 30) el.lastChild.remove();
}

function setOrb(state) {
    document.getElementById('orb').className = 'orb ' + state;
}

async function startInterview() {
    const domain = document.getElementById('domain').value;
    const level = document.getElementById('level').value;
    const name = document.getElementById('name').value || 'Candidate';

    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-end').disabled = false;
    document.getElementById('status').textContent = 'Connecting...';
    log('Starting realtime interview...');

    // Get mic access
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 24000, echoCancellation: true, noiseSuppression: true }
        });
    } catch(e) {
        log('Microphone access denied: ' + e.message, 'err');
        document.getElementById('btn-start').disabled = false;
        return;
    }

    // Connect WebSocket
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/interview`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        log('Connected to server');
        ws.send(JSON.stringify({ domain, level, candidate_name: name }));
        startAudioStream();
        setOrb('speaking');
        document.getElementById('status').textContent = 'AI is speaking...';
    };

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            // Audio from AI — play it
            playAudio(event.data);
        } else {
            const msg = JSON.parse(event.data);
            if (msg.type === 'transcript') {
                document.getElementById('transcript').textContent += msg.text;
            } else if (msg.type === 'speech_started') {
                setOrb('listening');
                document.getElementById('status').textContent = 'Listening...';
                document.getElementById('transcript').textContent = '';
            } else if (msg.type === 'speech_stopped') {
                setOrb('speaking');
                document.getElementById('status').textContent = 'AI thinking...';
            } else if (msg.type === 'response_done') {
                setOrb('listening');
                document.getElementById('status').textContent = 'Your turn — speak now';
            } else if (msg.type === 'evaluation') {
                const d = msg.data;
                if (d.turn) document.getElementById('s-q').textContent = d.turn;
                if (d.running_avg) document.getElementById('s-avg').textContent = d.running_avg;
                if (d.next_difficulty) document.getElementById('s-diff').textContent = d.next_difficulty;
                log(`Q${d.turn}: score=${d.running_avg} diff=${d.next_difficulty}`, 'eval');
            } else if (msg.type === 'error') {
                log('Error: ' + msg.message, 'err');
            }
        }
    };

    ws.onerror = (e) => { log('WebSocket error', 'err'); };
    ws.onclose = () => {
        log('Connection closed');
        setOrb('idle');
        document.getElementById('status').textContent = 'Interview ended';
        stopAudioStream();
    };
}

function startAudioStream() {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    source = audioCtx.createMediaStreamSource(mediaStream);
    scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);

    scriptNode.onaudioprocess = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        // Convert float32 to int16 PCM
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            pcm16[i] = Math.max(-32768, Math.min(32767, Math.floor(input[i] * 32768)));
        }
        ws.send(pcm16.buffer);
    };

    source.connect(scriptNode);
    scriptNode.connect(audioCtx.destination);
    log('Audio streaming started (24kHz PCM16)');
}

function stopAudioStream() {
    if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
    if (source) { source.disconnect(); source = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
}

// Play PCM16 audio from OpenAI
let playQueue = [];
let isPlaying = false;

function playAudio(buffer) {
    playQueue.push(buffer);
    if (!isPlaying) drainQueue();
}

async function drainQueue() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    isPlaying = true;
    while (playQueue.length > 0) {
        const buf = playQueue.shift();
        const pcm16 = new Int16Array(buf);
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) {
            float32[i] = pcm16[i] / 32768;
        }
        const audioBuffer = audioCtx.createBuffer(1, float32.length, 24000);
        audioBuffer.getChannelData(0).set(float32);
        const src = audioCtx.createBufferSource();
        src.buffer = audioBuffer;
        src.connect(audioCtx.destination);
        src.start();
        await new Promise(r => { src.onended = r; });
    }
    isPlaying = false;
}

function endInterview() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'end' }));
        log('Ending interview...');
    }
    document.getElementById('btn-end').disabled = true;
}
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("VLSI Interview — OpenAI Realtime Voice")
    print("=" * 50)
    print(f"  Model:  {REALTIME_MODEL}")
    print(f"  Voice:  {VOICE}")
    print(f"  API Key: {'set' if OPENAI_API_KEY else 'MISSING'}")
    print(f"  Open:   http://localhost:8002")
    print("=" * 50)
    uvicorn.run("realtime_interview:app", host="0.0.0.0", port=8002, reload=False)
