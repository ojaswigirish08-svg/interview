"""
Sample Interview UI — ElevenLabs Scribe v2 Realtime STT (Live words as you speak)
"""

import os
import json
import time
import asyncio
import base64
from flask import Flask, render_template_string, request, jsonify
import websockets

app = Flask(__name__)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_KEY_HERE")
INTERVIEW_FILE = "interview_data_elevenlabs.json"

QUESTIONS = [
    "Tell me about yourself and your experience in VLSI.",
    "What is the difference between setup time and hold time?",
    "What happens when setup time is violated?",
    "Explain clock tree synthesis and why it is important.",
    "What is metastability and how do you handle it?",
    "What is the difference between blocking and non-blocking assignments in Verilog?",
    "Explain IR drop and how it affects timing closure.",
    "What is OCV and how do you handle it in STA?",
    "What are the challenges in multi-voltage design?",
    "How do you debug a timing violation that only appears in silicon?",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VLSI Interview - ElevenLabs Realtime</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a1a; color: #eee; min-height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { width: 750px; background: #12122a; border-radius: 20px; padding: 40px; box-shadow: 0 8px 40px rgba(100,60,255,0.15); }
        h1 { text-align: center; color: #a78bfa; margin-bottom: 8px; font-size: 26px; }
        .badge { text-align: center; margin-bottom: 25px; }
        .badge span { background: #7c3aed; color: white; padding: 5px 14px; border-radius: 14px; font-size: 12px; letter-spacing: 0.5px; }
        .progress { text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }
        .question-box { background: #1a1a3e; padding: 22px; border-radius: 14px; margin-bottom: 25px; border-left: 4px solid #7c3aed; }
        .question-box p { font-size: 18px; line-height: 1.7; }
        .label { font-size: 11px; color: #a78bfa; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1.5px; font-weight: 600; }
        .controls { display: flex; gap: 15px; justify-content: center; margin-bottom: 25px; }
        button { padding: 14px 32px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; transition: all 0.3s; font-weight: 600; }
        .record-btn { background: #7c3aed; color: white; }
        .record-btn:hover { background: #6d28d9; transform: scale(1.03); }
        .record-btn.recording { background: #ef4444; animation: pulse 1s infinite; }
        .stop-btn { background: #4c1d95; color: white; }
        .stop-btn:hover { background: #3b0764; }
        .next-btn { background: transparent; color: #a78bfa; border: 2px solid #7c3aed; }
        .next-btn:hover { background: #7c3aed; color: white; }
        .submit-btn { background: #10b981; color: white; }
        .submit-btn:hover { background: #059669; }
        button:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
        .transcript-box { background: #0a0a1a; padding: 18px; border-radius: 10px; min-height: 80px; margin-bottom: 10px; border: 1px solid #2a2a4a; }
        .transcript-box p { color: #999; font-style: italic; font-size: 16px; line-height: 1.6; }
        .transcript-box p.final { color: #e2e8f0; font-style: normal; }
        .live-box { background: #0a0a1a; padding: 12px 18px; border-radius: 10px; min-height: 40px; margin-bottom: 20px; border: 1px dashed #7c3aed; }
        .live-box p { color: #7c3aed; font-size: 14px; min-height: 20px; }
        .status { text-align: center; color: #a78bfa; font-size: 13px; margin-top: 15px; min-height: 20px; }
        .timer { text-align: center; font-size: 28px; color: #ef4444; margin-bottom: 15px; font-family: 'Courier New', monospace; font-weight: bold; }
        .stats { display: flex; justify-content: center; gap: 30px; margin-top: 15px; font-size: 12px; color: #666; }
        .stats span { background: #1a1a3e; padding: 4px 12px; border-radius: 8px; }
        @keyframes pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0.4); } 50% { opacity: 0.8; box-shadow: 0 0 20px 5px rgba(239,68,68,0.2); } }
        .done { text-align: center; padding: 50px; }
        .done h2 { color: #10b981; margin-bottom: 15px; font-size: 28px; }
        .waveform { display: flex; align-items: center; justify-content: center; gap: 3px; height: 30px; margin-bottom: 15px; }
        .waveform .bar { width: 4px; background: #7c3aed; border-radius: 2px; transition: height 0.1s; }
        .waveform.off .bar { height: 4px !important; }
    </style>
</head>
<body>
    <div class="container">
        <h1>VLSI Mock Interview</h1>
        <div class="badge"><span>ElevenLabs Scribe v2 Realtime — 150ms Latency</span></div>
        <div class="progress" id="progress">Question 1 of {{ total }}</div>

        <div class="question-box">
            <div class="label">Interviewer Question</div>
            <p id="question">{{ first_question }}</p>
        </div>

        <div class="waveform off" id="waveform">
            <div class="bar" style="height:4px"></div><div class="bar" style="height:4px"></div>
            <div class="bar" style="height:4px"></div><div class="bar" style="height:4px"></div>
            <div class="bar" style="height:4px"></div><div class="bar" style="height:4px"></div>
            <div class="bar" style="height:4px"></div><div class="bar" style="height:4px"></div>
            <div class="bar" style="height:4px"></div><div class="bar" style="height:4px"></div>
            <div class="bar" style="height:4px"></div><div class="bar" style="height:4px"></div>
        </div>

        <div class="timer" id="timer">00:00</div>

        <div class="controls">
            <button class="record-btn" id="recordBtn" onclick="startRecording()">🎙 Record</button>
            <button class="stop-btn" id="stopBtn" onclick="stopRecording()" disabled>⏹ Stop</button>
            <button class="next-btn" id="nextBtn" onclick="nextQuestion()" disabled>Next →</button>
        </div>

        <div class="live-box">
            <div class="label">Live Transcription</div>
            <p id="liveText">...</p>
        </div>

        <div class="transcript-box">
            <div class="label">Your Answer (Final)</div>
            <p id="transcript" class="final">Click Record and speak your answer...</p>
        </div>

        <div class="status" id="status"></div>
        <div class="stats" id="stats"></div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let currentQuestion = 0;
        let questions = {{ questions | tojson }};
        let timerInterval;
        let seconds = 0;
        let wsConnection = null;
        let audioContext, analyser, micSource, animFrame;
        let fullTranscript = '';
        let startTime = 0;

        const API_KEY = '{{ api_key }}';
        const WS_URL = 'wss://api.elevenlabs.io/v1/speech-to-text/realtime'
            + '?model_id=scribe_v2_realtime'
            + '&audio_format=pcm_16000'
            + '&commit_strategy=vad'
            + '&include_timestamps=true'
            + '&language_code=en';

        function updateTimer() {
            seconds++;
            let m = String(Math.floor(seconds / 60)).padStart(2, '0');
            let s = String(seconds % 60).padStart(2, '0');
            document.getElementById('timer').textContent = m + ':' + s;
        }

        function animateWaveform(stream) {
            audioContext = new AudioContext();
            micSource = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 64;
            micSource.connect(analyser);
            const bars = document.querySelectorAll('.waveform .bar');
            const data = new Uint8Array(analyser.frequencyBinCount);
            document.getElementById('waveform').classList.remove('off');

            function draw() {
                analyser.getByteFrequencyData(data);
                bars.forEach((bar, i) => {
                    const val = data[i * 2] || 0;
                    bar.style.height = Math.max(4, val / 8) + 'px';
                });
                animFrame = requestAnimationFrame(draw);
            }
            draw();
        }

        function stopWaveform() {
            if (animFrame) cancelAnimationFrame(animFrame);
            if (audioContext) audioContext.close();
            document.getElementById('waveform').classList.add('off');
        }

        async function startRecording() {
            fullTranscript = '';
            seconds = 0;
            startTime = Date.now();
            document.getElementById('timer').textContent = '00:00';
            document.getElementById('liveText').textContent = '...';
            document.getElementById('transcript').textContent = 'Listening...';

            // Get microphone
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
            });

            animateWaveform(stream);

            // Connect to ElevenLabs WebSocket
            wsConnection = new WebSocket(WS_URL + '&xi-api-key=' + API_KEY);

            wsConnection.onopen = () => {
                document.getElementById('status').textContent = 'Connected — speak now!';
            };

            wsConnection.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                const type = msg.message_type;

                if (type === 'partial_transcript' && msg.text) {
                    document.getElementById('liveText').textContent = msg.text;
                } else if (type === 'committed_transcript' && msg.text) {
                    fullTranscript += (fullTranscript ? ' ' : '') + msg.text;
                    document.getElementById('transcript').textContent = fullTranscript;
                    document.getElementById('liveText').textContent = '...';
                }
            };

            wsConnection.onerror = (e) => {
                document.getElementById('status').textContent = 'WebSocket error — check API key';
                console.error('WS error:', e);
            };

            // Record raw PCM and send chunks
            const ac = new AudioContext({ sampleRate: 16000 });
            const source = ac.createMediaStreamSource(stream);
            const processor = ac.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
                    const float32 = e.inputBuffer.getChannelData(0);
                    const int16 = new Int16Array(float32.length);
                    for (let i = 0; i < float32.length; i++) {
                        int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
                    }
                    const b64 = btoa(String.fromCharCode(...new Uint8Array(int16.buffer)));
                    wsConnection.send(JSON.stringify({
                        message_type: 'input_audio_chunk',
                        audio_base_64: b64,
                        sample_rate: 16000,
                    }));
                }
            };

            source.connect(processor);
            processor.connect(ac.destination);

            // Store for cleanup
            window._stream = stream;
            window._ac2 = ac;
            window._processor = processor;
            window._source = source;

            timerInterval = setInterval(updateTimer, 1000);

            document.getElementById('recordBtn').classList.add('recording');
            document.getElementById('recordBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('nextBtn').disabled = true;
            document.getElementById('status').textContent = 'Recording — words appear live...';
        }

        function stopRecording() {
            clearInterval(timerInterval);
            stopWaveform();

            // Cleanup audio
            if (window._processor) window._processor.disconnect();
            if (window._source) window._source.disconnect();
            if (window._ac2) window._ac2.close();
            if (window._stream) window._stream.getTracks().forEach(t => t.stop());

            // Close WebSocket
            if (wsConnection) {
                wsConnection.close();
                wsConnection = null;
            }

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            document.getElementById('recordBtn').classList.remove('recording');
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('liveText').textContent = '(stopped)';

            if (fullTranscript.trim()) {
                document.getElementById('nextBtn').disabled = false;
                document.getElementById('status').textContent = 'Done!';
                document.getElementById('stats').innerHTML =
                    `<span>Duration: ${elapsed}s</span>` +
                    `<span>Words: ${fullTranscript.split(/\\s+/).length}</span>` +
                    `<span>Latency: ~150ms</span>`;

                // Save to server
                fetch('/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question_index: currentQuestion,
                        answer: fullTranscript,
                        duration: parseFloat(elapsed),
                    })
                });
            } else {
                document.getElementById('status').textContent = 'No speech detected. Try again.';
                document.getElementById('recordBtn').disabled = false;
            }
        }

        function nextQuestion() {
            currentQuestion++;
            if (currentQuestion >= questions.length) {
                document.querySelector('.container').innerHTML = `
                    <div class="done">
                        <h2>Interview Complete!</h2>
                        <p>All answers saved to interview_data_elevenlabs.json</p>
                        <br>
                        <button class="submit-btn" onclick="window.location.reload()">Restart</button>
                    </div>`;
                return;
            }
            document.getElementById('question').textContent = questions[currentQuestion];
            document.getElementById('progress').textContent = 'Question ' + (currentQuestion + 1) + ' of ' + questions.length;
            document.getElementById('transcript').textContent = 'Click Record and speak your answer...';
            document.getElementById('liveText').textContent = '...';
            document.getElementById('timer').textContent = '00:00';
            document.getElementById('status').textContent = '';
            document.getElementById('stats').innerHTML = '';
            document.getElementById('recordBtn').disabled = false;
            document.getElementById('nextBtn').disabled = true;
            seconds = 0;
        }
    </script>
</body>
</html>
"""


def load_data():
    if os.path.exists(INTERVIEW_FILE):
        with open(INTERVIEW_FILE, "r") as f:
            return json.load(f)
    return {"interviews": []}


def save_data(data):
    with open(INTERVIEW_FILE, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        first_question=QUESTIONS[0],
        total=len(QUESTIONS),
        questions=QUESTIONS,
        api_key=ELEVENLABS_API_KEY,
    )


@app.route("/save", methods=["POST"])
def save():
    body = request.json
    idx = body.get("question_index", 0)
    answer = body.get("answer", "")
    duration = body.get("duration", 0)

    data = load_data()
    data["interviews"].append({
        "question_number": idx + 1,
        "question": QUESTIONS[idx],
        "answer": answer,
        "duration": duration,
        "stt_engine": "elevenlabs_scribe_v2_realtime",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_data(data)

    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n=== VLSI Interview UI (ElevenLabs Realtime) ===")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"STT: ElevenLabs Scribe v2 Realtime (150ms latency)")
    print(f"Save to: {INTERVIEW_FILE}")
    print(f"\nOpen: http://localhost:5000\n")
    app.run(debug=True, port=5000)
