"""
Sample Interview UI — OpenAI Whisper STT (Record → Stop → Transcribe)
"""

import os
import json
import time
import tempfile
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
client = OpenAI(api_key=OPENAI_API_KEY)
INTERVIEW_FILE = "interview_data_openai.json"

QUESTIONS = [
    # Warmup
    "Tell me about yourself and your experience in Design Verification.",
    # UVM Fundamentals
    "Explain the UVM testbench architecture. What are the key components and how do they interact?",
    "What is the difference between uvm_sequence, uvm_sequence_item, and uvm_driver? How do they work together?",
    "What are UVM phases? Explain the build, connect, run, and report phases.",
    # SystemVerilog & Assertions
    "What is the difference between immediate assertions and concurrent assertions in SystemVerilog?",
    "Write an SVA property to check that after a request signal goes high, grant must come within 5 clock cycles.",
    "What is the difference between $rose, $fell, and $stable in SVA? Give a real use case for each.",
    # Coverage
    "What is the difference between code coverage and functional coverage? Why do you need both?",
    "You have 95% code coverage but only 60% functional coverage. What does that tell you and how do you fix it?",
    "How do you write a covergroup with cross coverage? Give an example with two variables.",
    # Constrained Random Verification
    "What is constrained random verification? Why is it better than directed testing?",
    "How do you use constraints in SystemVerilog? What is the difference between soft and hard constraints?",
    "What happens when two constraints conflict? How do you debug constraint solver failures?",
    # Debugging & Methodology
    "Your simulation passes but the checker is not firing. How do you debug this?",
    "What is a scoreboard in UVM? How do you handle out-of-order transactions?",
    "Explain the difference between factory override and type override in UVM. When would you use each?",
    # Advanced Topics
    "What is a virtual sequence and when do you use it? How is it different from a regular sequence?",
    "How do you verify a multi-clock domain design? What are the challenges?",
    "What is formal verification? When would you use it instead of simulation-based verification?",
    "Explain the difference between simulation, emulation, and FPGA prototyping. When do you use each?",
    # Bug Escapes & Real-World
    "A bug was found in silicon that was not caught in verification. How do you root-cause this and prevent it in the future?",
    "What is register verification using RAL in UVM? How do you verify CSR access?",
    "How do you handle verification of low-power designs with UPF/CPF?",
    "What is the difference between pass-by-reference and pass-by-value in SystemVerilog tasks and functions?",
    "What verification closure criteria do you follow before tapeout? How do you know verification is complete?",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VLSI Interview - OpenAI Whisper</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0f1a; color: #eee; min-height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { width: 720px; background: #131b2e; border-radius: 18px; padding: 40px; box-shadow: 0 8px 40px rgba(116,170,156,0.1); }
        h1 { text-align: center; color: #74aa9c; margin-bottom: 8px; font-size: 26px; }
        .badge { text-align: center; margin-bottom: 25px; }
        .badge span { background: #74aa9c; color: #0a0f1a; padding: 5px 14px; border-radius: 14px; font-size: 12px; font-weight: 600; }
        .progress { text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }
        .question-box { background: #1a2340; padding: 22px; border-radius: 14px; margin-bottom: 25px; border-left: 4px solid #74aa9c; }
        .question-box p { font-size: 18px; line-height: 1.7; }
        .label { font-size: 11px; color: #74aa9c; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1.5px; font-weight: 600; }
        .model-select { text-align: center; margin-bottom: 20px; }
        .model-select select { background: #1a2340; color: #eee; border: 1px solid #74aa9c; padding: 8px 15px; border-radius: 8px; font-size: 14px; }
        .controls { display: flex; gap: 15px; justify-content: center; margin-bottom: 25px; }
        button { padding: 14px 32px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; transition: all 0.3s; font-weight: 600; }
        .record-btn { background: #74aa9c; color: #0a0f1a; }
        .record-btn:hover { background: #5d9488; }
        .record-btn.recording { background: #ef4444; color: white; animation: pulse 1s infinite; }
        .stop-btn { background: #2a3a5c; color: white; }
        .stop-btn:hover { background: #1a2a4c; }
        .next-btn { background: transparent; color: #74aa9c; border: 2px solid #74aa9c; }
        .next-btn:hover { background: #74aa9c; color: #0a0f1a; }
        .submit-btn { background: #10b981; color: white; }
        button:disabled { opacity: 0.3; cursor: not-allowed; }
        .transcript-box { background: #0a0f1a; padding: 18px; border-radius: 10px; min-height: 70px; margin-bottom: 15px; border: 1px solid #1a2340; }
        .transcript-box p { color: #e2e8f0; font-size: 16px; line-height: 1.6; }
        .status { text-align: center; color: #74aa9c; font-size: 13px; margin-top: 12px; min-height: 20px; }
        .timer { text-align: center; font-size: 28px; color: #ef4444; margin-bottom: 15px; font-family: 'Courier New', monospace; font-weight: bold; }
        .stats { display: flex; justify-content: center; gap: 20px; margin-top: 12px; font-size: 12px; color: #555; }
        .stats span { background: #1a2340; padding: 4px 12px; border-radius: 8px; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.7; } }
        .done { text-align: center; padding: 50px; }
        .done h2 { color: #10b981; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Design Verification Interview</h1>
        <div class="badge"><span>OpenAI Whisper STT — DV / UVM / SVA</span></div>
        <div class="progress" id="progress">Question 1 of {{ total }}</div>

        <div class="model-select">
            <label class="label">Model: </label>
            <select id="modelSelect">
                <option value="gpt-4o-mini-transcribe" selected>gpt-4o-mini-transcribe ($0.18/hr, fast)</option>
                <option value="gpt-4o-transcribe">gpt-4o-transcribe ($0.36/hr, best)</option>
                <option value="whisper-1">whisper-1 ($0.36/hr, classic)</option>
            </select>
        </div>

        <div class="question-box">
            <div class="label">Interviewer Question</div>
            <p id="question">{{ first_question }}</p>
        </div>

        <div class="timer" id="timer">00:00</div>

        <div class="controls">
            <button class="record-btn" id="recordBtn" onclick="startRecording()">Record</button>
            <button class="stop-btn" id="stopBtn" onclick="stopRecording()" disabled>Stop</button>
            <button class="next-btn" id="nextBtn" onclick="nextQuestion()" disabled>Next</button>
        </div>

        <div class="transcript-box">
            <div class="label">Your Answer (Transcribed)</div>
            <p id="transcript">Click Record and speak your answer...</p>
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

        function updateTimer() {
            seconds++;
            let m = String(Math.floor(seconds / 60)).padStart(2, '0');
            let s = String(seconds % 60).padStart(2, '0');
            document.getElementById('timer').textContent = m + ':' + s;
        }

        async function startRecording() {
            audioChunks = [];
            seconds = 0;
            document.getElementById('timer').textContent = '00:00';

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                clearInterval(timerInterval);
                stream.getTracks().forEach(t => t.stop());
                sendAudio();
            };

            mediaRecorder.start();
            timerInterval = setInterval(updateTimer, 1000);

            document.getElementById('recordBtn').classList.add('recording');
            document.getElementById('recordBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('nextBtn').disabled = true;
            document.getElementById('status').textContent = 'Recording...';
            document.getElementById('transcript').textContent = 'Listening...';
            document.getElementById('stats').innerHTML = '';
        }

        function stopRecording() {
            mediaRecorder.stop();
            document.getElementById('recordBtn').classList.remove('recording');
            document.getElementById('stopBtn').disabled = true;
            const model = document.getElementById('modelSelect').value;
            document.getElementById('status').textContent = 'Transcribing with ' + model + '...';
        }

        async function sendAudio() {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio', blob, 'answer.webm');
            formData.append('question_index', currentQuestion);
            formData.append('model', document.getElementById('modelSelect').value);

            try {
                const res = await fetch('/transcribe', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.error) {
                    document.getElementById('transcript').textContent = 'Error: ' + data.error;
                    document.getElementById('status').textContent = 'Failed. Try again.';
                    document.getElementById('recordBtn').disabled = false;
                } else {
                    document.getElementById('transcript').textContent = data.text;
                    document.getElementById('status').textContent = 'Done!';
                    document.getElementById('stats').innerHTML =
                        '<span>Model: ' + data.model + '</span>' +
                        '<span>Time: ' + data.time + 's</span>' +
                        '<span>Words: ' + data.text.split(/\\s+/).length + '</span>';
                    document.getElementById('nextBtn').disabled = false;
                }
            } catch (err) {
                document.getElementById('transcript').textContent = 'Network error';
                document.getElementById('status').textContent = 'Failed.';
                document.getElementById('recordBtn').disabled = false;
            }
        }

        function nextQuestion() {
            currentQuestion++;
            if (currentQuestion >= questions.length) {
                document.querySelector('.container').innerHTML = `
                    <div class="done">
                        <h2>Interview Complete!</h2>
                        <p>All answers saved to interview_data_openai.json</p>
                        <br>
                        <button class="submit-btn" onclick="window.location.reload()">Restart</button>
                    </div>`;
                return;
            }
            document.getElementById('question').textContent = questions[currentQuestion];
            document.getElementById('progress').textContent = 'Question ' + (currentQuestion + 1) + ' of ' + questions.length;
            document.getElementById('transcript').textContent = 'Click Record and speak your answer...';
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
    )


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio")
    question_index = int(request.form.get("question_index", 0))
    model = request.form.get("model", "gpt-4o-mini-transcribe")

    if not audio_file:
        return jsonify({"error": "No audio received"}), 400

    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    audio_file.save(tmp.name)
    tmp.close()

    start = time.time()
    try:
        with open(tmp.name, "rb") as f:
            result = client.audio.transcriptions.create(
                file=("audio.webm", f),
                model=model,
                language="en",
            )

        elapsed = round(time.time() - start, 2)
        transcript = result.text

        # Save to JSON
        data = load_data()
        data["interviews"].append({
            "question_number": question_index + 1,
            "question": QUESTIONS[question_index],
            "answer": transcript,
            "model": model,
            "transcription_time": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_data(data)

        return jsonify({"text": transcript, "time": elapsed, "model": model})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    print("\n=== Design Verification Interview UI (OpenAI Whisper) ===")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Models: gpt-4o-mini-transcribe, gpt-4o-transcribe, whisper-1")
    print(f"Save to: {INTERVIEW_FILE}")
    print(f"\nOpen: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
