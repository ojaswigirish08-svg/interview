"""
Sample Interview UI — Voice Recording + Groq Whisper STT (FREE) + JSON Save
"""

import os
import json
import time
from flask import Flask, render_template_string, request, jsonify
from groq import Groq

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
groq_client = Groq(api_key=GROQ_API_KEY)
INTERVIEW_FILE = "interview_data_groq.json"

# Design Verification interview questions
QUESTIONS = [
    "Tell me about yourself and your experience in Design Verification.",
    "Explain the UVM testbench architecture. What are the key components and how do they interact?",
    "What is the difference between uvm_sequence, uvm_sequence_item, and uvm_driver? How do they work together?",
    "What are UVM phases? Explain the build, connect, run, and report phases.",
    "What is the difference between immediate assertions and concurrent assertions in SystemVerilog?",
    "Write an SVA property to check that after a request signal goes high, grant must come within 5 clock cycles.",
    "What is the difference between $rose, $fell, and $stable in SVA? Give a real use case for each.",
    "What is the difference between code coverage and functional coverage? Why do you need both?",
    "You have 95% code coverage but only 60% functional coverage. What does that tell you and how do you fix it?",
    "How do you write a covergroup with cross coverage? Give an example with two variables.",
    "What is constrained random verification? Why is it better than directed testing?",
    "How do you use constraints in SystemVerilog? What is the difference between soft and hard constraints?",
    "What happens when two constraints conflict? How do you debug constraint solver failures?",
    "Your simulation passes but the checker is not firing. How do you debug this?",
    "What is a scoreboard in UVM? How do you handle out-of-order transactions?",
    "Explain the difference between factory override and type override in UVM. When would you use each?",
    "What is a virtual sequence and when do you use it? How is it different from a regular sequence?",
    "How do you verify a multi-clock domain design? What are the challenges?",
    "What is formal verification? When would you use it instead of simulation-based verification?",
    "Explain the difference between simulation, emulation, and FPGA prototyping. When do you use each?",
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
    <title>VLSI Interview - Groq Whisper</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #0d1117; color: #eee; min-height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { width: 700px; background: #161b22; border-radius: 16px; padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
        h1 { text-align: center; color: #58a6ff; margin-bottom: 10px; font-size: 24px; }
        .badge { text-align: center; margin-bottom: 25px; }
        .badge span { background: #238636; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; }
        .progress { text-align: center; color: #8b949e; margin-bottom: 20px; font-size: 14px; }
        .question-box { background: #21262d; padding: 20px; border-radius: 12px; margin-bottom: 25px; border-left: 4px solid #58a6ff; }
        .question-box p { font-size: 18px; line-height: 1.6; }
        .label { font-size: 12px; color: #58a6ff; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1px; }
        .controls { display: flex; gap: 15px; justify-content: center; margin-bottom: 25px; }
        button { padding: 12px 30px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; transition: all 0.3s; }
        .record-btn { background: #da3633; color: white; }
        .record-btn:hover { background: #b62324; }
        .record-btn.recording { background: #ff0000; animation: pulse 1s infinite; }
        .stop-btn { background: #6e40c9; color: white; }
        .stop-btn:hover { background: #553098; }
        .next-btn { background: #21262d; color: white; border: 1px solid #58a6ff; }
        .next-btn:hover { background: #58a6ff; color: #0d1117; }
        .submit-btn { background: #238636; color: white; }
        .submit-btn:hover { background: #2ea043; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .transcript-box { background: #0d1117; padding: 15px; border-radius: 8px; min-height: 60px; margin-bottom: 20px; border: 1px solid #30363d; }
        .transcript-box p { color: #8b949e; font-style: italic; }
        .status { text-align: center; color: #58a6ff; font-size: 14px; margin-top: 15px; min-height: 20px; }
        .timer { text-align: center; font-size: 24px; color: #da3633; margin-bottom: 15px; font-family: monospace; }
        .model-select { text-align: center; margin-bottom: 20px; }
        .model-select select { background: #21262d; color: #eee; border: 1px solid #30363d; padding: 8px 15px; border-radius: 8px; font-size: 14px; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .done { text-align: center; padding: 40px; }
        .done h2 { color: #238636; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Design Verification Interview</h1>
        <div class="badge"><span>Groq Whisper — DV / UVM / SVA</span></div>
        <div class="progress" id="progress">Question 1 of {{ total }}</div>

        <div class="model-select">
            <label class="label">Whisper Model: </label>
            <select id="modelSelect">
                <option value="whisper-large-v3-turbo" selected>Whisper Large v3 Turbo (fastest)</option>
                <option value="whisper-large-v3">Whisper Large v3 (most accurate)</option>
            </select>
        </div>

        <div class="question-box">
            <div class="label">Interviewer Question</div>
            <p id="question">{{ first_question }}</p>
        </div>

        <div class="timer" id="timer">00:00</div>

        <div class="controls">
            <button class="record-btn" id="recordBtn" onclick="startRecording()">🎙 Record</button>
            <button class="stop-btn" id="stopBtn" onclick="stopRecording()" disabled>⏹ Stop</button>
            <button class="next-btn" id="nextBtn" onclick="nextQuestion()" disabled>Next →</button>
        </div>

        <div class="transcript-box">
            <div class="label">Your Answer (Transcribed)</div>
            <p id="transcript">Click Record and speak your answer...</p>
        </div>

        <div class="status" id="status"></div>
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
        }

        function stopRecording() {
            mediaRecorder.stop();
            document.getElementById('recordBtn').classList.remove('recording');
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').textContent = 'Transcribing with Groq Whisper...';
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
                    document.getElementById('status').textContent =
                        'Transcribed in ' + data.time + 's | Model: ' + data.model + ' ✓';
                    document.getElementById('nextBtn').disabled = false;
                }
            } catch (err) {
                document.getElementById('transcript').textContent = 'Network error';
                document.getElementById('status').textContent = 'Failed. Try again.';
                document.getElementById('recordBtn').disabled = false;
            }
        }

        function nextQuestion() {
            currentQuestion++;
            if (currentQuestion >= questions.length) {
                document.querySelector('.container').innerHTML = `
                    <div class="done">
                        <h2>Interview Complete!</h2>
                        <p>All answers saved to interview_data_groq.json</p>
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
            document.getElementById('recordBtn').disabled = false;
            document.getElementById('nextBtn').disabled = true;
            seconds = 0;
        }
    </script>
</body>
</html>
"""


def load_interview_data():
    if os.path.exists(INTERVIEW_FILE):
        with open(INTERVIEW_FILE, "r") as f:
            return json.load(f)
    return {"interviews": []}


def save_interview_data(data):
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
    model = request.form.get("model", "whisper-large-v3-turbo")

    if not audio_file:
        return jsonify({"error": "No audio received"}), 400

    # Save temp file
    temp_path = "temp_audio.webm"
    audio_file.save(temp_path)

    start = time.time()
    try:
        with open(temp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.webm", f.read()),
                model=model,
                language="en",
                response_format="verbose_json",
            )

        elapsed = round(time.time() - start, 2)
        transcript = transcription.text

        # Save to JSON
        data = load_interview_data()
        entry = {
            "question_number": question_index + 1,
            "question": QUESTIONS[question_index],
            "answer": transcript,
            "model": model,
            "duration": getattr(transcription, "duration", 0),
            "transcription_time": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        data["interviews"].append(entry)
        save_interview_data(data)

        return jsonify({"text": transcript, "time": elapsed, "model": model})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("\n=== VLSI Interview UI (Groq Whisper) ===")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"STT: Groq Whisper (FREE)")
    print(f"Models: whisper-large-v3-turbo, whisper-large-v3")
    print(f"Save to: {INTERVIEW_FILE}")
    print(f"\nOpen: http://localhost:5000\n")
    app.run(debug=True, port=5000)
