# Speech-to-Text (STT) Model Evaluation Report

---

## Summary Table

| Model | Accuracy | Latency | VLSI Domain | Fast Speech | Noise Handling | Verdict |
|---|---|---|---|---|---|---|
| **gpt-4o-mini-transcribe** | Very Good | 2-4s | Good | Good | Good | Best overall |
| **ElevenLabs Scribe v2** | Excellent | High | Good | Good | Good | Best accuracy, slow |
| **OpenAI Whisper (whisper-1)** | Good | High | Good | Moderate | Moderate | Good accuracy, more latency |
| **xAI Grok STT** | Good | Fast | Good | Fails (fast/low pitch) | Moderate | Good for clear speech only |
| **Groq Whisper Turbo** | Good | Very Fast | Weak (critical words) | Good | Moderate | Fast but misses VLSI terms |
| **Deepgram Nova-2** | Good | High | Moderate | Moderate | Moderate | Accurate but high latency, word mismatch |
| **ElevenLabs Realtime** | Poor | 150ms | Poor | Poor | Very Poor | Generates non-related words on noise |

---

## Detailed Evaluation

### 1. OpenAI gpt-4o-mini-transcribe
- **Accuracy:** Very Good
- **Latency:** 2-4 seconds
- **Strengths:**
  - Accurate transcription for VLSI domain terms
  - Fast response time (2-4 seconds)
  - Good noise handling
  - Handles both clear and fast speech well
- **Weaknesses:**
  - Not real-time (record, stop, transcribe)
- **Best For:** Production use — best balance of accuracy and speed
- **Rating:** 9/10

### 2. ElevenLabs Scribe v2 (Batch)
- **Accuracy:** Excellent (2.3% WER benchmark)
- **Latency:** High (slower than gpt-4o-mini)
- **Strengths:**
  - Most accurate transcription overall
  - Handles VLSI terminology well
  - Good at complex sentences
  - Speaker diarization, word timestamps
- **Weaknesses:**
  - Higher latency than competitors
- **Best For:** Final transcript accuracy — use for saved reports
- **Rating:** 8.5/10

### 3. OpenAI Whisper (whisper-1)
- **Accuracy:** Good
- **Latency:** High (more than gpt-4o-mini)
- **Strengths:**
  - Reliable, well-tested model
  - Decent accuracy for general speech
- **Weaknesses:**
  - More latency than gpt-4o-mini-transcribe
- **Best For:** Legacy support — gpt-4o-mini-transcribe is better in every way
- **Rating:** 7/10

### 4. xAI Grok STT
- **Accuracy:** Good (for clear speech)
- **Latency:** Fast
- **Strengths:**
  - Fast transcription
  - Good accuracy when user speaks clearly
- **Weaknesses:**
  - FAILS when user talks fast
  - FAILS with low pitch voices
  - Unreliable for varied speaking styles
- **Best For:** Clear, slow speech only — NOT suitable for real interviews
- **Rating:** 6/10

### 5. Groq Whisper Turbo
- **Accuracy:** Good (general), Weak (VLSI domain)
- **Latency:** Very Fast (228x real-time)
- **Strengths:**
  - Fastest transcription available
  - Good for general English
- **Weaknesses:**
  - FAILS at recognizing critical VLSI words (UVM, SVA, CTS, OCV, etc.)
  - Misses complex domain-specific terminology
  - Not suitable for technical interviews without post-processing
- **Best For:** Speed-critical, non-technical use cases
- **Rating:** 6.5/10

### 6. Deepgram Nova-2
- **Accuracy:** Good
- **Latency:** High
- **Strengths:**
  - Good overall accuracy
  - Supports real-time streaming
- **Weaknesses:**
  - High latency
  - Word mismatch issues (wrong words substituted)
- **Best For:** Streaming use cases where some word errors are acceptable
- **Rating:** 6.5/10

### 7. ElevenLabs Scribe v2 Realtime
- **Accuracy:** Poor (with any background noise)
- **Latency:** 150ms (fastest)
- **Strengths:**
  - Ultra-low latency (150ms)
  - Words appear as you speak
- **Weaknesses:**
  - GENERATES NON-RELATED WORDS when there is even slight noise
  - Very sensitive to background noise
  - Unreliable for real-world use
- **Best For:** Quiet studio environments only — NOT suitable for interviews
- **Rating:** 4/10

---

## VLSI Domain-Specific Accuracy

Common VLSI terms tested: UVM, SVA, CTS, OCV, STA, DRC, LVS, PVT, IR drop, metastability, floorplan, clock tree synthesis, setup time, hold time, scoreboard, covergroup

| Model | VLSI Term Recognition |
|---|---|
| **gpt-4o-mini-transcribe** | Good — recognizes most terms correctly |
| **ElevenLabs Scribe v2** | Very Good — best at technical terms |
| **OpenAI Whisper** | Good — recognizes most terms |
| **xAI Grok STT** | Good (clear speech) — fails with fast delivery |
| **Groq Whisper Turbo** | Weak — frequently misses critical VLSI terms |
| **Deepgram Nova-2** | Moderate — substitutes similar-sounding words |
| **ElevenLabs Realtime** | Poor — noise causes hallucinated words |

---

## Recommendation

### For VLSI Interview Application

| Use Case | Recommended Model | Why |
|---|---|---|
| **Production (primary STT)** | **gpt-4o-mini-transcribe** | Best accuracy + speed balance |
| **Final report transcript** | **ElevenLabs Scribe v2** | Most accurate, use for saved data |
| **Budget/testing** | **Groq Whisper Turbo** | Fast, acceptable for testing |
| **NOT recommended** | ElevenLabs Realtime | Too noisy, generates fake words |
| **NOT recommended** | xAI Grok STT | Fails on fast/low pitch speech |

### Recommended Architecture

```
Live Interview:     gpt-4o-mini-transcribe
                    |
                    Accurate + Fast (2-4s latency)
                    |
Post-Processing:    ElevenLabs Scribe v2 [optional]
                    |
                    Polish final transcript for report
```

---

## Conclusion

**gpt-4o-mini-transcribe** is the best overall choice for VLSI interview transcription:
- Accurate on domain-specific terms
- Fast enough (2-4s) for interview flow
- Handles varied speech patterns

For maximum accuracy in final reports, combine with **ElevenLabs Scribe v2** as a post-processing step.

Avoid **ElevenLabs Realtime** (noise hallucination) and **xAI Grok** (fails on fast/low pitch speech) for production interview use.
