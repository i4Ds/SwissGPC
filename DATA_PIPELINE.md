# Swiss German TTS Data Pipeline (Clean + Natural Speech)

This document describes the finalized end-to-end pipeline for extracting clean, natural, and weakly-labeled Swiss German speech suitable for TTS training. The design prioritizes **audio-first segmentation**, **speaker purity**, **paralinguistic naturalness**, and **minimal irreversible decisions**.

---

## 1. Core Principles

- Audio-first, text-later
- No hard sentence boundaries early
- Single-speaker purity over heuristic overlap detection
- Weak labels with confidence, not ground truth
- Preserve raw segments, derive views later

---

## 2. Models Used

### Voice Activity Detection & Diarization
- **pyannote**
  - VAD
  - Speaker diarization
  - Frame-level speaker posteriors

### Audio Event Tagging (Paralinguistics, Music)
- **PANNs / AudioSet CNN14**
  - Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn
  - Events used:
    - Laughter
    - Cough
    - Sneeze
    - Breathing
    - Music
    - Speech

### Emotion Recognition (Continuous)
- **MERaLiON-SER-v1**
  - https://huggingface.co/MERaLiON/MERaLiON-SER-v1
  - Outputs:
    - Valence
    - Arousal
    - Dominance

### Automatic Speech Recognition
- **Whisper Large v3**
  - Language: Standard German (DE)

### Swiss German Pseudo-Text
- **Meta Omni**
  - German → Swiss German
  - In-context learning with examples from the specific dialect.
  - Dialect-conditioned prompts

---

## 3. Pipeline Overview

### Step 1: Voice Activity Detection (VAD)
- Run VAD on raw audio
- Produce atomic speech segments
- Enforce silence thresholds (e.g. min silence ≈ 0.4s)
- Drop segments:
  - shorter than 2s
  - longer than 15s

Output: **atomic speech segments (audio-only)**

---

### Step 2: Diarization (Full Audio)
- Run speaker diarization on full audio
- Obtain frame-level speaker labels/posteriors

---

### Step 3: Speaker Purity Filtering
For each VAD segment:
- Compute speaker purity:
```

purity = frames_of_top_speaker / total_frames

```
- Assign speaker ID = dominant speaker
- Drop segment if:
- purity < 0.90
- effective speech duration < 0.90

**No explicit overlap detection required**  
Low purity implicitly captures:
- overlapping speech
- background music
- applause
- cross-talk

Output: **clean, single-speaker segments**

---

### Step 4: AudioSet Event Tagging (Naturalness)
- Run AudioSet tagging on remaining segments
- Use sliding windows:
- window: 1–2s
- hop: 0.5–1s
- Aggregate per segment (max or mean)

Store probabilities for:
- Speech
- Music
- Laughter
- Cough
- Sneeze
- Breathing

Filter:
- Drop if music probability is high
- Keep speech-dominant segments

Event attribution:
- Assign events to speaker by intersecting with diarization labels
- Only keep events with high event-level purity

Derived flags:
- `has_laughter`
- `has_cough`
- `has_sneeze`
- `has_breath`

---

### Step 5: Dialect Identification
- Run dialect classifier on clean segments (src/classification_i4ds)
- Majority-vote on same speaker in the same file.
- Keep only high-confidence Swiss German.
- Store:
- dialect label
- confidence / purity

Used later for:
- Meta Omni in context learning.

---

### Step 6: ASR (Standard German)
- Transcribe with Whisper Large v3
- Optional short context within same speaker turn
- Store:
- transcript
- confidence proxies
- Drop segments with low ASR quality

---

### Step 7: Swiss German Pseudo-Text
- Use Meta Omni with:
- dialect tag
- in-context examples
- Generate Swiss German text
- Store as auxiliary field
- Mark as machine-generated

---

### Step 8: Continuous Emotion Tagging
- Run MERaLiON-SER-v1
- Store continuous values:
- Valence
- Arousal
- Dominance
- Treat as weak labels with confidence

---

## 4. Stored Metadata (Per Segment)

Minimum recommended fields:
- audio_path, start_time, end_time, duration
- speaker_id, speaker_purity
- p_speech, p_music
- p_laughter, p_cough, p_sneeze, p_breath
- dialect_label, dialect_confidence
- whisper_text_de, whisper_confidence
- omni_text_ch, omni_prompt_version
- valence, arousal, dominance

---