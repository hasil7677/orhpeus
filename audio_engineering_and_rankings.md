# 🔬 Audio Engineering Deep-Dive, Model Rankings & TTS Correction

> Answering: What is a high-pass filter actually doing? Have we covered all the audio tricks? How do our models rank? How do you fix TTS output?

---

## Question 1: Is the High-Pass Filter "The Cleaning Engine"?

**No.** A high-pass filter is just **one tool** in the toolbox. It does exactly one thing:

### What a High-Pass Filter Actually Does

```
             What gets through a High-Pass Filter (cutoff: 80Hz)
  
  Amplitude
     │
     │  ❌ BLOCKED          ✅ PASSES THROUGH
     │  ◄──────────►  ◄─────────────────────────────────────►
     │
     │                    ┌──────────────────────────────────
     │                   /
     │                  /
     │                 /    ← This slope is the "roll-off"
     │________________/        (steeper = more aggressive)
     │
     └────┬────┬────┬────┬────┬────┬────┬────┬────┬────────►
         20   40   80  160  300  1k   2k   4k   8k  16k  Hz
              │         │
              │         └── Cutoff frequency
              │
              └── AC hum (50/60Hz), room rumble, wind,
                  breath pops — all live down here
```

**That's it.** It removes low-frequency junk. It does NOT:
- ❌ Remove background noise (that's spectral gating / DeepFilterNet)
- ❌ Remove echo (that's acoustic echo cancellation)
- ❌ Make quiet parts louder (that's compression / normalization)
- ❌ Remove harsh "s" sounds (that's de-essing)
- ❌ Make speech clearer (that's EQ / pre-emphasis)

### The Full Cleaning Engine = Many Tools Together

Here's every tool in the audio engineering toolbox, what it does, and when you need it:

```
    THE COMPLETE AUDIO CLEANING PIPELINE
    ════════════════════════════════════

    Raw Mic Audio
         │
         ▼
    ┌──────────────────────────────────────────┐
    │ 1. HIGH-PASS FILTER (80Hz)               │  ← Removes rumble, hum, wind
    │    "Let everything above 80Hz through"    │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ 2. NOISE GATE / SPECTRAL GATING          │  ← Removes constant background noise
    │    "Learn the noise profile, subtract it" │     (fan, AC, hiss, electrical buzz)
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ 3. DE-ESSING (optional)                  │  ← Tames harsh "s" / "sh" sounds
    │    "Compress only the 4-8kHz band"        │     (sibilance)
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ 4. EQ / PRE-EMPHASIS (optional)          │  ← Shape the tone
    │    "Boost clarity (2-4kHz), cut mud"       │     Make speech more intelligible
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ 5. COMPRESSION                           │  ← Even out loud/quiet parts
    │    "Make whispers and shouts similar vol"  │     Consistent dynamics
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ 6. NORMALIZATION (LUFS)                  │  ← Set consistent overall loudness
    │    "Bring everything to -16 LUFS"         │     Industry standard for speech
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    Clean, consistent, clear audio
```

---

## Question 2: Every Audio Manipulation Technique — What It Does

### The Filters (Frequency Manipulation)

Think of sound as a piano keyboard. Filters decide which keys are allowed to ring:

```
FILTER TYPES — Visualized on the Frequency Spectrum

    ┌─ HIGH-PASS ─────────────────────────────────────────┐
    │                                                      │
    │  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
    │  BLOCKED        PASSES THROUGH                       │
    │  Removes: rumble, hum, wind                          │
    └──────────────────────────────────────────────────────┘

    ┌─ LOW-PASS ──────────────────────────────────────────┐
    │                                                      │
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████ │
    │  PASSES THROUGH                         BLOCKED      │
    │  Removes: hiss, high-frequency noise                 │
    └──────────────────────────────────────────────────────┘

    ┌─ BAND-PASS ─────────────────────────────────────────┐
    │                                                      │
    │  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████  │
    │  BLOCKED    PASSES THROUGH           BLOCKED         │
    │  Isolates: e.g., the "speech band" (300Hz-3.4kHz)    │
    └──────────────────────────────────────────────────────┘

    ┌─ NOTCH (Band-Reject) ───────────────────────────────┐
    │                                                      │
    │  ░░░░░░░░░░░░░░░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
    │  PASSES            KILL          PASSES              │
    │  Removes: one specific frequency (e.g., 60Hz hum)    │
    └──────────────────────────────────────────────────────┘
```

```python
from scipy.signal import butter, sosfilt, iirnotch

def high_pass(audio, cutoff=80, sr=16000):
    sos = butter(5, cutoff, btype='high', fs=sr, output='sos')
    return sosfilt(sos, audio).astype(np.float32)

def low_pass(audio, cutoff=8000, sr=16000):
    sos = butter(5, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, audio).astype(np.float32)

def band_pass(audio, low=300, high=3400, sr=16000):
    sos = butter(5, [low, high], btype='band', fs=sr, output='sos')
    return sosfilt(sos, audio).astype(np.float32)

def notch_filter(audio, freq=60, Q=30, sr=16000):
    """Kill a specific frequency (e.g., 60Hz AC hum)."""
    b, a = iirnotch(freq, Q, sr)
    from scipy.signal import lfilter
    return lfilter(b, a, audio).astype(np.float32)
```

### Noise Reduction (Spectral Gating)

This is the "smart" noise removal. It works like this:

```
HOW SPECTRAL GATING WORKS

Step 1: Capture a "noise profile" (a moment of pure noise, no speech)

    Frequency →
    │████████████│   ← This is what your background noise looks like
    │████████████│      (e.g., fan at 200-500Hz, hiss at 4-8kHz)
    └────────────┘

Step 2: For every frame of audio, subtract the noise profile

    Speech + Noise:           Noise Profile:           Result:
    │██████████████│    -     │████████████│    =     │░░████████░░│
    │██████████████│          │████████████│          │░░████████░░│
    │██████████████│          │████████████│          │░░░░████░░░░│
      ▲ speech peaks            ▲ learned noise         ▲ clean speech
      above the noise           floor
```

```python
import noisereduce as nr

# Simple: let noisereduce estimate the noise
cleaned = nr.reduce_noise(y=audio, sr=16000, stationary=True)

# Better: provide an actual noise sample (first 0.5s of silence)
noise_sample = audio[:8000]  # 0.5s at 16kHz
cleaned = nr.reduce_noise(y=audio, sr=16000, y_noise=noise_sample)
```

### Parametric EQ (Shaping the Tone)

This is like a mixing board — boost or cut specific frequency ranges:

```
SPEECH FREQUENCY MAP — Where different sounds live

    20Hz ─────── 250Hz ─────── 1kHz ─────── 4kHz ─────── 16kHz
    │             │              │             │              │
    │  RUMBLE     │  BODY        │  CLARITY    │  PRESENCE    │  AIR
    │  Subwoofer  │  Warmth      │  Vowels     │  Consonants  │  Brightness
    │  Floor thud │  Chest tone  │  "a" "e"    │  "t" "k" "s" │  Shimmer
    │             │              │             │              │
    │  Usually    │  Cut 200Hz   │  Boost here │  Boost here  │  Cut here
    │  CUT this   │  if "muddy"  │  for intel- │  for "cut    │  if
    │             │              │  ligibility │  through"    │  "harsh"
```

```python
from scipy.signal import iirpeak

def presence_boost(audio, freq=3000, gain_db=3, Q=2, sr=16000):
    """Boost a specific frequency range to add clarity."""
    # Convert gain to linear
    import math
    gain = 10 ** (gain_db / 20)
    
    # Create a peak filter
    b, a = iirpeak(freq, Q, sr)
    from scipy.signal import lfilter
    boosted = lfilter(b * gain, a, audio)
    return boosted.astype(np.float32)
```

### Dynamic Range Compression

Makes quiet parts louder and loud parts quieter — essential for consistent voice output:

```
BEFORE COMPRESSION:

    Volume │
           │          ██
           │    ██    ██              ██
           │    ██    ██    ██  ██    ██
           │ ██ ██ ██ ██ ██ ██  ██ ██ ██
           │_██_██_██_██_██_██__██_██_██____
                "whisper"  "SHOUT"  "normal"

AFTER COMPRESSION (ratio 4:1, threshold -20dB):

    Volume │
           │    ██    ██              ██
           │    ██    ██    ██  ██    ██
           │ ██ ██ ██ ██ ██ ██  ██ ██ ██
           │ ██ ██ ██ ██ ██ ██  ██ ██ ██
           │_██_██_██_██_██_██__██_██_██____
               Everything much more even!
```

```python
def compress(audio, threshold_db=-20, ratio=4.0):
    """Simple dynamic range compressor."""
    threshold = 10 ** (threshold_db / 20)
    
    result = np.copy(audio)
    mask = np.abs(audio) > threshold
    
    # For samples above threshold, reduce by ratio
    excess = np.abs(audio[mask]) - threshold
    compressed_excess = excess / ratio
    result[mask] = np.sign(audio[mask]) * (threshold + compressed_excess)
    
    return result.astype(np.float32)
```

### De-Essing (Taming Harsh "S" Sounds)

Sibilance (harsh "sss", "shh") lives in the 4-8kHz range. A de-esser is a compressor that only acts on those frequencies:

```python
def de_ess(audio, sr=16000, threshold_db=-15, freq_low=4000, freq_high=8000):
    """Reduce sibilance by compressing the 4-8kHz band."""
    # Isolate the sibilant band
    sos = butter(4, [freq_low, freq_high], btype='band', fs=sr, output='sos')
    sibilant = sosfilt(sos, audio).astype(np.float32)
    
    # Compress only the sibilant energy
    threshold = 10 ** (threshold_db / 20)
    sibilant_compressed = compress(sibilant, threshold_db=threshold_db, ratio=6.0)
    
    # Replace the original sibilant band with compressed version
    result = audio - sibilant + sibilant_compressed
    return result.astype(np.float32)
```

### Loudness Normalization (LUFS — The Pro Way)

Peak normalization is amateur. **LUFS** is the broadcast standard — it measures perceived loudness, not just the tallest waveform spike:

```python
import pyloudnorm as pyln

def normalize_loudness(audio, sr=16000, target_lufs=-16.0):
    """
    Normalize to target LUFS (industry standard for speech).
    -16 LUFS = standard for streaming/podcasts
    -14 LUFS = louder (YouTube)
    -23 LUFS = broadcast TV (EBU R128)
    """
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)
    normalized = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    return normalized.astype(np.float32)
```

---

## Question 3: How Do Our Models Rank Against Industry?

### STT (Speech-to-Text) Rankings

```
    ACCURACY (lower WER = better)          LATENCY (lower = better)
    ═══════════════════════════            ════════════════════════

    🥇 Whisper Large-v3   ~4% WER         🥇 Deepgram Nova-3    <300ms
    🥈 Deepgram Nova-3    ~5-8% WER       🥈 Moonshine tiny     ~100ms
    🥉 Moonshine base     ~8-12% WER      🥉 Moonshine base     ~200ms
       Whisper small       ~10% WER           Whisper turbo      ~500ms+
       Moonshine tiny      ~12-15% WER        Whisper large      ~2-5s
```

> [!IMPORTANT]
> **Honest take on Moonshine:** It's NOT the most accurate STT out there. Whisper Large-v3 and Deepgram Nova-3 will beat it on raw accuracy. But Moonshine wins on **latency + local deployment**. For a voice agent where you need sub-200ms response, Moonshine is the right trade-off. If you're doing offline transcription where accuracy is king, use Whisper Large-v3 or Deepgram.

**Where Moonshine struggles:**
- Heavy accents / non-native speakers
- Multiple speakers talking over each other
- Very noisy environments (this is where audio cleaning REALLY matters)
- Domain-specific jargon (medical, legal)

**Where Moonshine shines:**
- Clean-ish audio, conversational speech
- Edge devices (Raspberry Pi, phones)
- Privacy-critical applications
- Real-time turn-taking conversations

### TTS (Text-to-Speech) Rankings

```
    NATURALNESS (MOS — higher = better)    LATENCY
    ═══════════════════════════════════    ════════════════════════

    🥇 ElevenLabs         ~4.5 MOS        🥇 Kokoro              ~50ms*
    🥈 Kokoro (82M)       ~4.4 MOS        🥈 ElevenLabs Flash    ~200ms
    🥉 XTTS v2            ~4.1 MOS        🥉 XTTS v2             ~500ms
       Bark               ~3.8 MOS           Bark                 ~2-5s
       pyttsx3 (offline)  ~2.5 MOS           pyttsx3              instant

    * Kokoro latency = time to first audio chunk on local GPU
```

> [!TIP]
> **Kokoro at 82M params scoring MOS ~4.4 is genuinely insane.** ElevenLabs likely has 500M-2B+ params. Kokoro gets 98% of the way there with 1/10th the model size. For a local stack, this is the best TTS available right now, period.

**Where Kokoro falls short vs. ElevenLabs:**
- Emotional range (ElevenLabs can do anger, excitement, sadness more convincingly)
- Voice cloning (ElevenLabs clones from short samples; Kokoro has fixed voice presets)
- Whispering, laughing, non-verbal vocal cues
- Some edge cases in prosody on long/complex sentences

**Where Kokoro wins:**
- Runs locally, offline, free
- No API rate limits or costs
- Data never leaves your machine
- Faster time-to-first-audio on local hardware

---

## Question 4: Fixing / Correcting TTS Output

Even good TTS models sometimes produce audio that needs fixing. Here's the post-processing pipeline:

### Common TTS Output Problems

| Problem | What It Sounds Like | Fix |
|---|---|---|
| **Inconsistent volume** | Some words loud, some quiet | LUFS normalization |
| **Harsh sibilance** | "Ssss" sounds sharp/painful | De-essing (4-8kHz compression) |
| **Muddy/boxy** | Sounds like speaking through a box | EQ cut at 200-400Hz |
| **Thin/tinny** | Lacks warmth, robotic | EQ boost at 200-300Hz |
| **No "presence"** | Hard to understand, flat | EQ boost at 2-4kHz |
| **Silent gaps** | Long pauses between sentences | Silence trimming |
| **Clipping** | Distorted loud parts | Limiter / compression |

### The TTS Post-Processing Pipeline

```python
import numpy as np
from scipy.signal import butter, sosfilt
import pyloudnorm as pyln

class TTSPostProcessor:
    """Polish TTS output to broadcast quality."""

    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        self.meter = pyln.Meter(sample_rate)

        # Pre-compute filters
        self.hp_sos = butter(4, 60, btype='high', fs=sample_rate, output='sos')
        self.lp_sos = butter(4, 12000, btype='low', fs=sample_rate, output='sos')

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Full post-processing chain for TTS output."""

        # 1. Remove DC offset and sub-bass rumble
        audio = audio - np.mean(audio)
        audio = sosfilt(self.hp_sos, audio).astype(np.float32)

        # 2. Remove ultrasonic content (anti-aliasing cleanup)
        audio = sosfilt(self.lp_sos, audio).astype(np.float32)

        # 3. Gentle de-essing
        audio = self._de_ess(audio)

        # 4. Light compression for consistency
        audio = self._compress(audio, threshold_db=-18, ratio=3.0)

        # 5. LUFS normalization to broadcast standard
        audio = self._normalize_lufs(audio, target=-16.0)

        # 6. Trim leading/trailing silence
        audio = self._trim_silence(audio)

        # 7. Final safety limiter (prevent clipping)
        audio = np.clip(audio, -0.99, 0.99)

        return audio.astype(np.float32)

    def _de_ess(self, audio, threshold_db=-20):
        """Reduce sibilance in the 4-8kHz range."""
        sos = butter(4, [4000, 8000], btype='band', fs=self.sr, output='sos')
        sibilant = sosfilt(sos, audio).astype(np.float32)

        threshold = 10 ** (threshold_db / 20)
        compressed = np.copy(sibilant)
        mask = np.abs(sibilant) > threshold
        if np.any(mask):
            excess = np.abs(sibilant[mask]) - threshold
            compressed[mask] = np.sign(sibilant[mask]) * (threshold + excess / 6.0)

        return (audio - sibilant + compressed).astype(np.float32)

    def _compress(self, audio, threshold_db=-18, ratio=3.0):
        """Gentle dynamic range compression."""
        threshold = 10 ** (threshold_db / 20)
        result = np.copy(audio)
        mask = np.abs(audio) > threshold
        if np.any(mask):
            excess = np.abs(audio[mask]) - threshold
            result[mask] = np.sign(audio[mask]) * (threshold + excess / ratio)
        return result.astype(np.float32)

    def _normalize_lufs(self, audio, target=-16.0):
        """Normalize to target LUFS."""
        try:
            loudness = self.meter.integrated_loudness(audio)
            if np.isinf(loudness):
                return audio
            return pyln.normalize.loudness(audio, loudness, target).astype(np.float32)
        except Exception:
            # Fallback to peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return (audio / peak * 0.95).astype(np.float32)
            return audio

    def _trim_silence(self, audio, threshold_db=-40, pad_ms=50):
        """Trim leading/trailing silence, keep a small pad."""
        threshold = 10 ** (threshold_db / 20)
        pad_samples = int(self.sr * pad_ms / 1000)

        # Find first and last non-silent sample
        above = np.where(np.abs(audio) > threshold)[0]
        if len(above) == 0:
            return audio

        start = max(0, above[0] - pad_samples)
        end = min(len(audio), above[-1] + pad_samples)
        return audio[start:end]


# Usage:
# post = TTSPostProcessor(sample_rate=24000)
# polished_audio = post.process(kokoro_raw_output)
```

---

## Summary: What's Actually In the Toolbox

```
┌──────────────────────────────────────────────────────────────┐
│                    AUDIO ENGINEERING TOOLBOX                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  FILTERS (frequency manipulation):                           │
│    ├── High-pass filter     → removes low rumble/hum         │
│    ├── Low-pass filter      → removes high hiss              │
│    ├── Band-pass filter     → isolates a frequency range     │
│    ├── Notch filter         → kills one exact frequency      │
│    └── Parametric EQ        → boost/cut any frequency range  │
│                                                               │
│  NOISE REMOVAL:                                               │
│    ├── Spectral gating      → subtract learned noise profile │
│    └── DeepFilterNet (DNN)  → AI-based noise removal         │
│                                                               │
│  DYNAMICS:                                                    │
│    ├── Compression          → even out loud/quiet parts      │
│    ├── Limiter              → prevent clipping (hard ceiling)│
│    └── Noise gate           → silence audio below threshold  │
│                                                               │
│  CORRECTION:                                                  │
│    ├── De-essing            → tame harsh "s" sounds          │
│    ├── De-reverb            → reduce room echo               │
│    └── Pre-emphasis         → boost highs to counter rolloff │
│                                                               │
│  NORMALIZATION:                                               │
│    ├── Peak normalization   → scale to max amplitude (basic) │
│    ├── RMS normalization    → scale to average energy         │
│    └── LUFS normalization   → perceptual loudness (industry) │
│                                                               │
│  VAD (voice activity detection):                             │
│    ├── Silero VAD           → neural, most accurate          │
│    └── WebRTC VAD           → lightweight, fast, basic       │
│                                                               │
└──────────────────────────────────────────────────────────────┘

High-pass filter = ONE item in this list. Not the whole engine.
```

---

> [!NOTE]
> **Bottom line on model choices:**
> - **Moonshine** is ~90-95% as good as the best cloud STT for clean conversational audio. It's the right call for local/real-time. Clean your audio well and that gap shrinks further.
> - **Kokoro** at MOS ~4.4 is genuinely competitive with ElevenLabs (~4.5). For a free, local model, this is as good as it gets in 2025/2026.
> - The **audio cleaning pipeline is more important than the model choice**. A mediocre STT model with clean audio will beat a great STT model with noisy audio every time.
