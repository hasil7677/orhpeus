# 🧠 How STT & TTS Actually Work + Building the Sleekest Local Pipeline

> A deep-dive into the internals, the local stack (Silero VAD + Moonshine + Kokoro), audio cleaning/enhancement, and how to architect it all cleanly.

---

## Part 1: How Speech-to-Text (STT) Actually Works

### The Big Picture

Your voice is just pressure waves hitting a microphone. STT turns those waves into text. Here's what happens at every step:

```
🎤 Air Pressure Waves
        │
        ▼
┌─────────────────────┐
│  1. DIGITIZATION     │   Mic samples air pressure 16,000× per second
│     (ADC)            │   Each sample = a 16-bit integer (-32768 to 32767)
└──────────┬──────────┘
           │  raw PCM bytes
           ▼
┌─────────────────────┐
│  2. FRAMING          │   Chop the stream into overlapping 25ms windows
│     + WINDOWING      │   Apply Hamming window to smooth edges
└──────────┬──────────┘
           │  frames
           ▼
┌─────────────────────┐
│  3. MEL SPECTROGRAM  │   FFT each frame → frequency bins → Mel scale
│     (Feature Extract)│   Output: 2D image (time × frequency)
└──────────┬──────────┘
           │  [T × 80] matrix
           ▼
┌─────────────────────┐
│  4. ENCODER          │   Transformer layers read the spectrogram
│     (Neural Network) │   Produces context-rich hidden representations
└──────────┬──────────┘
           │  encoded features
           ▼
┌─────────────────────┐
│  5. DECODER          │   Generates text tokens one at a time
│     + ALIGNMENT      │   Uses Attention or CTC to align audio→text
└──────────┬──────────┘
           │
           ▼
        📝 "Hello, how are you?"
```

### Step-by-Step Breakdown

#### Step 1: Digitization
Your microphone's ADC (Analog-to-Digital Converter) samples the air pressure wave:
- **Sample rate:** 16,000 Hz = 16,000 measurements per second
- **Bit depth:** 16-bit = each sample is a number from -32,768 to +32,767
- **Result:** A 1D array of integers — this is raw **PCM audio**

```python
# What raw audio looks like in memory
# 1 second of audio at 16kHz = 16,000 int16 values
import numpy as np
raw_audio = np.frombuffer(mic_bytes, dtype=np.int16)
# array([  23, -145,  312, -89, ...])  ← pressure values over time
```

#### Step 2: Framing + Windowing
Raw PCM is a 1D signal — useless for frequency analysis. We chop it into overlapping frames:
- **Frame size:** 25ms (~400 samples at 16kHz)
- **Hop size:** 10ms (~160 samples) — frames overlap by 15ms
- **Window function:** Hamming window smooths the edges to prevent spectral leakage

```
Audio:   [====|====|====|====|====|====]
Frame 1: [========]
Frame 2:     [========]          ← 60% overlap
Frame 3:         [========]
```

#### Step 3: Mel Spectrogram — The Key Transformation

This is where the magic happens. Each frame gets transformed from time-domain to frequency-domain:

1. **FFT (Fast Fourier Transform):** Decomposes each frame into its frequency components
2. **Power spectrum:** Square the magnitudes to get energy at each frequency
3. **Mel filter bank:** Apply triangular filters spaced on the **Mel scale** (mimics human hearing — we hear differences between 100Hz and 200Hz more than between 8000Hz and 8100Hz)
4. **Log:** Take the log of energies (humans perceive loudness logarithmically)

```
          Frequency (Mel Scale)
    Low ◄─────────────────────► High
    ┌──────────────────────────────┐
    │░░░░░░░░░                     │  Frame 1
    │░░░░░░░░░░░                   │  Frame 2
    │░░░░░░░░░░░░░░                │  Frame 3   ← Vowel sounds
    │░░░░░░░░░░░                   │  Frame 4      light up here
    │░░░                           │  Frame 5   ← Silence
    │░░░░░░░░░░░░░░░░░░░░░         │  Frame 6
    │░░░░░░░░░░░░░░░░░░░░░░░░      │  Frame 7   ← "S" sound lights
    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  Frame 8      up high frequencies
    └──────────────────────────────┘
    Time ▼

    This 2D matrix IS the mel spectrogram.
    Typically [num_frames × 80] (80 mel bins)
```

> [!TIP]
> Think of a mel spectrogram as an **X-ray of sound**. It shows you which frequencies have energy at each moment in time. Vowels light up the low-mid frequencies. Consonants like "s" and "t" light up high frequencies. Silence is dark.

#### Step 4 + 5: Encoder-Decoder (The Neural Network)

Modern STT models (Whisper, Moonshine, etc.) use a **Transformer encoder-decoder**:

```
Mel Spectrogram [T×80]
        │
        ▼
┌──────────────────┐
│    ENCODER        │    Multiple Transformer layers
│                   │    Self-attention: each frame "looks at"
│  [Attention] ×N   │    all other frames to understand context
│                   │    ("is this 's' or 'z'? look at what comes next")
└──────────┬───────┘
           │  Hidden states [T × d_model]
           ▼
┌──────────────────┐
│    DECODER        │    Autoregressive: generates one token at a time
│                   │
│  Cross-Attention  │    "Which part of the audio am I transcribing now?"
│  + Self-Attention │    Looks at encoded audio AND previously generated text
│                   │
└──────────┬───────┘
           │
           ▼
    Token: "H" → "e" → "l" → "l" → "o" → ...
```

**The alignment problem:** Audio is ~1,600 frames per second, but text might be 5 words per second. How does the model know which frames correspond to which words?

- **CTC (Connectionist Temporal Classification):** Outputs one character per frame, with "blank" tokens for silence. Collapses `HH_ee_ll_ll__oo` → `Hello`. Simple but can't model language context well.
- **Attention:** The decoder learns to "attend" to the right part of the audio for each output token. More powerful but needs more data.
- **Hybrid (CTC + Attention):** Most modern models use both — CTC for rough alignment, attention for fine-grained decoding.

---

## Part 2: How Text-to-Speech (TTS) Actually Works

TTS is essentially STT **in reverse** — but it's harder because you need to generate the audio waveform, not just classify it.

```
📝 "Hello, how are you?"
        │
        ▼
┌─────────────────────┐
│  1. TEXT ANALYSIS     │   Normalize text → Convert to phonemes
│     (Front-end)      │   "Hello" → /h ɛ l oʊ/
└──────────┬──────────┘
           │  phoneme sequence
           ▼
┌─────────────────────┐
│  2. ACOUSTIC MODEL   │   Predict how each phoneme should SOUND
│     (The Brain)      │   Outputs: Mel Spectrogram + Duration + Pitch
└──────────┬──────────┘
           │  mel spectrogram [T × 80]
           ▼
┌─────────────────────┐
│  3. VOCODER          │   Convert spectrogram → actual audio waveform
│     (The Voice)      │   Reconstructs the missing phase information
└──────────┬──────────┘
           │
           ▼
        🔊 Audio waveform (PCM samples)
```

### Stage 1: Text Analysis (Front-End)

```
Input:   "Dr. Smith paid $5.50 for 2 items."
                │
                ▼
Normalized: "Doctor Smith paid five dollars and fifty cents for two items."
                │
                ▼
Phonemes:   /d ɑː k t ər   s m ɪ θ   p eɪ d   f aɪ v .../
```

- **Text normalization:** Expand abbreviations, numbers, symbols
- **G2P (Grapheme-to-Phoneme):** Convert letters to sound units. "knight" → /naɪt/ (the 'k' is silent!)

### Stage 2: Acoustic Model

This is where different TTS architectures diverge. For **Kokoro/StyleTTS 2**:

```
Phonemes: /h ɛ l oʊ/
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌──────────┐        ┌──────────────┐
│ Duration  │        │ Style Vector  │
│ Predictor │        │ (Prosody,     │
│           │        │  Emotion,     │
│ How long  │        │  Speaker ID)  │
│ each      │        │              │
│ phoneme   │        │ "sound like  │
│ lasts     │        │  Rachel,     │
│           │        │  calm tone"  │
└─────┬────┘        └──────┬───────┘
      │                     │
      └──────────┬──────────┘
                 │
                 ▼
        ┌──────────────┐
        │ Mel Spectrogram│
        │  Generator     │   Diffusion / Flow matching
        │               │   refines the spectrogram
        └───────┬───────┘
                │
                ▼
          Mel Spectrogram [T × 80]
```

The **style vector** is what makes modern TTS sound human — it encodes:
- **Prosody:** Rhythm and timing (stressed vs. unstressed syllables)
- **Pitch contour:** How the voice rises and falls
- **Speaking rate:** Speed variations within the sentence
- **Emotion/affect:** Happy, sad, calm, excited

### Stage 3: Vocoder — The Final Mile

The vocoder converts the mel spectrogram into actual audio samples. This is the hardest part because the mel spectrogram is **lossy** — it throws away phase information.

```
Mel Spectrogram (what we have):
    Tells us WHAT frequencies are present at each moment
    Does NOT tell us the PHASE of each frequency

Phase (what's missing):
    The precise timing offset of each sine wave
    Without it, audio sounds robotic / noisy

Neural Vocoder (what solves it):
    A GAN trained to hallucinate realistic phase
    Generator produces waveform, Discriminator judges if it sounds real
```

**HiFi-GAN** (used in many TTS systems):
- Generator: Multi-scale upsampling with residual blocks
- Discriminators: Multi-Period + Multi-Scale discriminators judge both fine and coarse audio quality
- Result: Near-perfect audio quality at real-time speed

**iSTFTNet** (used in Kokoro/StyleTTS 2):
- Instead of generating raw samples, it predicts STFT coefficients (magnitude + phase)
- Then uses inverse STFT to reconstruct the waveform
- Faster than HiFi-GAN because it works in frequency domain

---

## Part 3: Your Local Stack — The Pieces

### 3.1 Silero VAD — The Gatekeeper

**What it does:** Tells you if a chunk of audio contains speech (probability 0.0 → 1.0).

**Why you need it:**  You don't want to send silence to your STT model. VAD acts as a smart gate:

```
Audio Stream: ───silence───[SPEECH STARTS]──"hey how are"──[SPEECH ENDS]───silence───
                                │                                │
Silero VAD:   ──0.02──0.01──0.03──0.94──0.97──0.98──0.95──0.12──0.03──0.01──
                                │                                │
Trigger:                    START recording              STOP → send to STT
```

**How it works internally:**
- Small neural network (~1-2MB) trained on thousands of hours of speech/non-speech audio
- Input: 32ms chunks of 16kHz mono float32 PCM
- Output: Single float — probability of speech
- Runs in **<1ms per chunk on CPU** — basically free

```python
# Silero VAD usage
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

model = load_silero_vad()

# For a file:
wav = read_audio('audio.wav')  # Returns float32 tensor, 16kHz
timestamps = get_speech_timestamps(wav, model, return_seconds=True)
# [{'start': 0.5, 'end': 2.3}, {'start': 4.1, 'end': 6.8}]

# For real-time streaming:
speech_prob = model(audio_chunk_tensor, 16000).item()
if speech_prob > 0.5:
    # Speech detected → start buffering for STT
```

---

### 3.2 Moonshine — The Ear (Local STT)

**What it is:** Tiny, fast, local speech-to-text. Built by Useful Sensors. Two sizes:
- `moonshine-base`: 58M params, ~200ms for 5s of audio
- `moonshine-tiny`: 27M params, even faster

**Why it's special vs. Whisper:**

| Feature | Whisper | Moonshine |
|---|---|---|
| **Input window** | Fixed 30s (padded with zeros) | Variable length (no waste) |
| **Latency** | ~1-3s per chunk | ~200ms per chunk |
| **State caching** | ❌ Recomputes everything | ✅ KV cache across chunks |
| **Position encoding** | Absolute (fixed length) | RoPE (handles any length) |
| **Best for** | Batch transcription | Real-time conversation |

**Architecture deep-dive:**

```
Audio (variable length)
        │
        ▼
┌───────────────────┐
│ Feature Extractor  │   Conv layers → Mel-like features
│ (Convolutional)    │   Handles any input length (no padding!)
└──────────┬────────┘
           │
           ▼
┌───────────────────┐
│ Encoder            │   Transformer with RoPE
│ (Transformer)      │   
│                    │   Key innovation: RoPE (Rotary Position Embedding)
│ Self-Attention     │   encodes position as rotation in vector space
│ with RoPE          │   → works with ANY sequence length
│                    │   
│ KV Cache ──────────│──→ Saved! Don't recompute for overlapping audio
└──────────┬────────┘
           │
           ▼
┌───────────────────┐
│ Decoder            │   Generates text tokens
│ (Transformer)      │   Cross-attention to encoder output
│                    │   Also uses KV cache for efficiency
└──────────┬────────┘
           │
           ▼
    "hello how are you"
```

**Usage:**
```python
from moonshine import MoonshineModel

model = MoonshineModel("moonshine/base")  # or "moonshine/tiny"

# Transcribe audio
text = model.transcribe(audio_array, sample_rate=16000)
```

**ONNX deployment (most portable):**
```python
# pip install onnxruntime moonshine-onnx
from moonshine_onnx import MoonshineOnnxModel

model = MoonshineOnnxModel(model_name="moonshine/base")
text = model.generate(audio_float32_array)
```

---

### 3.3 Kokoro — The Voice (Local TTS)

**What it is:** 82M parameter TTS model based on StyleTTS 2. State-of-the-art quality from a tiny model.

**Why 82M params is insane:** ElevenLabs and other commercial TTS models are likely 500M-2B+ parameters. Kokoro achieves comparable quality at a fraction of the size.

**Architecture (StyleTTS 2 + iSTFTNet):**

```
Text: "Hello, how are you?"
        │
        ▼
┌───────────────┐
│ Text Encoder   │   Phoneme embedding + Transformer
│               │   Understands linguistic structure
└───────┬───────┘
        │
        ├─── Duration Predictor ──→ How long each sound lasts
        │
        ├─── Pitch Predictor ────→ F0 contour (voice melody)
        │
        ▼
┌───────────────┐
│ Style Encoder  │   Learns speaking style from voice ID
│               │   Each voice (af_heart, am_michael) = different style
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Decoder        │   Combines text features + style + duration + pitch
│               │   → Generates mel spectrogram
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ iSTFTNet       │   Vocoder: mel → waveform
│ (Vocoder)      │   Predicts STFT magnitude + phase
│               │   Inverse STFT → audio samples
└───────┬───────┘
        │
        ▼
🔊 24kHz PCM audio
```

**Usage:**
```python
from kokoro import KPipeline
import soundfile as sf

# Initialize
pipeline = KPipeline(lang_code='a')  # 'a' = American English

# Generate speech
generator = pipeline(
    "Hello! This is running completely locally.",
    voice='af_heart',   # Voice preset
    speed=1.0           # 0.5 = slow, 2.0 = fast
)

# Kokoro yields chunks (for streaming!)
for i, (graphemes, phonemes, audio) in enumerate(generator):
    # audio = numpy array, 24kHz sample rate
    sf.write(f'chunk_{i}.wav', audio, 24000)
```

**Available voices:**
| Voice ID | Description |
|---|---|
| `af_heart` | American Female, warm/natural |
| `af_bella` | American Female, clear |
| `am_michael` | American Male, professional |
| `bf_emma` | British Female |
| `bm_george` | British Male |

---

## Part 4: Audio Cleaning & Enhancement

This is where you make shitty microphone audio actually usable. Here's the full chain:

### The Audio Preprocessing Pipeline

```
Raw Mic Input (noisy, variable level, DC offset, rumble)
        │
        ▼
┌───────────────────┐
│ 1. FORMAT          │   Ensure: 16kHz, mono, float32 normalized [-1, 1]
│    STANDARDIZE     │
└──────────┬────────┘
           │
           ▼
┌───────────────────┐
│ 2. DC OFFSET       │   Remove the constant bias from the signal
│    REMOVAL         │   signal -= np.mean(signal)
└──────────┬────────┘
           │
           ▼
┌───────────────────┐
│ 3. HIGH-PASS       │   Remove frequencies below ~80Hz
│    FILTER          │   Kills: AC hum (50/60Hz), room rumble, wind
└──────────┬────────┘
           │
           ▼
┌───────────────────┐
│ 4. NOISE           │   Spectral gating OR deep learning
│    REDUCTION       │   Option A: noisereduce (simple, fast)
│                    │   Option B: DeepFilterNet (better, heavier)
└──────────┬────────┘
           │
           ▼
┌───────────────────┐
│ 5. NORMALIZATION   │   Scale to consistent loudness
│                    │   Peak normalize or RMS normalize
└──────────┬────────┘
           │
           ▼
┌───────────────────┐
│ 6. PRE-EMPHASIS    │   Boost high frequencies (+6dB/octave)
│    (Optional)      │   Compensates for natural speech roll-off
└──────────┬────────┘
           │
           ▼
Clean audio → ready for VAD + STT
```

### Implementation

```python
import numpy as np
from scipy.signal import butter, sosfilt
import noisereduce as nr

class AudioPreprocessor:
    """Clean and enhance raw microphone audio for STT."""

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        # Pre-compute high-pass filter coefficients (80Hz cutoff)
        self.hp_sos = butter(5, 80, btype='high', fs=sample_rate, output='sos')
        # Noise profile (updated from first ~0.5s of "silence")
        self.noise_profile = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Full preprocessing chain. Input: int16 or float32 array."""
        # 1. Standardize to float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # 2. DC offset removal
        audio = audio - np.mean(audio)
        
        # 3. High-pass filter (remove rumble/hum)
        audio = sosfilt(self.hp_sos, audio).astype(np.float32)
        
        # 4. Noise reduction (spectral gating)
        audio = nr.reduce_noise(
            y=audio,
            sr=self.sr,
            stationary=True,     # Good for constant noise (fan, AC)
            prop_decrease=0.75,  # How aggressive (0=none, 1=max)
            n_fft=512,           # Smaller = lower latency
            hop_length=128,
        )
        
        # 5. Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95  # Leave 5% headroom
        
        return audio.astype(np.float32)

    def set_noise_profile(self, noise_audio: np.ndarray):
        """Capture ambient noise for better reduction."""
        self.noise_profile = noise_audio
```

### Deep Enhancement: DeepFilterNet

If `noisereduce` isn't cutting it (it can create "musical noise" artifacts), use **DeepFilterNet** — a DNN specifically trained for real-time speech enhancement:

```python
# pip install deepfilternet
from df.enhance import enhance, init_df

# Initialize (downloads model on first run, ~10MB)
model, df_state, _ = init_df()

# Enhance audio (numpy float32, any sample rate)
enhanced_audio = enhance(model, df_state, noisy_audio)
# Result: Dramatically cleaner speech, even in noisy environments
```

> [!IMPORTANT]
> **DeepFilterNet vs noisereduce — when to use which:**
> | Scenario | Use |
> |---|---|
> | Quiet room, just removing slight hiss | `noisereduce` (lighter, simpler) |
> | Noisy environment (café, street, fan) | DeepFilterNet (much better) |
> | Edge device / Raspberry Pi | `noisereduce` (no PyTorch needed) |
> | Real-time latency critical (<10ms) | `noisereduce` with small FFT |

---

## Part 5: The Sleek Pipeline — Putting It All Together

### Architecture: Event-Driven State Machine

The cleanest way to build this is as a **state machine** with async events:

```
                    ┌──────────┐
                    │  IDLE    │ ◄─── Not listening
                    └────┬─────┘
                         │ VAD detects speech
                         ▼
                    ┌──────────┐
          ┌────────│ LISTENING │ ◄─── Buffering audio
          │        └────┬─────┘
          │             │ VAD detects silence (endpointing)
          │             ▼
          │        ┌──────────────┐
          │        │ PROCESSING   │ ◄─── STT → LLM → TTS
          │        └────┬─────────┘
          │             │ TTS starts producing audio
          │             ▼
          │        ┌──────────┐
          └───────►│ SPEAKING │ ◄─── Playing AI response
   (barge-in:      └────┬─────┘
    user speaks          │ Audio playback finished
    during AI            │
    response)            ▼
                    ┌──────────┐
                    │  IDLE    │
                    └──────────┘
```

### The Complete Sleek Pipeline

```python
"""
voice_pipeline.py — The sleekest local voice AI pipeline.

Stack: Silero VAD → Moonshine STT → [Your LLM] → Kokoro TTS

Usage:
    python voice_pipeline.py
"""

import asyncio
import numpy as np
import threading
import queue
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import deque

# ─── Models ─────────────────────────────────────────────────
from silero_vad import load_silero_vad
from moonshine_onnx import MoonshineOnnxModel
from kokoro import KPipeline

# ─── Audio Preprocessing ────────────────────────────────────
from scipy.signal import butter, sosfilt
import noisereduce as nr


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    # Audio
    sample_rate: int = 16000
    chunk_duration_ms: int = 32           # Silero VAD needs 32ms+
    channels: int = 1
    
    # VAD
    vad_threshold: float = 0.5            # Speech probability threshold
    silence_timeout_ms: int = 700         # How long silence = "done talking"
    min_speech_ms: int = 250              # Ignore speech shorter than this
    
    # STT
    stt_model: str = "moonshine/base"     # or "moonshine/tiny" for speed
    
    # TTS
    tts_voice: str = "af_heart"           # Kokoro voice preset
    tts_speed: float = 1.0
    tts_sample_rate: int = 24000
    
    # Audio Enhancement
    enable_noise_reduction: bool = True
    highpass_cutoff: int = 80             # Hz

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


# ═══════════════════════════════════════════════════════════
# Pipeline State Machine
# ═══════════════════════════════════════════════════════════
class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class AudioPreprocessor:
    """Cleans raw mic audio in real-time."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.hp_sos = butter(
            4, config.highpass_cutoff, btype='high',
            fs=config.sample_rate, output='sos'
        )
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """int16 → cleaned float32 [-1, 1]"""
        # Standardize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # DC offset removal
        audio -= np.mean(audio)

        # High-pass filter
        audio = sosfilt(self.hp_sos, audio).astype(np.float32)

        # Noise reduction (if enabled and chunk is large enough)
        if self.config.enable_noise_reduction and len(audio) > 1024:
            audio = nr.reduce_noise(
                y=audio, sr=self.config.sample_rate,
                stationary=True, prop_decrease=0.6,
                n_fft=512, hop_length=128,
            )

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak * 0.95

        return audio.astype(np.float32)


class VoicePipeline:
    """
    Event-driven voice pipeline.
    
    Flow: Mic → Clean → VAD → STT → LLM → TTS → Speaker
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.state = State.IDLE
        
        # ── Load models (one-time cost) ──
        print("⏳ Loading models...")
        self.vad = load_silero_vad()
        self.stt = MoonshineOnnxModel(model_name=self.config.stt_model)
        self.tts = KPipeline(lang_code='a')
        self.preprocessor = AudioPreprocessor(self.config)
        print("✅ All models loaded.")

        # ── Buffers ──
        self.speech_buffer: list[np.ndarray] = []  # Accumulate speech chunks
        self.silence_counter: int = 0               # Track consecutive silence
        
        # ── Conversation ──
        self.conversation_history: list[dict] = []

    def on_audio_chunk(self, raw_chunk: np.ndarray) -> np.ndarray | None:
        """
        Process a single audio chunk from the microphone.
        Returns TTS audio to play, or None.
        
        This is the heart of the pipeline — a clean state machine.
        """
        # ── Preprocess ──
        clean = self.preprocessor.process(raw_chunk)
        
        # ── VAD ──
        import torch
        chunk_tensor = torch.from_numpy(clean)
        speech_prob = self.vad(chunk_tensor, self.config.sample_rate).item()
        is_speech = speech_prob > self.config.vad_threshold

        # ── State machine ──
        if self.state == State.IDLE:
            if is_speech:
                self.state = State.LISTENING
                self.speech_buffer = [clean]
                self.silence_counter = 0
                print("🎤 Listening...")
            return None

        elif self.state == State.LISTENING:
            if is_speech:
                self.speech_buffer.append(clean)
                self.silence_counter = 0
            else:
                self.silence_counter += self.config.chunk_duration_ms
                self.speech_buffer.append(clean)  # Keep some trailing context
                
                if self.silence_counter >= self.config.silence_timeout_ms:
                    # Check minimum speech duration
                    total_ms = len(self.speech_buffer) * self.config.chunk_duration_ms
                    if total_ms >= self.config.min_speech_ms:
                        return self._process_utterance()
                    else:
                        # Too short — probably noise, discard
                        print("  (too short, ignoring)")
                        self.state = State.IDLE
                        self.speech_buffer = []
            return None

        elif self.state == State.SPEAKING:
            if is_speech:
                # Barge-in! User started talking over the AI
                print("⚡ Barge-in detected!")
                self.state = State.LISTENING
                self.speech_buffer = [clean]
                self.silence_counter = 0
                return np.array([], dtype=np.float32)  # Signal to stop playback
            return None

        return None

    def _process_utterance(self) -> np.ndarray | None:
        """STT → LLM → TTS pipeline for a complete utterance."""
        self.state = State.PROCESSING

        # ── 1. Combine buffered audio ──
        full_audio = np.concatenate(self.speech_buffer)
        self.speech_buffer = []

        # ── 2. STT (Moonshine) ──
        print("  🧠 Transcribing...")
        transcript = self.stt.generate(full_audio)
        if not transcript or not transcript.strip():
            print("  (empty transcript)")
            self.state = State.IDLE
            return None
        print(f"  📝 You said: \"{transcript}\"")

        # ── 3. LLM (plug in your own) ──
        response = self._get_llm_response(transcript)
        print(f"  🤖 AI: \"{response}\"")

        # ── 4. TTS (Kokoro) ──
        print("  🔊 Synthesizing speech...")
        audio_chunks = []
        for _, _, audio in self.tts(
            response,
            voice=self.config.tts_voice,
            speed=self.config.tts_speed,
        ):
            audio_chunks.append(audio)

        self.state = State.SPEAKING
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        
        self.state = State.IDLE
        return None

    def _get_llm_response(self, user_text: str) -> str:
        """
        Plug in ANY LLM here:
        - Groq (cloud, fastest)
        - Ollama (local, private)
        - llama-cpp-python (local, no server)
        - OpenAI / Anthropic (cloud)
        """
        # Example with Groq:
        # from groq import Groq
        # client = Groq()
        # response = client.chat.completions.create(
        #     model="llama-3.3-70b-versatile",
        #     messages=[
        #         {"role": "system", "content": "You are a concise voice assistant."},
        #         *self.conversation_history,
        #         {"role": "user", "content": user_text},
        #     ]
        # )
        # return response.choices[0].message.content

        # Placeholder — replace with your LLM call
        self.conversation_history.append({"role": "user", "content": user_text})
        reply = f"I heard you say: {user_text}"
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply


# ═══════════════════════════════════════════════════════════
# Main Loop (with PyAudio)
# ═══════════════════════════════════════════════════════════
def main():
    import pyaudio

    config = PipelineConfig()
    pipeline = VoicePipeline(config)

    p = pyaudio.PyAudio()

    # Mic input
    mic = p.open(
        format=pyaudio.paInt16,
        channels=config.channels,
        rate=config.sample_rate,
        input=True,
        frames_per_buffer=config.chunk_samples,
    )

    # Speaker output
    speaker = p.open(
        format=pyaudio.paFloat32,
        channels=config.channels,
        rate=config.tts_sample_rate,
        output=True,
    )

    print("\n🎙️  Voice Pipeline Ready. Speak!")
    print("   Press Ctrl+C to stop.\n")

    try:
        while True:
            # Read mic chunk
            raw = mic.read(config.chunk_samples, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16)

            # Process through pipeline
            tts_audio = pipeline.on_audio_chunk(chunk)

            if tts_audio is not None and len(tts_audio) > 0:
                # Play the response
                speaker.write(tts_audio.astype(np.float32).tobytes())
                pipeline.state = State.IDLE
                print("  ✅ Done speaking.\n")

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        mic.close()
        speaker.close()
        p.terminate()


if __name__ == "__main__":
    main()
```

---

## Part 6: Cloud vs. Local — The Tradeoff

| Dimension | Cloud Stack | Local Stack |
|---|---|---|
| **STT** | Deepgram Nova-3 | Moonshine base/tiny |
| **LLM** | Groq (Llama 3.3 70B) | Ollama / llama.cpp (Llama 3.1 8B) |
| **TTS** | ElevenLabs Flash v2.5 | Kokoro (82M) |
| **Latency** | ~500-800ms (network bound) | ~300-600ms (compute bound) |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ (very close!) |
| **Cost** | $$ per request | Free after setup |
| **Privacy** | Data leaves your machine | 100% local |
| **Offline** | ❌ Needs internet | ✅ Works anywhere |
| **GPU needed** | ❌ (cloud does compute) | Helps but not required |
| **Setup time** | 10 min (API keys) | 30 min (model downloads) |

---

## Dependencies for the Local Stack

```
# requirements-local.txt
silero-vad>=5.0
moonshine-onnx>=0.1.0      # or: moonshine
kokoro>=0.9.0
soundfile>=0.12.0
numpy>=1.24.0
scipy>=1.10.0
noisereduce>=3.0.0
pyaudio>=0.2.14
torch>=2.0.0               # For Silero VAD
onnxruntime>=1.16.0         # For Moonshine ONNX inference

# Optional (better audio enhancement):
# deepfilternet>=0.5.0

# Optional (local LLM):
# groq>=0.9.0              # Cloud LLM (fastest)
# ollama>=0.3.0            # Local LLM server
```

---

> [!NOTE]
> **The "sleek" factor comes from the architecture, not the code volume.** The state machine pattern (`IDLE → LISTENING → PROCESSING → SPEAKING`) keeps the logic clean and debuggable. The preprocessing chain ensures your STT model gets clean audio. And the chunked pipeline (VAD gates STT, STT feeds LLM, LLM streams to TTS) means each piece does one thing well.
