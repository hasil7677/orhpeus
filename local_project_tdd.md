# 🚀 Local Voice AI Pipeline — Technical Design Document (TDD) & Developer Guide

**Version:** 1.0.0
**Purpose:** This document serves as the canonical starting guide, harness specification, and technical blueprint for developers building or extending the 100% local voice AI pipeline.

---

## 1. System Requirements & Setup

Before touching the code, the environment must be configured exactly as specified to ensure hardware acceleration (if available) and audio driver stability.

### 1.1 Hardware Minimums
*   **CPU:** 4+ cores (AVX2 support required for ONNX Runtime).
*   **RAM:** 8GB minimum (16GB recommended to hold STT, TTS, and LLM in memory).
*   **GPU (Optional but recommended):** NVIDIA GPU with CUDA 11.8+ or 12.1+ for LLM and TTS acceleration.
*   **Audio:** A physical or virtual microphone and speaker device accessible via PortAudio.

### 1.2 Environment Bootstrap
We strictly use `venv` to avoid global package pollution.

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows

# 2. Install PyAudio dependencies (Windows usually has binaries, Linux needs portaudio)
# sudo apt-get install portaudio19-dev  (Linux only)

# 3. Install core ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # (If using NVIDIA GPU)
pip install onnxruntime  # (Use onnxruntime-gpu if NVIDIA GPU is present)

# 4. Install pipeline dependencies
pip install silero-vad moonshine-onnx kokoro soundfile pydantic pyaudio noisereduce pyloudnorm
```

---

## 2. Global Configuration Schema

We use `pydantic` for strict type validation of the configuration. This ensures that a typo in `.env` doesn't crash the pipeline at runtime.

**File:** `config.py`

```python
from pydantic import BaseModel, Field

class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000, description="Input sample rate for VAD/STT")
    tts_sample_rate: int = Field(default=24000, description="Output sample rate from Kokoro")
    chunk_ms: int = Field(default=32, description="Buffer size in milliseconds")
    channels: int = Field(default=1, description="Mono audio required")

class VADConfig(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Speech probability threshold")
    silence_timeout_ms: int = Field(default=700, description="Consecutive silence before cutting utterance")
    min_speech_ms: int = Field(default=250, description="Minimum duration to be considered valid speech")

class ModelConfig(BaseModel):
    stt_model: str = Field(default="moonshine/base")
    tts_voice: str = Field(default="af_heart")
    tts_speed: float = Field(default=1.0)
    llm_api_base: str = Field(default="http://localhost:11434/v1", description="Ollama local endpoint")

class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    models: ModelConfig = ModelConfig()
```

---

## 3. Data Structures & Types

Standardizing what flows between components is critical. 

**File:** `core/types.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class AudioChunk:
    """Raw audio data passed around the system."""
    data: np.ndarray      # float32 array, normalized [-1.0, 1.0]
    sample_rate: int
    is_speech: bool = False

@dataclass
class Utterance:
    """A complete thought spoken by the user."""
    audio_data: np.ndarray
    transcript: str = ""
    duration_ms: int = 0
```

---

## 4. Component Interface Specifications (The Harness)

To allow swapping models in the future (e.g., swapping Moonshine for Whisper), every model must conform to a strict interface.

### 4.1 Audio Processing Interfaces

**`audio/preprocessor.py`**
```python
class AudioPreprocessor:
    def __init__(self, sample_rate: int):
        pass
        
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Contract:
        1. Input: int16 or float32.
        2. Output: float32, normalized, DC offset removed, high-passed at 80Hz.
        3. Must execute in < 2ms.
        """
        pass
```

### 4.2 ML Model Interfaces

**`models/vad.py`**
```python
class VoiceActivityDetector:
    def __init__(self, config: VADConfig):
        pass
        
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Contract:
        1. Input: float32 numpy array, exactly 32ms length.
        2. Output: boolean based on config.threshold.
        3. Internal state (KV cache) must be maintained for streaming accuracy.
        """
        pass
```

**`models/stt.py`**
```python
class SpeechToTextProvider(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Contract:
        1. Input: Variable length float32 array (the complete utterance).
        2. Output: String transcript.
        3. Must strip leading/trailing whitespace and hallucinations (e.g., "[silence]").
        """
        pass
```

**`models/tts.py`**
```python
from typing import AsyncGenerator

class TextToSpeechProvider(Protocol):
    async def synthesize_stream(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[np.ndarray, None]:
        """
        Contract:
        1. Input: An async generator yielding tokens/words from the LLM.
        2. Output: An async generator yielding float32 audio chunks (24kHz).
        3. Must handle sentence-boundary buffering internally (e.g., don't synthesize mid-word).
        """
        pass
```

---

## 5. The Core Orchestrator (State & Threading)

This is the most complex part of the system. It handles threading, queues, and the State Machine.

### 5.1 Concurrency Architecture

We use **three threads/loops**:
1.  **Input Thread:** Blocking PortAudio read loop. Puts raw bytes into `input_queue`.
2.  **Output Thread:** Blocking PortAudio write loop. Reads from `output_queue`.
3.  **Main Async Event Loop:** Runs the orchestrator. Uses `asyncio.Queue` wrappers around the threaded queues to remain non-blocking.

### 5.2 The State Machine (Orchestrator)

**`core/orchestrator.py`**

```python
import asyncio
from enum import Enum, auto

class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()

class PipelineOrchestrator:
    def __init__(self, config: AppConfig):
        self.state = PipelineState.IDLE
        self.current_task: asyncio.Task | None = None
        
        # Thread-safe queues bridged to asyncio
        self.mic_queue = asyncio.Queue()
        self.speaker_queue = asyncio.Queue()
        
        # Audio buffer for the current utterance
        self.utterance_buffer = []

    async def run_loop(self):
        """The main hot loop."""
        while True:
            chunk = await self.mic_queue.get()
            clean_chunk = self.preprocessor.process_chunk(chunk)
            is_speech = self.vad.is_speech(clean_chunk)
            
            await self._handle_state_transition(clean_chunk, is_speech)

    async def _handle_state_transition(self, chunk: np.ndarray, is_speech: bool):
        if self.state == PipelineState.IDLE:
            if is_speech:
                self.state = PipelineState.LISTENING
                self.utterance_buffer.append(chunk)
                
        elif self.state == PipelineState.LISTENING:
            self.utterance_buffer.append(chunk)
            if self._check_silence_timeout(is_speech):
                # Transition to THINKING
                audio_data = np.concatenate(self.utterance_buffer)
                self.utterance_buffer.clear()
                self.state = PipelineState.THINKING
                
                # Fire off the heavy lifting in the background
                self.current_task = asyncio.create_task(self._process_utterance(audio_data))

        elif self.state == PipelineState.SPEAKING:
            if is_speech:
                # BARGE-IN TRIGGERED
                self._handle_barge_in()
                self.state = PipelineState.LISTENING
                self.utterance_buffer.append(chunk)

    def _handle_barge_in(self):
        """Interrupts current playback and inference."""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        
        # Flush the speaker queue to stop audio immediately
        while not self.speaker_queue.empty():
            self.speaker_queue.get_nowait()
            
    async def _process_utterance(self, audio: np.ndarray):
        """The STT -> LLM -> TTS chain."""
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe, audio)
            
            # Start LLM stream
            token_stream = self.llm.generate_stream(transcript)
            
            # Pipe tokens to TTS, pipe audio to Speaker
            self.state = PipelineState.SPEAKING
            async for audio_chunk in self.tts.synthesize_stream(token_stream):
                # Post-process (LUFS normalize)
                polished_chunk = self.postprocessor.process(audio_chunk)
                await self.speaker_queue.put(polished_chunk)
                
            self.state = PipelineState.IDLE
            
        except asyncio.CancelledError:
            # Task was cancelled due to barge-in, exit cleanly
            pass
```

---

## 6. Testing Harness & Verification

How to verify each component works in isolation before plugging them together.

### 6.1 Testing VAD (The Gatekeeper Test)
Create `tests/test_vad.py`. Feed it a known audio file with exactly 2 seconds of silence, 2 seconds of speech, 2 seconds of silence. Assert that `is_speech()` returns `True` exactly between frames corresponding to 2s and 4s.

### 6.2 Testing STT (The Ear Test)
Create `tests/test_stt.py`. Pass a clean, pre-recorded `float32` numpy array of a human saying "Hello world." Assert the output string `== "hello world"`. (Strip punctuation and lowercase before asserting).

### 6.3 Testing Latency (The Critical Benchmark)
Add timing hooks in `PipelineOrchestrator`:
1. `T0`: Moment `_check_silence_timeout` returns True.
2. `T1`: Moment `stt.transcribe` returns.
3. `T2`: Moment `llm.generate_stream` yields the first token.
4. `T3`: Moment `tts.synthesize_stream` yields the first audio chunk.

**Target Metrics on local hardware:**
*   `T1 - T0` (STT Latency): < 300ms
*   `T2 - T1` (LLM TTFT): < 200ms
*   `T3 - T2` (TTS TTFA): < 150ms
*   **Total system latency:** < 650ms.

---

## 7. Error Handling & Edge Cases

1. **Audio Device Disconnect:** If PortAudio throws an `IOError` (e.g., Bluetooth mic dies), the Input Thread must catch it, log an error, and attempt to reinitialize the default input device every 5 seconds.
2. **Buffer Overflows:** If the async orchestrator loop falls behind, the `input_queue` will fill up. Use a bounded queue (e.g., `maxsize=50`). If it's full, drop the oldest chunks (`queue.get_nowait()`) to stay real-time. Dropping frames is better than creeping latency.
3. **Empty Transcripts:** Sometimes breathing triggers VAD, but STT returns `""` or `" "`. The orchestrator must check `if not transcript.strip(): return to IDLE` before hitting the LLM.
4. **LLM Hallucinations:** Local LLMs sometimes generate markdown or actions like `*smiles*`. Add a regex filter before sending text to the TTS engine to strip text enclosed in asterisks.

---

> [!NOTE]
> This TDD defines strict contracts. If you want to swap Kokoro TTS for a different engine later, you only need to ensure the new class conforms to the `TextToSpeechProvider` Protocol (taking an async generator of strings, yielding an async generator of numpy arrays). The rest of the pipeline will not need to change.
