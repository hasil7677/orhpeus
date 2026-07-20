import os
import re
from pathlib import Path
from typing import AsyncGenerator

import gpu_setup  # noqa: F401 — must run before onnxruntime is imported (below)

import numpy as np
from kokoro_onnx import Kokoro

from config import ModelConfig

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_ASTERISK_ACTIONS = re.compile(r"\*[^*]*\*")

MODELS_DIR = Path(__file__).parent.parent / "assets"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

_DOWNLOAD_HELP = (
    "Kokoro model files not found. Download them into local/assets/:\n"
    "  kokoro-v1.0.onnx  -> https://github.com/thewh1teagle/kokoro-onnx/releases\n"
    "  voices-v1.0.bin   -> https://github.com/thewh1teagle/kokoro-onnx/releases"
)


class TextToSpeechProvider:
    """Kokoro-onnx wrapper (GPU via CUDAExecutionProvider when available).

    Contract:
    1. Input: an async generator yielding tokens/words from the LLM.
    2. Output: an async generator yielding float32 audio chunks (24kHz).
    3. Buffers tokens internally and flushes on sentence boundaries so Kokoro
       never has to synthesize mid-sentence fragments.
    """

    def __init__(self, config: ModelConfig, sample_rate: int):
        if not KOKORO_MODEL_PATH.exists() or not KOKORO_VOICES_PATH.exists():
            raise FileNotFoundError(_DOWNLOAD_HELP)

        # kokoro-onnx picks up onnxruntime.get_available_providers() itself,
        # or an explicit ONNX_PROVIDER env var — no constructor kwarg for it.
        os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
        self.kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
        self.voice = config.tts_voice
        self.speed = config.tts_speed
        self.sample_rate = sample_rate

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[np.ndarray, None]:
        buffer = ""
        async for token in text_stream:
            buffer += token
            parts = _SENTENCE_BOUNDARY.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    async for chunk in self._synthesize_sentence(sentence):
                        yield chunk
                buffer = parts[-1]

        if buffer.strip():
            async for chunk in self._synthesize_sentence(buffer):
                yield chunk

    async def _synthesize_sentence(self, sentence: str) -> AsyncGenerator[np.ndarray, None]:
        clean = _ASTERISK_ACTIONS.sub("", sentence).strip()
        if not clean:
            return
        async for samples, _sr in self.kokoro.create_stream(
            clean, voice=self.voice, speed=self.speed, lang="en-us"
        ):
            yield samples.astype(np.float32)
