# orpheus

A real-time, voice-to-voice AI pipeline — speak to it, it talks back, with low enough latency to feel like a conversation.

Two builds, in progress:

- **[`local/`](local/)** — runs speech detection, transcription, and voice synthesis entirely on your own GPU (tested on a 4GB GTX 1650), with the LLM ("brain") outsourced to [Groq's](https://console.groq.com) free API. Privacy for the parts that matter, low latency, no local LLM hardware required.
- **cloud** (planned) — a fully API-based version (STT + LLM + TTS all cloud-hosted) for when you don't have a GPU at all.

## Demo

<video src="https://raw.githubusercontent.com/hasil7677/orhpeus/main/demo.mp4" controls width="100%"></video>

(If the player above doesn't render, [watch/download it directly](demo.mp4).)

## Stack (local build)

| Stage | Model | Where it runs |
|---|---|---|
| Voice activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) | CPU |
| Speech-to-text | [Moonshine](https://github.com/moonshine-ai/moonshine) (ONNX) | GPU |
| LLM | Groq-hosted (`openai/gpt-oss-20b`) | Cloud |
| Text-to-speech | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (82M, ONNX) | GPU |

See [`local/README.md`](local/README.md) for setup and running it.

## License

MIT — see [LICENSE](LICENSE).
