# Architecture Notes: Sub-300ms Local Voice AI Pipeline

## 1. System Overview & Latency Targets
To achieve human-like conversational latency (sub-300ms), the system abandons traditional "batch" processing (wait to finish speaking -> transcribe -> generate text -> generate audio). Instead, it uses a **fully streaming, overlapping pipeline**.

**Latency Budget Breakdown:**
1. **Audio Ingestion & VAD:** ~10ms
2. **Streaming STT (Final Transcript):** ~100-150ms
3. **LLM Time-To-First-Token (TTFT) & Buffer:** ~100ms
4. **TTS Time-To-First-Audio (TTFA):** ~50-80ms
**Total System Latency:** ~260ms - 340ms

---

## 2. Hardware Strategy: The 24GB VRAM Blueprint
For real-time local inference, **VRAM capacity and Memory Bandwidth** dictate performance. Splitting models across multiple GPUs introduces PCIe bottleneck latency. 

* **The Goal:** Fit the entire pipeline (STT, LLM, TTS) into a single contiguous block of VRAM.
* **The Hardware:** A single **Nvidia RTX 3090 (24GB VRAM)**. 
* **Why 24GB?** 
  * 16GB VRAM restricts you to an 8B parameter LLM.
  * 24GB VRAM allows you to load a heavily quantized 14B–32B model (e.g., Qwen 2.5 32B), achieving near-GPT-4 reasoning while leaving enough VRAM for the STT and TTS engines.
* **System RAM:** 64GB DDR5 (Allows offloading of massive models via `llama.cpp` for non-real-time tasks).

---

## 3. Pipeline Component Breakdown

### A. The Ingestion Layer (VAD & DSP)
You cannot send raw microphone audio directly to the STT. It must be pre-processed on the fly.
* **Audio Format:** Stream audio via WebRTC. Resample on the fly to `16kHz, 16-bit Mono PCM`.
* **DSP (Digital Signal Processing):** Apply lightweight Acoustic Echo Cancellation (AEC) and Noise Suppression via standard WebRTC libraries. Avoid neural denoisers (too much latency).
* **VAD (Voice Activity Detection):** Run **Silero VAD**. It detects exactly when speech starts and stops.
  * *Trigger:* When Silero detects 300ms of silence, it triggers the `speech_end` event, forcing the STT to lock in its final transcript and fire it to the LLM.

### B. The Ear: Moonshine STT (Streaming Transcription)
Traditional STTs like Whisper use "full attention" over 30-second windows, making them useless for real-time. 
* **Model:** **Moonshine Tiny (27M)** or **Moonshine Base**.
* **Architecture:** Uses *Sliding Window Attention*. It processes raw audio waveforms in tiny chunks and caches the mathematical state of its encoder. 
* **Behavior:** It outputs partial transcripts while the user is talking. When Silero triggers `speech_end`, Moonshine instantly outputs the final transcript without needing to recompute the entire sentence.

### C. The Brain: Local LLM & The Sentence Buffer
The LLM must generate text fast enough to feed the TTS engine without starving it.
* **Model:** Llama 3 (8B) or Qwen 2.5 (14B-32B). 
* **Inference Engine:** **vLLM**. It uses PagedAttention to maximize GPU memory bandwidth and token generation speed.
* **The "Sentence Buffer" Hack:** 
  Do not wait for the LLM to finish its response. Write a buffer script that intercepts the token stream. The moment the buffer detects a sentence boundary (e.g., `.`, `!`, `?` followed by a space), it flushes that single sentence directly to the TTS engine. The LLM continues generating sentence #2 while sentence #1 is being spoken.

### D. The Voice: Kokoro-82M TTS
The TTS engine must synthesize audio faster than real-time (Real-Time Factor < 1.0).
* **Model:** **Kokoro-82M**.
* **Architecture:** 82 million parameters. Uses a decoder-only architecture based on StyleTTS 2 and ISTFTNet (no heavy diffusion or encoder stack).
* **Performance:** Takes up <400MB of VRAM (or can run purely on CPU). It achieves an RTF of ~0.45, meaning it generates 1 second of audio in less than 0.5 seconds.
* **Output:** Streams 24kHz audio chunks back to the WebRTC client instantly.

---

## 4. Orchestration & The Interruption Protocol (Barge-in)
Wiring WebRTC threads, audio buffers, and model handoffs from scratch is incredibly difficult. 
* **Framework:** Use **Pipecat** or **LiveKit Agents**. These Python frameworks handle the networking, packet loss, and microservice orchestration.
* **The Barge-in Protocol:** To make the AI feel human, the user must be able to interrupt it.
  1. The microphone remains "hot" while the TTS is playing.
  2. If Silero VAD detects user speech (`speech_start`), the orchestrator fires an interrupt signal.
  3. **Action 1:** Immediately flush the TTS audio playback buffer (stop the voice).
  4. **Action 2:** Send a stop token to vLLM.
  5. **Action 3:** Append the interrupted text to the LLM context (e.g., `[System: User interrupted the response at "I think that..."]`) so the AI retains conversational memory of where it was cut off.


  graph TD
    %% Styling
    classDef client fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef layer fill:#cce5ff,stroke:#004085,stroke-width:2px;
    classDef model fill:#fff3cd,stroke:#856404,stroke-width:2px;
    classDef logic fill:#f8d7da,stroke:#721c24,stroke-width:2px;

    %% Client Ingestion
    subgraph Client_Side ["Client Side (User Device)"]
        A[User Microphone]:::client -->|Raw Audio Stream| B[WebRTC / WebSockets]:::client
        L[User Audio Output]:::client
    end

    %% Audio Processing & VAD Layer
    subgraph Ingestion_Layer ["1. Ingestion & DSP Layer (~10ms)"]
        B -->|Audio Chunks| C[Lightweight WebRTC DSP]:::layer
        C -->|Resample to 16kHz Mono PCM| D[Silero VAD]:::model
    end

    %% STT Layer
    subgraph STT_Layer ["2. Streaming STT (~100-150ms)"]
        D -->|If User Speaking| E[Moonshine STT Tiny 27M]:::model
        E -->|Partial Transcripts| F[Stream Cache]:::layer
        D -->|Silence Detected: speech_end| G[Lock Final Transcript]:::layer
    end

    %% LLM Layer
    subgraph LLM_Layer ["3. Brain / Inference Layer (~100ms TTFT)"]
        G -->|Send Full Prompt| H[vLLM Engine]:::layer
        H -->|Load Model: Qwen 2.5 32B INT4| I[LLM Token Generation]:::model
    end

    %% Streaming Buffer Logic
    subgraph Buffer_Layer ["4. Orchestration & Buffer Logic"]
        I -->|Streaming Tokens| J[Sentence Aggregator Buffer]:::logic
        J -->|Detect . ! ?| K[Flush Complete Sentence]:::logic
        
        %% Interruption / Barge-in Logic
        D -.->|New Speech Detected during AI Playback| M[Barge-In Protocol]:::logic
        M -.->|1. Kill Stream| L
        M -.->|2. Stop Generation| I
        M -.->|3. Inject Context| H
    end

    %% TTS Layer
    subgraph TTS_Layer ["5. Streaming TTS (~50-80ms TTFA)"]
        K --> N[Kokoro-82M TTS]:::model
        N -->|Generate Audio Chunks RTF 0.45| O[Stream Back to Client]:::layer
        O --> B
        B -->|Playback Audio| L
    end

    %% Class Applications
    class A,B,L client;
    class C,G,F,H,O layer;
    class D,E,I,N model;
    class J,K,M logic;
