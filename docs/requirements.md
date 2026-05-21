# Technical Requirements & Specification

This document details the exact requirements, execution SLA, and system architecture for the ultra-low latency, CPU-bound structured data extraction pipeline in Rust.

---

## 1. System Architecture & Layers

The data extraction funnel operates across three layers to balance maximum speed with perfect computational logic accuracy.

```
                    [ Unstructured Text Input ]
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ Layer 1: GLiNER (Zero-Shot Token-span)       │  < 15ms SLA
        └──────────────────────┬───────────────────────┘
                               │ (If variables missing / unparsable)
                               ▼
        ┌──────────────────────────────────────────────┐
        │ Layer 2: Flan-T5 (Sequence-to-Sequence Translation) │  < 60ms SLA
        └──────────────────────┬───────────────────────┘
                               │ (If parser fails / format mismatch)
                               ▼
        ┌──────────────────────────────────────────────┐
        │ Layer 3: Qwen 3.5 0.8B (GGUF Function Call)  │  < 1100ms SLA
        └──────────────────────┬───────────────────────┘
                               │ (If LLM fails / syntax error)
                               ▼
            [ Failsafe Dead-Letter payload ]
```

### Layer Detail Table

| Layer | Model / Technology | Primary Purpose | Latency Target | Trigger / Routing Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | **GLiNER-Medium (ONNX)** | Token-span matching based on zero-shot semantic labels. | **5 - 15ms** | Clean, concise inputs containing explicit entities (e.g. `"product a bought for 100 plus 18% tax"`). |
| **Layer 2** | **Flan-T5-Base (ONNX)** | Text-to-key-values mapping via small token sequence translation. | **30 - 60ms** | Semantic variations where mathematical formulas or keywords vary but match structured forms. |
| **Layer 3** | **Qwen 3.5 0.8B (GGUF)** | Autoregressive JSON extraction constrained by grammar. | **600 - 1100ms** | Messy, multi-sentence conversational streams with implicit variables or complex contexts. |

---

## 2. Technical Stack & Crate Layout

- **Language**: Rust (Edition 2021)
- **Serialization**: `serde` and `serde_json` for type-safe validation.
- **ONNX Inference**: `ort` crate (v2.0.0-rc.12+) wrapping native ONNX Runtime 1.24+ for Layers 1 & 2.
- **GGML / GGUF Inference**: `llama-cpp-2` crate (v0.1.11+) wrapping `llama.cpp` for Layer 3.
- **Tokenization**: `tokenizers` crate (v0.15+) for Hugging Face tokenizer support.

### Dependency Table (`Cargo.toml`)
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
ort = { version = "2.0.0-rc.12", features = ["ndarray"] }
ndarray = "0.15"
tokenizers = "0.15"
llama-cpp-2 = { version = "0.1.11", features = ["sampler"] }
num_cpus = "1.16"
```

---

## 3. High Performance & Zero-Latency Allocation Rules

To ensure sub-millisecond dispatch overhead:
1. **Model Persistence**: Models and ONNX sessions must be instantiated exactly **once** at startup inside a shared thread-safe state wrapper (`std::sync::Arc`). Never load models within request worker threads or loop iterations.
2. **Strict Thread Control**: Since ONNX Runtime and `llama.cpp` spin up parallel worker threads, strict hardware core limits must be set in the host environment to prevent CPU context-switching decay:
   ```bash
   export ORT_INTRA_OP_NUM_THREADS=$(nproc --all)
   export OMP_NUM_THREADS=$(nproc --all)
   ```
   In the Rust code, set intra-threads using `.with_intra_threads(num_cpus::get_physical())`.
3. **Structured Sampling Constraint**: For Layer 3 (Qwen 3.5), a strict GBNF (GGML Backus-Naur Form) JSON grammar must be enforced on token selection, preventing hallucinated tags and reducing syntax validation overhead.

---

## 4. Models Export & Preparation

### Layer 1 & 2: ONNX Models Export
The ONNX models must be exported from Hugging Face weights using Hugging Face Optimum:
```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model urchade/gliner_medium-v2.1 models/gliner/
optimum-cli export onnx --model google/flan-t5-base models/flan/
```

### Layer 3: GGUF Model Setup
To support the newer Qwen 3.5 0.8B model, the underlying `llama.cpp` runtime must be up-to-date:
- **Model Target**: `Qwen/Qwen3.5-0.8B-Instruct-GGUF` (or standard `qwen3.5_0.8b_q4.gguf`).
- **Compatibility**: Supports GGUF serialization version 3+. Newer Qwen architectures utilize specific tokenizers (Qwen2Tokenizer) and tokenization parameters that require a modern `llama.cpp` core compiler.
- **Inference Mode**: SIMD/AVX-512 acceleration should be enabled during compilation of the C++ libraries inside the `llama-cpp-2` build script (configured automatically when compiling natively).
