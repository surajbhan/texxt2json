# Task List - Fully Generalized Schema-Driven structured JSON Extraction Library

- [x] Define dynamic schema structures (`ExtractionSchema`, `FieldSchema`, `FieldType`) in `src/schema.rs`
- [x] Refactor `src/lib.rs` to utilize `ExtractionSchema` and implement the Generalized Proximity-Based Heuristic Fallback
- [x] Implement on-the-fly GBNF JSON grammar generation and Sampler binding in `src/layers/qwen.rs`
- [x] Update `src/layers/gliner.rs` and `src/layers/flan.rs` signatures and adapt their post-processing to use `ExtractionSchema`
- [x] Re-generate the 50-case and 1,000-case datasets with embedded schema definitions and expected JSON values
- [x] Rewrite the evaluation harness `src/bin/eval.rs` to execute verification on dynamic schema-driven payloads
- [x] Validate compilation with `cargo check` and execute full evaluation suite with `cargo run --bin eval`

