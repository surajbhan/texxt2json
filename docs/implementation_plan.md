# Implementation Plan - Fully Generalized Schema-Driven structured JSON Extraction Library

This plan outlines the architecture to generalize the library completely, removing all hardcoded key assumptions (like `"product"`, `"paid"`, `"tax"`, etc.). The updated library will accept an arbitrary natural language input alongside a **user-defined schema**, dynamically routing the extraction through a schema-aligned multi-layer cascade.

---

## User Review Required

> [!IMPORTANT]
> **API Signature Changes**
> - The static types (`ExtractedPayload`, `RawExtraction`) will be replaced by a dynamic **Schema-driven API**.
> - The primary entry point of `MultiLayerParser` will change from:
>   `pub fn process_text(&self, input: &str) -> ExtractedPayload`
>   to:
>   `pub fn process_text(&self, input: &str, schema: &ExtractionSchema) -> serde_json::Value`
> - Layer 3 (Qwen GGUF) will dynamically generate its Backus-Naur Form (GBNF) grammar string from the `ExtractionSchema` on the fly to constrain LLM token generation to the exact desired JSON layout.
> - Layer 4 (Resilient Fallback) will be replaced by a **Generalized Proximity-Based Association Heuristic** that parses quoted/unquoted strings, numbers, and booleans, assigning them to fields by minimizing the token distance between candidate values and field name identifiers.

---

## Proposed Changes

### Component: Generalized Schema Definitions

#### [NEW] [schema.rs](file:///data/work/text2json/src/schema.rs)
We will introduce a formal schema representation module with case-insensitive / alias-supported deserialization:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    #[serde(alias = "string", alias = "String")]
    String,
    #[serde(alias = "integer", alias = "Integer", alias = "int", alias = "Int")]
    Integer,
    #[serde(alias = "float", alias = "Float", alias = "number", alias = "Number")]
    Float,
    #[serde(alias = "boolean", alias = "Boolean", alias = "bool", alias = "Bool")]
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub field_type: FieldType,
    pub description: String,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSchema {
    pub fields: Vec<FieldSchema>,
}
```

---

### Component: Core MultiLayerParser Generalization

#### [MODIFY] [lib.rs](file:///data/work/text2json/src/lib.rs)
- Remove `ExtractedPayload` and `RawExtraction`.
- Expose `ExtractionSchema`, `FieldSchema`, and `FieldType` (re-exporting from `schema.rs`).
- Update `process_text` signature to accept `&ExtractionSchema` and return `serde_json::Value` (which will contain the extracted key-value pairs).
- Re-engineer **Layer 4 (Resilient Heuristic Fallback)** into a generalized proximity parser:
  1. Parse all text enclosed in single or double quotes, and support unquoted strings using landmarks when the field name suggests a product/item.
  2. Parse all numbers along with their character positions in the text.
  3. Scan the text for the lowercase representation of each field name (or synonyms like price/tax/product).
  4. Associate each quote or number with the field that minimizes the token distance to its keyword indicator, applying high-fidelity bonuses for matching percentages to tax fields and avoiding distractor words for prices.

---

### Component: Dynamic GBNF Grammar Generation

#### [MODIFY] [layers/qwen.rs](file:///data/work/text2json/src/layers/qwen.rs)
- Implement a GBNF grammar generator function:
  `pub fn generate_gbnf(schema: &ExtractionSchema) -> String`
- This function will construct a custom grammar on the fly. For a schema containing `{"name": String, "age": Integer}`, it will generate a grammar forcing exactly:
  ```
  root ::= "{\n" "  \"name\": " string ",\n" "  \"age\": " integer "\n" "}"
  string ::= "\"" [^\"]* "\""
  integer ::= "-"? [0-9]+
  ```
- Pass this dynamic grammar to the `llama-cpp-2` sampler using `LlamaSampler::grammar(&grammar_str, "root")`.

---

### Component: Evaluation Harness Upgrade

#### [MODIFY] [eval.rs](file:///data/work/text2json/src/bin/eval.rs)
- Modify the evaluation loader to support arbitrary schemas.
- Update `test_cases.json` and `test_cases_large.json` to include the schema description inside each test item:
  ```json
  {
    "input": "...",
    "schema": {
      "fields": [
        { "name": "product", "field_type": "String", "description": "product name", "required": true },
        { "name": "price", "field_type": "Float", "description": "base price", "required": true },
        { "name": "tax", "field_type": "Float", "description": "tax rate percentage", "required": true }
      ]
    },
    "expected": {
      "product": "sriracha",
      "price": 1.48,
      "tax": 8.5
    },
    "expected_paid": 1.6058
  }
  ```
- Change `eval.rs` to validate the dynamic JSON outputs (`serde_json::Value`) against the `expected` objects. For floating point values, compare them with a tolerance of `1e-4`.

---

## Verification Plan

### Automated Tests
1. **Rust Library Validation**:
   ```bash
   cargo check
   ```
2. **Execute Evaluation Runner**:
   ```bash
   cargo run --bin eval data/test_cases_large.json
   ```

### Manual Verification
- Review the generated GBNF grammar formats for diverse schemas.
- Confirm that the proximity fallback correctly pairs fields in multi-variable sentences.
