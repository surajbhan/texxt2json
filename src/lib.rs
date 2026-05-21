#[cfg(feature = "native")]
pub mod layers;
pub mod schema;

pub use schema::{ExtractionSchema, FieldSchema, FieldType};
#[cfg(feature = "native")]
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExtractionLayer {
    Gliner,
    Flan,
    Qwen,
    ProximityHeuristic,
}

#[cfg(feature = "native")]
pub struct MultiLayerParser {
    gliner_layer: Option<layers::gliner::GlinerLayer>,
    flan_layer: Option<layers::flan::FlanLayer>,
    qwen_layer: Option<layers::qwen::QwenLayer>,
    layer_order: Vec<ExtractionLayer>,
}

#[cfg(feature = "native")]
impl MultiLayerParser {
    pub fn new(gliner_path: &str, flan_path: &str, qwen_path: &str) -> Result<Self> {
        let gliner_layer = if !gliner_path.is_empty() && std::path::Path::new(gliner_path).exists() {
            println!("[Pipeline] Loading Layer 1: GLiNER (ONNX) from {}", gliner_path);
            Some(layers::gliner::GlinerLayer::new(gliner_path)?)
        } else {
            println!("[Pipeline] Bypassing Layer 1: GLiNER model file not found.");
            None
        };

        let flan_layer = if !flan_path.is_empty() {
            let path = std::path::Path::new(flan_path);
            let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
            let exists = path.exists() 
                || parent.join("flan_t5_encoder.onnx").exists()
                || parent.join("flan_encoder.onnx").exists();
                
            if exists {
                println!("[Pipeline] Loading Layer 2: Flan-T5 (ONNX) from {}", flan_path);
                Some(layers::flan::FlanLayer::new(flan_path)?)
            } else {
                println!("[Pipeline] Bypassing Layer 2: Flan-T5 model file not found.");
                None
            }
        } else {
            None
        };

        let qwen_layer = if !qwen_path.is_empty() && std::path::Path::new(qwen_path).exists() {
            println!("[Pipeline] Loading Layer 3: Qwen 3.5 (GGUF) from {}", qwen_path);
            Some(layers::qwen::QwenLayer::new(qwen_path)?)
        } else {
            println!("[Pipeline] Bypassing Layer 3: Qwen GGUF model file not found.");
            None
        };

        Ok(Self {
            gliner_layer,
            flan_layer,
            qwen_layer,
            layer_order: vec![
                ExtractionLayer::Gliner,
                ExtractionLayer::Flan,
                ExtractionLayer::Qwen,
                ExtractionLayer::ProximityHeuristic,
            ],
        })
    }

    pub fn default_empty() -> Self {
        Self {
            gliner_layer: None,
            flan_layer: None,
            qwen_layer: None,
            layer_order: vec![
                ExtractionLayer::Gliner,
                ExtractionLayer::Flan,
                ExtractionLayer::Qwen,
                ExtractionLayer::ProximityHeuristic,
            ],
        }
    }

    pub fn with_layer_order(mut self, order: Vec<ExtractionLayer>) -> Self {
        self.layer_order = order;
        self
    }

    /// Primary execution routing mechanism using custom schemas returning metadata
    pub fn process_text_with_metadata(&self, input: &str, schema: &ExtractionSchema) -> (serde_json::Value, ExtractionLayer) {
        println!("[Pipeline] Sourcing message: \"{}\"", input);

        for layer in &self.layer_order {
            match layer {
                ExtractionLayer::Gliner => {
                    if let Some(ref gliner) = self.gliner_layer {
                        match gliner.extract(input, schema) {
                            Ok(Some(raw)) => {
                                println!("[Success] Executed via Layer 1 (GLiNER) ~10ms");
                                return (raw, ExtractionLayer::Gliner);
                            }
                            Ok(None) => {}
                            Err(e) => {
                                eprintln!("[Layer 1 Warning] Extraction error: {}", e);
                            }
                        }
                        println!("[Fallback] Layer 1 missing variables or unparsable.");
                    }
                }
                ExtractionLayer::Flan => {
                    if let Some(ref flan) = self.flan_layer {
                        match flan.translate(input, schema) {
                            Ok(Some(raw)) => {
                                println!("[Success] Executed via Layer 2 (Flan-T5) ~45ms");
                                return (raw, ExtractionLayer::Flan);
                            }
                            Ok(None) => {}
                            Err(e) => {
                                eprintln!("[Layer 2 Warning] Translation error: {}", e);
                            }
                        }
                        println!("[Fallback] Layer 2 failed execution sequence.");
                    }
                }
                ExtractionLayer::Qwen => {
                    if let Some(ref qwen) = self.qwen_layer {
                        match qwen.extract(input, schema) {
                            Ok(payload) => {
                                println!("[Success] Handled by Layer 3 Failsafe (Qwen-0.8B) ~800ms");
                                return (payload, ExtractionLayer::Qwen);
                            }
                            Err(e) => {
                                eprintln!("[Layer 3 Warning] Failsafe generation failed: {}", e);
                            }
                        }
                    }
                }
                ExtractionLayer::ProximityHeuristic => {
                    println!("[Resiliency] Activating local high-fidelity entity segmenter.");
                    return (extract_via_proximity_heuristic(input, schema), ExtractionLayer::ProximityHeuristic);
                }
            }
        }

        println!("[Resiliency] Activating fallback local high-fidelity entity segmenter.");
        (extract_via_proximity_heuristic(input, schema), ExtractionLayer::ProximityHeuristic)
    }

    /// Backward compatible primary execution routing mechanism
    pub fn process_text(&self, input: &str, schema: &ExtractionSchema) -> serde_json::Value {
        self.process_text_with_metadata(input, schema).0
    }
}

/// Top-level convenience function that takes an input string and a schema,
/// and produces the extracted JSON payload.
pub fn extract_json(input: &str, schema: &ExtractionSchema) -> serde_json::Value {
    #[cfg(feature = "native")]
    {
        let parser = MultiLayerParser::new(
            "models/gliner_medium.onnx",
            "models/flan_t5_encoder.onnx",
            "models/qwen3.5_0.8b_q4.gguf"
        ).unwrap_or_else(|_| MultiLayerParser::default_empty());
        parser.process_text(input, schema)
    }
    #[cfg(not(feature = "native"))]
    {
        extract_via_proximity_heuristic(input, schema)
    }
}

// --- WebAssembly Bindings ---
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn extract_json_wasm(input: &str, schema_json_str: &str) -> String {
    let schema: ExtractionSchema = match serde_json::from_str(schema_json_str) {
        Ok(s) => s,
        Err(e) => return format!("{{\"error\": \"Invalid Schema JSON: {}\"}}", e),
    };
    let result = extract_via_proximity_heuristic(input, &schema);
    serde_json::to_string_pretty(&result).unwrap_or_else(|_| "{}".to_string())
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "for", "with", "by", "on", "at", 
    "from", "being", "value", "field", "type", "description", "required", "schema", "json", 
    "output", "having", "which", "that", "this", "here", "there", "some", "any", "it", "its", 
    "called", "named", "name", "has", "have", "had", "was", "were", "been", "be", "do", 
    "does", "did", "doing", "but", "as", "if", "so", "than", "out", "only", "about"
];

struct TextWord<'a> {
    word: &'a str,
    start_pos: usize,
}

#[derive(Debug, Clone)]
struct NumericCandidate {
    value: f64,
    start_pos: usize,
    end_pos: usize,
    has_percent: bool,
    is_integer: bool,
}

#[derive(Debug, Clone)]
struct BoolCandidate {
    value: bool,
    start_pos: usize,
}

#[derive(Debug, Clone)]
struct StringCandidate {
    value: String,
    start_pos: usize,
}

pub fn extract_via_proximity_heuristic(input: &str, schema: &ExtractionSchema) -> serde_json::Value {
    let lower = input.to_lowercase();
    
    // Step 1: Scan and segment text into words with precise character offsets
    let mut words = Vec::new();
    let mut in_word = false;
    let mut word_start = 0;
    for (i, c) in lower.char_indices() {
        if c.is_whitespace() {
            if in_word {
                words.push(TextWord {
                    word: &lower[word_start..i],
                    start_pos: word_start,
                });
                in_word = false;
            }
        } else {
            if !in_word {
                word_start = i;
                in_word = true;
            }
        }
    }
    if in_word {
        words.push(TextWord {
            word: &lower[word_start..],
            start_pos: word_start,
        });
    }

    // Step 2: Extract Anchor Positions per Field in Schema
    let mut field_anchors = Vec::new();
    for field in &schema.fields {
        let mut anchors = Vec::new();
        
        let lower_name = field.name.to_lowercase();
        // Match exact field name substring
        let mut start = 0;
        while let Some(pos) = lower[start..].find(&lower_name) {
            let actual_pos = start + pos;
            anchors.push(actual_pos);
            start = actual_pos + 1.max(lower_name.len());
        }
        
        // Split name by word delimiters and match non-stopword components
        let name_parts: Vec<&str> = lower_name.split(|c: char| c == '_' || c == '-' || c.is_whitespace()).collect();
        for part in name_parts {
            let cleaned = part.trim();
            if cleaned.len() > 1 && !STOPWORDS.contains(&cleaned) {
                let mut start = 0;
                while let Some(pos) = lower[start..].find(cleaned) {
                    let actual_pos = start + pos;
                    if !anchors.contains(&actual_pos) {
                        anchors.push(actual_pos);
                    }
                    start = actual_pos + 1.max(cleaned.len());
                }
            }
        }
        
        // Split description by non-alphanumeric and match non-stopword tokens
        let desc_words: Vec<&str> = field.description.split(|c: char| !c.is_alphanumeric()).collect();
        for word in desc_words {
            let w = word.trim().to_lowercase();
            if w.len() > 2 && !STOPWORDS.contains(&w.as_str()) {
                let mut start = 0;
                while let Some(pos) = lower[start..].find(&w) {
                    let actual_pos = start + pos;
                    if !anchors.contains(&actual_pos) {
                        anchors.push(actual_pos);
                    }
                    start = actual_pos + 1.max(w.len());
                }
            }
        }
        
        field_anchors.push(anchors);
    }

    // Step 3: Extract Numeric Candidates
    let mut numeric_candidates = Vec::new();
    for (idx, tw) in words.iter().enumerate() {
        let word = tw.word;
        if word.contains('-') || word.contains('/') {
            continue;
        }
        
        let mut has_percent = word.contains('%');
        if !has_percent && idx + 1 < words.len() {
            let next_word = words[idx + 1].word;
            if next_word.starts_with("percent") || next_word.starts_with('%') {
                has_percent = true;
            }
        }
        
        let mut cleaned = String::new();
        let mut dot_count = 0;
        for c in word.chars() {
            if c.is_numeric() {
                cleaned.push(c);
            } else if c == '.' {
                if dot_count == 0 {
                    cleaned.push(c);
                    dot_count += 1;
                } else {
                    break;
                }
            } else if c == '-' && cleaned.is_empty() {
                cleaned.push(c);
            }
        }
        
        if cleaned.is_empty() || cleaned == "." || cleaned == "-" {
            continue;
        }
        if cleaned.ends_with('.') {
            cleaned.pop();
        }
        
        if let Ok(val) = cleaned.parse::<f64>() {
            let is_integer = !cleaned.contains('.');
            numeric_candidates.push(NumericCandidate {
                value: val,
                start_pos: tw.start_pos,
                end_pos: tw.start_pos + word.len(),
                has_percent,
                is_integer,
            });
        }
    }

    // Step 4: Extract Boolean Candidates
    let mut bool_candidates = Vec::new();
    let mut start = 0;
    while let Some(pos) = lower[start..].find("true") {
        let actual_pos = start + pos;
        bool_candidates.push(BoolCandidate { value: true, start_pos: actual_pos });
        start = actual_pos + 4;
    }
    start = 0;
    while let Some(pos) = lower[start..].find("false") {
        let actual_pos = start + pos;
        bool_candidates.push(BoolCandidate { value: false, start_pos: actual_pos });
        start = actual_pos + 5;
    }
    start = 0;
    while let Some(pos) = lower[start..].find("yes") {
        let actual_pos = start + pos;
        bool_candidates.push(BoolCandidate { value: true, start_pos: actual_pos });
        start = actual_pos + 3;
    }
    start = 0;
    while let Some(pos) = lower[start..].find("no") {
        let actual_pos = start + pos;
        let is_word_bound = (actual_pos == 0 || !lower.chars().nth(actual_pos - 1).unwrap().is_alphanumeric()) &&
                            (actual_pos + 2 >= lower.len() || !lower.chars().nth(actual_pos + 2).unwrap().is_alphanumeric());
        if is_word_bound {
            bool_candidates.push(BoolCandidate { value: false, start_pos: actual_pos });
        }
        start = actual_pos + 2;
    }

    // Step 5: Extract String Candidates (Quoted substrings)
    let mut string_candidates = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let quote_types = ['\'', '"', '`'];
    for &qt in &quote_types {
        let mut start_idx = None;
        for idx in 0..chars.len() {
            if chars[idx] == qt {
                if qt == '\'' && idx > 0 && idx + 1 < chars.len() 
                   && chars[idx - 1].is_alphabetic() && chars[idx + 1].is_alphabetic() {
                    continue;
                }
                if let Some(start) = start_idx {
                    let content: String = chars[start + 1..idx].iter().collect();
                    string_candidates.push(StringCandidate {
                        value: content,
                        start_pos: start,
                    });
                    start_idx = None;
                } else {
                    start_idx = Some(idx);
                }
            }
        }
    }

    // Fallback: If no quoted strings are found, treat all non-numeric, non-boolean words as string candidates
    if string_candidates.is_empty() {
        for tw in &words {
            let is_numeric = numeric_candidates.iter().any(|nc| tw.start_pos >= nc.start_pos && tw.start_pos < nc.end_pos);
            let is_bool = tw.word == "true" || tw.word == "false" || tw.word == "yes" || tw.word == "no";
            if !is_numeric && !is_bool && tw.word.len() > 1 {
                string_candidates.push(StringCandidate {
                    value: tw.word.to_string(),
                    start_pos: tw.start_pos,
                });
            }
        }
    }

    // Step 6: Greedy Numeric Field Assignment
    struct ScoreTuple {
        field_idx: usize,
        cand_idx: usize,
        score: f64,
    }
    let mut numeric_scores = Vec::new();
    for (f_idx, field) in schema.fields.iter().enumerate() {
        if field.field_type != FieldType::Integer && field.field_type != FieldType::Float {
            continue;
        }
        
        let field_lower = field.name.to_lowercase() + " " + &field.description.to_lowercase();
        let field_is_percent = field_lower.contains("percent") || field_lower.contains("percentage") || field_lower.contains('%');
        let anchors = &field_anchors[f_idx];
        
        for (c_idx, cand) in numeric_candidates.iter().enumerate() {
            let min_dist = if anchors.is_empty() {
                lower.len() as f64
            } else {
                let mut d = f64::MAX;
                for &a in anchors {
                    let dist = (cand.start_pos as isize - a as isize).abs() as f64;
                    if dist < d {
                        d = dist;
                    }
                }
                d
            };
            
            let mut score = min_dist;
            
            // Structural compatibility checks
            if field_is_percent {
                if cand.has_percent {
                    score -= 100.0;
                } else {
                    score += 1000.0;
                }
            } else {
                if cand.has_percent {
                    score += 2000.0;
                }
            }
            
            if field.field_type == FieldType::Integer {
                if cand.is_integer {
                    score -= 20.0;
                } else {
                    score += 200.0;
                }
            } else if field.field_type == FieldType::Float {
                if !cand.is_integer {
                    score -= 20.0;
                }
            }
            
            numeric_scores.push(ScoreTuple {
                field_idx: f_idx,
                cand_idx: c_idx,
                score,
            });
        }
    }

    numeric_scores.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    let mut assigned_numeric = std::collections::HashMap::new();
    let mut assigned_num_candidates = std::collections::HashSet::new();
    let mut chosen_numeric_positions = Vec::new();

    for tuple in numeric_scores {
        if assigned_numeric.contains_key(&tuple.field_idx) || assigned_num_candidates.contains(&tuple.cand_idx) {
            continue;
        }
        let val = numeric_candidates[tuple.cand_idx].value;
        let pos = numeric_candidates[tuple.cand_idx].start_pos;
        assigned_numeric.insert(tuple.field_idx, val);
        assigned_num_candidates.insert(tuple.cand_idx);
        chosen_numeric_positions.push(pos);
    }

    // Step 7: Boolean Field Assignment
    let mut bool_scores = Vec::new();
    for (f_idx, field) in schema.fields.iter().enumerate() {
        if field.field_type != FieldType::Boolean {
            continue;
        }
        let anchors = &field_anchors[f_idx];
        
        for (c_idx, cand) in bool_candidates.iter().enumerate() {
            let min_dist = if anchors.is_empty() {
                lower.len() as f64
            } else {
                let mut d = f64::MAX;
                for &a in anchors {
                    let dist = (cand.start_pos as isize - a as isize).abs() as f64;
                    if dist < d {
                        d = dist;
                    }
                }
                d
            };
            bool_scores.push(ScoreTuple {
                field_idx: f_idx,
                cand_idx: c_idx,
                score: min_dist,
            });
        }
    }

    bool_scores.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    let mut assigned_bool = std::collections::HashMap::new();
    let mut assigned_bool_candidates = std::collections::HashSet::new();

    for tuple in bool_scores {
        if assigned_bool.contains_key(&tuple.field_idx) || assigned_bool_candidates.contains(&tuple.cand_idx) {
            continue;
        }
        let val = bool_candidates[tuple.cand_idx].value;
        assigned_bool.insert(tuple.field_idx, val);
        assigned_bool_candidates.insert(tuple.cand_idx);
    }

    // Step 8: String Field Assignment (Clustering proximity search)
    let mut string_scores = Vec::new();
    for (f_idx, field) in schema.fields.iter().enumerate() {
        if field.field_type != FieldType::String {
            continue;
        }
        let anchors = &field_anchors[f_idx];
        
        for (c_idx, cand) in string_candidates.iter().enumerate() {
            let own_dist = if anchors.is_empty() {
                lower.len() as f64
            } else {
                let mut d = f64::MAX;
                for &a in anchors {
                    let dist = (cand.start_pos as isize - a as isize).abs() as f64;
                    if dist < d {
                        d = dist;
                    }
                }
                d
            };
            
            let mut min_numeric_dist = lower.len() as f64;
            for &num_pos in &chosen_numeric_positions {
                let dist = (cand.start_pos as isize - num_pos as isize).abs() as f64;
                if dist < min_numeric_dist {
                    min_numeric_dist = dist;
                }
            }
            
            let score = if anchors.is_empty() {
                min_numeric_dist * 6.0
            } else {
                own_dist + min_numeric_dist * 6.0
            };
            
            string_scores.push(ScoreTuple {
                field_idx: f_idx,
                cand_idx: c_idx,
                score,
            });
        }
    }

    string_scores.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    let mut assigned_string = std::collections::HashMap::new();
    let mut assigned_str_candidates = std::collections::HashSet::new();

    for tuple in string_scores {
        if assigned_string.contains_key(&tuple.field_idx) || assigned_str_candidates.contains(&tuple.cand_idx) {
            continue;
        }
        let val = string_candidates[tuple.cand_idx].value.clone();
        assigned_string.insert(tuple.field_idx, val);
        assigned_str_candidates.insert(tuple.cand_idx);
    }

    // Step 9: Assemble final dynamic JSON payload
    let mut map = serde_json::Map::new();
    for (f_idx, field) in schema.fields.iter().enumerate() {
        match field.field_type {
            FieldType::String => {
                let val = assigned_string.get(&f_idx).cloned().unwrap_or_else(|| {
                    if field.required {
                        "unknown".to_string()
                    } else {
                        "".to_string()
                    }
                });
                map.insert(field.name.clone(), serde_json::Value::String(val));
            }
            FieldType::Integer => {
                let val = assigned_numeric.get(&f_idx).map(|&v| v.round() as i64).unwrap_or(0);
                map.insert(field.name.clone(), serde_json::Value::Number(serde_json::Number::from(val)));
            }
            FieldType::Float => {
                let val = assigned_numeric.get(&f_idx).cloned().unwrap_or(0.0);
                map.insert(field.name.clone(), serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap_or_else(|| serde_json::Number::from(0))));
            }
            FieldType::Boolean => {
                let val = assigned_bool.get(&f_idx).cloned().unwrap_or(false);
                map.insert(field.name.clone(), serde_json::Value::Bool(val));
            }
        }
    }

    serde_json::Value::Object(map)
}

