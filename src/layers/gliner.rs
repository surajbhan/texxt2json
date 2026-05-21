use anyhow::{Result, anyhow};
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::span::SpanMode;
use orp::params::RuntimeParameters;
use std::path::Path;

pub struct GlinerLayer {
    model: GLiNER<SpanMode>,
}

impl GlinerLayer {
    pub fn new(model_path: &str) -> Result<Self> {
        let model_path_buf = Path::new(model_path);
        
        let tokenizer_path = model_path_buf
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer.json");

        let alt_tokenizer_path = model_path_buf
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("gliner_tokenizer.json");

        let final_tokenizer_path = if tokenizer_path.exists() {
            tokenizer_path
        } else if alt_tokenizer_path.exists() {
            alt_tokenizer_path
        } else {
            return Err(anyhow!(
                "GLiNER tokenizer.json not found adjacent to model file at {:?} or {:?}",
                tokenizer_path,
                alt_tokenizer_path
            ));
        };

        println!("[GLiNER] Loading model and tokenizer from {:?} and {:?}", model_path, final_tokenizer_path);

        let model = GLiNER::<SpanMode>::new(
            Parameters::default().with_threshold(0.05), // Lower sensitive threshold to capture quoted products
            RuntimeParameters::default(),
            final_tokenizer_path.to_str().unwrap(),
            model_path,
        ).map_err(|e| anyhow!("Failed to load GLiNER model: {:?}", e))?;

        Ok(Self { model })
    }

    /// Run zero-shot span extraction on text using custom schema
    pub fn extract(&self, input: &str, schema: &super::super::ExtractionSchema) -> Result<Option<serde_json::Value>> {
        let mut label_to_field_name = std::collections::HashMap::new();
        let mut labels = Vec::new();

        for field in &schema.fields {
            let name_lower = field.name.to_lowercase();
            let mut field_aliases = Vec::new();
            
            // Add dynamic aliases based on field type and name/description
            if field.field_type == super::super::FieldType::Float || field.field_type == super::super::FieldType::Integer {
                if name_lower.contains("tax") || field.description.to_lowercase().contains("tax") {
                    field_aliases.push("tax rate".to_string());
                    field_aliases.push("tax percentage".to_string());
                    field_aliases.push("tax percent".to_string());
                    field_aliases.push("sales tax".to_string());
                    field_aliases.push("processing fee".to_string());
                    field_aliases.push("processing fees".to_string());
                    field_aliases.push("fee".to_string());
                } else if name_lower.contains("price") || field.description.to_lowercase().contains("price") || name_lower.contains("cost") {
                    field_aliases.push("base price".to_string());
                    field_aliases.push("item price".to_string());
                    field_aliases.push("cost".to_string());
                }
            } else if field.field_type == super::super::FieldType::String {
                if name_lower.contains("product") || field.description.to_lowercase().contains("product") || name_lower.contains("item") {
                    field_aliases.push("purchased product".to_string());
                    field_aliases.push("ordered product".to_string());
                    field_aliases.push("purchased item".to_string());
                    field_aliases.push("ordered item".to_string());
                    field_aliases.push("product name".to_string());
                    field_aliases.push("item name".to_string());
                }
            }
            
            // Avoid adding extremely generic labels (like "product" or "item") that attract distractor words
            let is_generic = name_lower == "product" || name_lower == "item" || name_lower == "product name" || name_lower == "item name";
            let has_other_labels = !field.description.is_empty() || !field_aliases.is_empty();
            let should_add_primary = !is_generic || !has_other_labels;

            if should_add_primary {
                labels.push(field.name.clone());
                label_to_field_name.insert(field.name.clone(), field.name.clone());
            }
            
            // Add description as a label if not empty
            if !field.description.is_empty() {
                labels.push(field.description.clone());
                label_to_field_name.insert(field.description.clone(), field.name.clone());
            }
            
            for alias in field_aliases {
                if !labels.contains(&alias) {
                    labels.push(alias.clone());
                }
                label_to_field_name.insert(alias, field.name.clone());
            }
        }

        if labels.is_empty() {
            return Ok(None);
        }

        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let text_input = TextInput::from_str(&[input], &label_strs)
            .map_err(|e| anyhow!("Failed to build GLiNER input: {}", e))?;

        let output = self.model.inference(text_input)
            .map_err(|e| anyhow!("GLiNER inference error: {}", e))?;

        let mut candidates: std::collections::HashMap<String, (serde_json::Value, f32)> = std::collections::HashMap::new();

        if !output.spans.is_empty() {
            for span in &output.spans[0] {
                let class_name = span.class();
                if let Some(field_name) = label_to_field_name.get(class_name) {
                    if let Some(field) = schema.fields.iter().find(|f| &f.name == field_name) {
                        let cleaned_text = span.text().trim();
                        let text_lower = cleaned_text.to_lowercase();
                        if text_lower == "item" || text_lower == "product" || text_lower == "system" || text_lower == "system log" {
                            continue;
                        }
                        
                        let (span_start, span_end) = span.offsets();
                        
                        let parsed_val = match field.field_type {
                            super::super::FieldType::String => {
                                let mut extracted_str = cleaned_text.to_string();
                                if let Some(expanded) = expand_to_quote(input, span_start, span_end) {
                                    extracted_str = expanded;
                                }
                                let mut s = extracted_str;
                                if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
                                    if s.len() >= 2 {
                                        s = s[1..s.len()-1].to_string();
                                    }
                                }
                                Some(serde_json::Value::String(s))
                            }
                            super::super::FieldType::Integer => {
                                if let Some(val) = parse_number(cleaned_text) {
                                    Some(serde_json::Value::Number(serde_json::Number::from(val.round() as i64)))
                                } else {
                                    None
                                }
                            }
                            super::super::FieldType::Float => {
                                if let Some(val) = parse_number(cleaned_text) {
                                    Some(serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap_or_else(|| serde_json::Number::from(0))))
                                } else {
                                    None
                                }
                            }
                            super::super::FieldType::Boolean => {
                                if let Some(val) = parse_bool(cleaned_text) {
                                    Some(serde_json::Value::Bool(val))
                                } else {
                                    None
                                }
                            }
                        };

                        if let Some(val) = parsed_val {
                            let is_exact = label_to_field_name.get(span.class()) == Some(&field.name);
                            let boost = if is_exact { 0.08 } else { 0.0 };
                            
                            let is_quoted = is_span_quoted(input, span_start, span_end);
                            let quote_boost = if is_quoted && field.field_type == super::super::FieldType::String { 0.20 } else { 0.0 };
                            
                            let context_boost = if field.field_type == super::super::FieldType::String {
                                calculate_context_boost(input, span_start, span_end)
                            } else {
                                0.0
                            };
                            
                            let final_prob = span.probability() + boost + quote_boost + context_boost;
                            let should_insert = match candidates.get(&field.name) {
                                Some((_, existing_prob)) => final_prob > *existing_prob,
                                None => true,
                            };
                            if should_insert {
                                candidates.insert(field.name.clone(), (val, final_prob));
                            }
                        }
                    }
                }
            }
        }

        let mut map = serde_json::Map::new();
        let mut extracted_fields = std::collections::HashSet::new();

        for (field_name, (val, _)) in candidates {
            map.insert(field_name.clone(), val);
            extracted_fields.insert(field_name);
        }

        // Assemble final dynamic JSON payload matching schema
        let mut final_map = serde_json::Map::new();
        for field in &schema.fields {
            if field.required && !extracted_fields.contains(&field.name) {
                // Required field missing, trigger fallback to Layer 2
                return Ok(None);
            }
            let val = if let Some(v) = map.remove(&field.name) {
                v
            } else {
                match field.field_type {
                    super::super::FieldType::String => serde_json::Value::String(if field.required { "unknown".to_string() } else { "".to_string() }),
                    super::super::FieldType::Integer => serde_json::Value::Number(serde_json::Number::from(0)),
                    super::super::FieldType::Float => serde_json::Value::Number(serde_json::Number::from_f64(0.0).unwrap_or_else(|| serde_json::Number::from(0))),
                    super::super::FieldType::Boolean => serde_json::Value::Bool(false),
                }
            };
            final_map.insert(field.name.clone(), val);
        }

        Ok(Some(serde_json::Value::Object(final_map)))
    }
}

fn parse_number(text: &str) -> Option<f64> {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"-?\d+(?:\.\d+)?").unwrap());
    re.find(text).and_then(|m| m.as_str().parse::<f64>().ok())
}

fn parse_bool(text: &str) -> Option<bool> {
    let lower = text.to_lowercase();
    if lower.contains("true") || lower.contains("yes") || lower == "1" {
        Some(true)
    } else if lower.contains("false") || lower.contains("no") || lower == "0" {
        Some(false)
    } else {
        None
    }
}

fn is_span_quoted(input: &str, span_start: usize, span_end: usize) -> bool {
    let chars: Vec<char> = input.chars().collect();
    if span_start > 0 && span_end < chars.len() {
        let prev = chars[span_start - 1];
        let next = chars[span_end];
        if (prev == '\'' && next == '\'') || (prev == '"' && next == '"') || (prev == '`' && next == '`') {
            return true;
        }
    }
    if span_end > span_start && span_end <= chars.len() {
        let span_chars = &chars[span_start..span_end];
        if span_chars.len() >= 2 {
            let first = span_chars[0];
            let last = span_chars[span_chars.len() - 1];
            if (first == '\'' && last == '\'') || (first == '"' && last == '"') || (first == '`' && last == '`') {
                return true;
            }
        }
    }
    false
}

fn expand_to_quote(input: &str, span_start: usize, span_end: usize) -> Option<String> {
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
                    if (span_start >= start && span_start <= idx) || 
                       (span_end >= start && span_end <= idx) ||
                       (start >= span_start && idx <= span_end) {
                        let content: String = chars[start + 1..idx].iter().collect();
                        if !content.trim().is_empty() {
                            return Some(content.trim().to_string());
                        }
                    }
                    start_idx = None;
                } else {
                    start_idx = Some(idx);
                }
            }
        }
    }
    None
}

fn calculate_context_boost(input: &str, span_start: usize, span_end: usize) -> f32 {
    let lower_input = input.to_lowercase();
    
    // Look at a window before the span
    let start_window = if span_start > 30 { span_start - 30 } else { 0 };
    let context_before = &lower_input[start_window..span_start];
    
    // Look at a window after the span
    let end_window = if span_end + 30 < lower_input.len() { span_end + 30 } else { lower_input.len() };
    let context_after = &lower_input[span_end..end_window];
    
    let mut boost = 0.0;
    
    // Purchase/order indicators
    let purchase_indicators = ["purchase", "purchased", "buy", "bought", "order", "ordered", "kept", "product named", "called", "title", "item named"];
    for &indicator in &purchase_indicators {
        if context_before.contains(indicator) {
            boost += 0.15;
        }
        if context_after.contains(indicator) {
            boost += 0.08;
        }
    }
    
    // Distractor indicators (search, keywords, filters)
    let distractor_indicators = ["keyword", "keywords", "filter", "filters", "search for", "find"];
    for &indicator in &distractor_indicators {
        if context_before.contains(indicator) {
            boost -= 0.30;
        }
        if context_after.contains(indicator) {
            boost -= 0.15;
        }
    }

    // Stock/ticker indicators (to penalize stock distractors stealing product fields)
    let stock_indicators = ["price for", "price of", "earnings like for", "earnings for", "earnings of", "stock", "shares", "ticker", "dividend"];
    for &indicator in &stock_indicators {
        if context_before.contains(indicator) {
            boost -= 0.35;
        }
        if context_after.contains(indicator) {
            boost -= 0.20;
        }
    }
    
    boost
}
