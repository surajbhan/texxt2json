use anyhow::{Result, anyhow};
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::LlamaModel,
    model::params::LlamaModelParams,
    context::params::LlamaContextParams,
    sampling::LlamaSampler,
    llama_batch::LlamaBatch,
};
use std::sync::{Arc, Mutex, OnceLock};
use std::path::Path;
use std::num::NonZeroU32;

static BACKEND: OnceLock<Arc<Mutex<LlamaBackend>>> = OnceLock::new();

fn get_backend() -> Result<Arc<Mutex<LlamaBackend>>> {
    let backend = BACKEND.get_or_init(|| {
        let init_backend = LlamaBackend::init()
            .expect("Failed to initialize LlamaBackend");
        Arc::new(Mutex::new(init_backend))
    });
    Ok(backend.clone())
}

pub struct QwenLayer {
    model: LlamaModel,
}

impl QwenLayer {
    pub fn new(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(anyhow!("Qwen GGUF model file not found at {:?}", model_path));
        }

        let backend_arc = get_backend()?;
        let backend = backend_arc.lock().map_err(|e| anyhow!("Failed to lock backend: {}", e))?;

        // 1. Model Initialization
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| anyhow!("Failed to load GGUF model: {:?}", e))?;

        Ok(Self {
            model,
        })
    }

    /// Autoregressive structured extraction from GGUF using schema and dynamic GBNF
    pub fn extract(&self, input: &str, schema: &super::super::ExtractionSchema) -> Result<serde_json::Value> {
        let schema_json = serde_json::to_string(schema)?;
        // Construct Qwen 3.5 Instruction Chat Template format
        let prompt = format!(
            "<|im_start|>system\nYou are a precise structured data extraction engine. Extract JSON matching the schema: {}.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            schema_json,
            input
        );

        let backend_arc = get_backend()?;
        let backend = backend_arc.lock().map_err(|e| anyhow!("Failed to lock backend: {}", e))?;

        // 2. Context Configuration
        let n_ctx = NonZeroU32::new(512); 
        let threads = num_cpus::get_physical() as i32;
        let context_params = LlamaContextParams::default()
            .with_n_ctx(n_ctx)
            .with_n_threads(threads);

        let mut context = self.model.new_context(&backend, context_params)
            .map_err(|e| anyhow!("Failed to create model context: {:?}", e))?;

        // Tokenize prompt
        use llama_cpp_2::model::AddBos;
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)
            .map_err(|e| anyhow!("Qwen tokenization failed: {:?}", e))?;

        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(anyhow!("Prompt resolved to empty token sequence"));
        }

        // Initialize Batch
        let mut batch = LlamaBatch::new(512, 1);
        
        // Add prompt tokens to batch
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == seq_len - 1;
            batch.add(token, i as i32, &[0], is_last)?;
        }

        // Decode initial prompt batch
        context.decode(&mut batch)
            .map_err(|e| anyhow!("Failed decoding initial context: {:?}", e))?;

        // Construct dynamic GBNF grammar
        let gbnf_grammar = generate_gbnf(schema);
        let grammar_sampler = LlamaSampler::grammar(&self.model, "root", &gbnf_grammar)
            .map_err(|e| anyhow!("Failed to initialize grammar sampler: {:?}", e))?;

        // Set up sampler (Grammar Sampling + Greedy Sampling)
        let mut sampler = LlamaSampler::chain_simple([
            grammar_sampler,
            LlamaSampler::greedy(),
        ]);

        let mut output_str = String::new();
        let mut pos = seq_len as i32;
        let eos_token = self.model.token_eos();

        // Setup encoding_rs decoder for piece generation
        let mut utf8_decoder = encoding_rs::UTF_8.new_decoder();

        // Autoregressive decoding loop (up to 256 tokens max output)
        for _ in 0..256 {
            // Sample next token
            let token = sampler.sample(&context, batch.n_tokens() - 1);
            sampler.accept(token);

            if token == eos_token {
                break;
            }

            // Decode token byte piece
            let piece_str = self.model.token_to_piece(token, &mut utf8_decoder, true, None)
                .map_err(|e| anyhow!("Failed to convert token to piece: {:?}", e))?;
            output_str.push_str(&piece_str);

            // Re-feed single token to context
            batch.clear();
            batch.add(token, pos, &[0], true)?;
            context.decode(&mut batch)
                .map_err(|e| anyhow!("Failed decoding generated token: {:?}", e))?;

            pos += 1;
        }

        // Clean JSON parsing from markdown fences
        let cleaned_json = Self::clean_json_blocks(&output_str);
        
        let payload: serde_json::Value = serde_json::from_str(&cleaned_json)
            .map_err(|e| anyhow!("Failed parsing extracted JSON from Qwen: {}\nRaw output: {}", e, output_str))?;

        Ok(payload)
    }

    fn clean_json_blocks(raw: &str) -> String {
        let mut cleaned = raw.trim().to_string();
        
        // Remove ```json wrapper if present
        if cleaned.starts_with("```json") {
            cleaned = cleaned.trim_start_matches("```json").to_string();
        } else if cleaned.starts_with("```") {
            cleaned = cleaned.trim_start_matches("```").to_string();
        }
        
        if cleaned.ends_with("```") {
            cleaned = cleaned.trim_end_matches("```").to_string();
        }
        
        cleaned.trim().to_string()
    }
}

pub fn generate_gbnf(schema: &super::super::ExtractionSchema) -> String {
    let mut gbnf = String::new();
    
    // root rule
    gbnf.push_str("root ::= \"{\\n\"");
    
    for (i, field) in schema.fields.iter().enumerate() {
        let is_last = i == schema.fields.len() - 1;
        
        // Add field name escaped for GBNF literal
        let field_name_literal = format!(" \"  \\\\\\\"{}\\\\\\\": \" ", field.name);
        gbnf.push_str(&field_name_literal);
        
        // Add field value rule reference
        match field.field_type {
            super::super::FieldType::String => gbnf.push_str("string"),
            super::super::FieldType::Integer => gbnf.push_str("integer"),
            super::super::FieldType::Float => gbnf.push_str("float"),
            super::super::FieldType::Boolean => gbnf.push_str("boolean"),
        }
        
        if is_last {
            gbnf.push_str(" \"\\n}\"\n");
        } else {
            gbnf.push_str(" \",\\n\"\n");
        }
    }
    
    // Add sub-rules
    gbnf.push_str("string ::= \"\\\"\" [^\"]* \"\\\"\"\n");
    gbnf.push_str("integer ::= \"-\"? [0-9]+\n");
    gbnf.push_str("float ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n");
    gbnf.push_str("boolean ::= \"true\" | \"false\"\n");
    
    gbnf
}
