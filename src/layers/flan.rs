use anyhow::{Result, anyhow};
use ort::session::Session;
use ndarray::Array2;
use tokenizers::Tokenizer;
use std::path::Path;

pub struct FlanLayer {
    encoder_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
}

impl FlanLayer {
    pub fn new(model_path: &str) -> Result<Self> {
        let model_path_buf = Path::new(model_path);
        let parent = model_path_buf.parent().unwrap_or_else(|| Path::new("."));
        
        let tokenizer_path = parent.join("tokenizer.json");
        let alt_tokenizer_path = parent.join("flan_tokenizer.json");
        let final_tokenizer_path = if tokenizer_path.exists() {
            tokenizer_path
        } else if alt_tokenizer_path.exists() {
            alt_tokenizer_path
        } else {
            return Err(anyhow!(
                "Flan-T5 tokenizer.json not found adjacent to model file at {:?} or {:?}",
                tokenizer_path,
                alt_tokenizer_path
            ));
        };

        let tokenizer = Tokenizer::from_file(&final_tokenizer_path)
            .map_err(|e| anyhow!("Failed to load Flan-T5 tokenizer: {}", e))?;

        // Resolve encoder & decoder paths
        let encoder_path = if model_path.contains("encoder") {
            model_path_buf.to_path_buf()
        } else if parent.join("flan_t5_encoder.onnx").exists() {
            parent.join("flan_t5_encoder.onnx")
        } else {
            model_path_buf.to_path_buf()
        };

        let decoder_path = if encoder_path.to_str().unwrap().contains("encoder") {
            Path::new(&encoder_path.to_str().unwrap().replace("encoder", "decoder")).to_path_buf()
        } else if parent.join("flan_t5_decoder.onnx").exists() {
            parent.join("flan_t5_decoder.onnx")
        } else {
            parent.join("decoder.onnx")
        };

        println!("[Flan-T5] Loading encoder session from {:?}", encoder_path);
        let encoder_session = Session::builder()
            .map_err(|e| anyhow!("Failed to build Session builder: {:?}", e))?
            .with_intra_threads(num_cpus::get_physical())
            .map_err(|e| anyhow!("Failed to set intra threads: {:?}", e))?
            .commit_from_file(&encoder_path)
            .map_err(|e| anyhow!("Failed to load Flan-T5 Encoder ONNX: {:?}", e))?;

        println!("[Flan-T5] Loading decoder session from {:?}", decoder_path);
        let decoder_session = Session::builder()
            .map_err(|e| anyhow!("Failed to build Session builder: {:?}", e))?
            .with_intra_threads(num_cpus::get_physical())
            .map_err(|e| anyhow!("Failed to set intra threads: {:?}", e))?
            .commit_from_file(&decoder_path)
            .map_err(|e| anyhow!("Failed to load Flan-T5 Decoder ONNX: {:?}", e))?;

        Ok(Self {
            encoder_session,
            decoder_session,
            tokenizer,
        })
    }

    /// Perform sequence-to-sequence parsing on text using custom schema
    pub fn translate(&self, input: &str, schema: &super::super::ExtractionSchema) -> Result<Option<serde_json::Value>> {
        let schema_json = serde_json::to_string(schema)?;
        let prompt = format!(
            "Extract JSON matching schema: {}\nText: {}\nJSON:",
            schema_json,
            input
        );

        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("Flan-T5 Tokenization failed: {}", e))?;
            
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Ok(None);
        }

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)?;
        
        let input_ids_tensor = ort::value::Tensor::from_array(input_ids_array)
            .map_err(|e| anyhow!("Failed to create input_ids tensor: {:?}", e))?;
        let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask_array.clone())
            .map_err(|e| anyhow!("Failed to create attention_mask tensor: {:?}", e))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ]?;
        
        let encoder_outputs = self.encoder_session.run(inputs)
            .map_err(|e| anyhow!("Flan encoder session run failed: {:?}", e))?;
            
        let last_hidden_state_val = encoder_outputs.get("last_hidden_state")
            .ok_or_else(|| anyhow!("Failed to get last_hidden_state output from Flan encoder"))?;
            
        let last_hidden_state_view = last_hidden_state_val.try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract last_hidden_state: {:?}", e))?;
        let last_hidden_state_owned = last_hidden_state_view.to_owned();

        // Autoregressive greedy decoding loop
        let mut decoder_input_ids = vec![0i64];
        let max_new_tokens = 128;
        let eos_token_id = 1i64; // Flan-T5 EOS token is 1
        
        for _step in 0..max_new_tokens {
            let dec_len = decoder_input_ids.len();
            let decoder_input_ids_array = Array2::from_shape_vec((1, dec_len), decoder_input_ids.clone())?;
            let decoder_input_ids_tensor = ort::value::Tensor::from_array(decoder_input_ids_array)
                .map_err(|e| anyhow!("Failed to create decoder input_ids tensor: {:?}", e))?;
                
            let encoder_attention_mask_tensor = ort::value::Tensor::from_array(attention_mask_array.clone())
                .map_err(|e| anyhow!("Failed to create encoder_attention_mask tensor: {:?}", e))?;
                
            let last_hidden_state_tensor = ort::value::Tensor::from_array(last_hidden_state_owned.clone())
                .map_err(|e| anyhow!("Failed to create last_hidden_state tensor: {:?}", e))?;

            let decoder_inputs = ort::inputs![
                "encoder_attention_mask" => encoder_attention_mask_tensor,
                "input_ids" => decoder_input_ids_tensor,
                "encoder_hidden_states" => last_hidden_state_tensor,
            ]?;
            
            let decoder_outputs = self.decoder_session.run(decoder_inputs)
                .map_err(|e| anyhow!("Flan decoder session run failed: {:?}", e))?;
                
            let logits_val = decoder_outputs.get("logits")
                .ok_or_else(|| anyhow!("Failed to get logits from Flan decoder"))?;
                
            let logits_view = logits_val.try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("Failed to extract logits: {:?}", e))?;
                
            let logits_3d = logits_view.into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| anyhow!("Logits shape mismatch: {:?}", e))?;
                
            let last_token_logits = logits_3d.slice(ndarray::s![0, dec_len - 1, ..]);
            
            let mut max_val = f32::MIN;
            let mut max_idx = 0;
            for (idx, &val) in last_token_logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = idx;
                }
            }
            
            let next_token = max_idx as i64;
            if next_token == eos_token_id {
                break;
            }
            
            decoder_input_ids.push(next_token);
        }

        let u32_ids: Vec<u32> = decoder_input_ids.iter().map(|&x| x as u32).collect();
        let decoded_str = self.tokenizer.decode(&u32_ids, true)
            .map_err(|e| anyhow!("Failed to decode Flan-T5 output: {}", e))?;
            
        let cleaned = clean_json_blocks(&decoded_str);
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cleaned) {
            if val.is_object() {
                Ok(Some(val))
            } else {
                println!("[Flan-T5 Warning] Output was valid JSON but not a JSON object: {:?}", decoded_str);
                Ok(None)
            }
        } else {
            println!("[Flan-T5 Warning] Output was not valid JSON. Raw output: {:?}", decoded_str);
            Ok(None)
        }
    }
}

fn clean_json_blocks(raw: &str) -> String {
    let mut cleaned = raw.trim().to_string();
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
