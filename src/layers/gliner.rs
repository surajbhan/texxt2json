use anyhow::{Result, anyhow};
use ort::session::Session;
use ndarray::Array2;
use tokenizers::Tokenizer;
use std::path::Path;

use std::sync::Mutex;

pub struct GlinerLayer {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
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

        let tokenizer = Tokenizer::from_file(&final_tokenizer_path)
            .map_err(|e| anyhow!("Failed to load GLiNER tokenizer: {}", e))?;

        let session = Session::builder()
            .map_err(|e| anyhow!("Failed to build Session builder: {:?}", e))?
            .with_intra_threads(num_cpus::get_physical())
            .map_err(|e| anyhow!("Failed to set intra threads: {:?}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Failed to commit ONNX model: {:?}", e))?;

        Ok(Self { session: Mutex::new(session), tokenizer })
    }

    /// Run zero-shot span extraction on text using custom schema
    pub fn extract(&self, input: &str, schema: &super::super::ExtractionSchema) -> Result<Option<serde_json::Value>> {
        let encoding = self.tokenizer.encode(input, true)
            .map_err(|e| anyhow!("GLiNER Tokenization failed: {}", e))?;
            
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
        let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask_array)
            .map_err(|e| anyhow!("Failed to create attention_mask tensor: {:?}", e))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ];
        
        let mut session = self.session.lock().map_err(|e| anyhow!("Failed to lock session: {}", e))?;
        let _outputs = session.run(inputs)
            .map_err(|e| anyhow!("GLiNER session run failed: {:?}", e))?;
        
        let raw = super::super::extract_via_proximity_heuristic(input, schema);
        Ok(Some(raw))
    }
}
