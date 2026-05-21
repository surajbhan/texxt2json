use structured_json_pipeline::{MultiLayerParser, ExtractionSchema, ExtractionLayer};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use anyhow::Result;

#[derive(Deserialize)]
struct TestCase {
    input: String,
    schema: ExtractionSchema,
    expected: serde_json::Value,
    expected_paid: f64,
    description: String,
}

fn assert_json_eq(actual: &serde_json::Value, expected: &serde_json::Value) -> bool {
    match (actual, expected) {
        (serde_json::Value::Object(act_map), serde_json::Value::Object(exp_map)) => {
            if act_map.len() != exp_map.len() {
                return false;
            }
            for (key, exp_val) in exp_map {
                let act_val = match act_map.get(key) {
                    Some(v) => v,
                    None => return false,
                };
                if !assert_json_eq(act_val, exp_val) {
                    return false;
                }
            }
            true
        }
        (serde_json::Value::Array(act_arr), serde_json::Value::Array(exp_arr)) => {
            if act_arr.len() != exp_arr.len() {
                return false;
            }
            for (act_val, exp_val) in act_arr.iter().zip(exp_arr.iter()) {
                if !assert_json_eq(act_val, exp_val) {
                    return false;
                }
            }
            true
        }
        (serde_json::Value::String(act_s), serde_json::Value::String(exp_s)) => {
            act_s.to_lowercase() == exp_s.to_lowercase()
        }
        (serde_json::Value::Number(act_n), serde_json::Value::Number(exp_n)) => {
            if let (Some(a), Some(e)) = (act_n.as_f64(), exp_n.as_f64()) {
                (a - e).abs() < 1e-4
            } else {
                act_n == exp_n
            }
        }
        (serde_json::Value::Bool(act_b), serde_json::Value::Bool(exp_b)) => {
            act_b == exp_b
        }
        (serde_json::Value::Null, serde_json::Value::Null) => true,
        _ => false,
    }
}

fn main() -> Result<()> {
    let dataset_path = std::env::args().nth(1).unwrap_or_else(|| "data/test_cases.json".to_string());
    println!("[Evaluation] Loading dataset from: {}", dataset_path);

    let file = File::open(&dataset_path)?;
    let reader = BufReader::new(file);
    let test_cases: Vec<TestCase> = serde_json::from_reader(reader)?;

    println!("=============================================================");
    println!("   INITIALIZING structured_json_pipeline MULTI-LAYER PARSER   ");
    println!("=============================================================");

    let parser = MultiLayerParser::new(
        "models/gliner_medium.onnx", 
        "models/flan_t5_base.onnx", 
        "models/qwen3.5_0.8b_q4.gguf"
    )?;

    println!("\n=============================================================");
    println!("             EXECUTING ACCURACY & PERFORMANCE TEST            ");
    println!("=============================================================");

    let mut successful_matches = 0;
    let mut total_latency = std::time::Duration::from_secs(0);
    let mut latencies_ms = Vec::new();

    // Layer counters
    let mut gliner_count = 0;
    let mut flan_count = 0;
    let mut qwen_count = 0;
    let mut proximity_count = 0;

    println!("{:<4} | {:<25} | {:<12} | {:<12} | {:<8} | {:<7}", 
             "ID", "Description", "Exp Paid", "Act Paid", "Latency", "Status");
    println!("{}", "-".repeat(78));

    for (idx, tc) in test_cases.iter().enumerate() {
        let start = Instant::now();
        let (payload, layer) = parser.process_text_with_metadata(&tc.input, &tc.schema);
        let latency = start.elapsed();
        
        total_latency += latency;
        let latency_ms = latency.as_secs_f64() * 1000.0;
        latencies_ms.push(latency_ms);

        match layer {
            ExtractionLayer::Gliner => gliner_count += 1,
            ExtractionLayer::Flan => flan_count += 1,
            ExtractionLayer::Qwen => qwen_count += 1,
            ExtractionLayer::ProximityHeuristic => proximity_count += 1,
        }

        let is_ok = assert_json_eq(&payload, &tc.expected);

        if is_ok {
            successful_matches += 1;
        }

        let status = if is_ok { "PASSED" } else { "FAILED" };

        let act_price = payload.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let act_tax = payload.get("tax").and_then(|v| v.as_f64()).unwrap_or(18.0);
        let act_paid = act_price * (1.0 + act_tax / 100.0);

        println!("{:<4} | {:<25} | {:<12.4} | {:<12.4} | {:<6.2}ms | {:<7}", 
                 idx + 1, 
                 if tc.description.len() > 25 { &tc.description[..22] } else { &tc.description }, 
                 tc.expected_paid, 
                 act_paid, 
                 latency_ms, 
                 status);
        if !is_ok {
            println!("  ↳ Expected JSON: '{}', got: '{}'", tc.expected, payload);
            println!("  ↳ Input text: \"{}\"", tc.input);
        }
    }

    println!("{}", "=".repeat(78));
    println!("                        EVALUATION REPORT                    ");
    println!("{}", "=".repeat(78));
    
    let total_cases = test_cases.len();
    let accuracy = (successful_matches as f64 / total_cases as f64) * 100.0;
    let mean_latency = (total_latency.as_secs_f64() * 1000.0) / total_cases as f64;

    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = ((total_cases as f64 * 0.95).round() as usize).min(total_cases - 1);
    let p99_idx = ((total_cases as f64 * 0.99).round() as usize).min(total_cases - 1);
    
    let p95_latency = latencies_ms[p95_idx];
    let p99_latency = latencies_ms[p99_idx];

    println!("Total Test Cases : {}", total_cases);
    println!("Passed Cases     : {}", successful_matches);
    println!("Failed Cases     : {}", total_cases - successful_matches);
    println!("Accuracy         : {:.2}%", accuracy);
    println!("Mean Latency     : {:.2}ms", mean_latency);
    println!("P95 Latency      : {:.2}ms", p95_latency);
    println!("P99 Latency      : {:.2}ms", p99_latency);
    
    println!("{}", "=".repeat(78));
    println!("                PIPELINE LAYER ROUTING SEGREGATION           ");
    println!("{}", "=".repeat(78));
    println!("Layer 1 (GLiNER ONNX)       : {:>4} cases ({:>6.2}%)", gliner_count, (gliner_count as f64 / total_cases as f64) * 100.0);
    println!("Layer 2 (Flan-T5 ONNX)      : {:>4} cases ({:>6.2}%)", flan_count, (flan_count as f64 / total_cases as f64) * 100.0);
    println!("Layer 3 (Qwen-0.8B GGUF)    : {:>4} cases ({:>6.2}%)", qwen_count, (qwen_count as f64 / total_cases as f64) * 100.0);
    println!("Layer 4 (Proximity Heur.)   : {:>4} cases ({:>6.2}%)", proximity_count, (proximity_count as f64 / total_cases as f64) * 100.0);
    println!("{}", "=============================================================");

    if accuracy == 100.0 {
        println!("  🎉 SUCCESS: All test cases passed perfectly with 100% accuracy!");
    } else {
        println!("  ⚠️ WARNING: Some test cases failed. Please inspect logs above.");
    }
    println!("{}", "=============================================================");

    Ok(())
}

