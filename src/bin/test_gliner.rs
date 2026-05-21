use orp::params::RuntimeParameters;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::span::SpanMode;

fn main() {
    let model = GLiNER::<SpanMode>::new(
        Parameters::default().with_threshold(0.15),
        RuntimeParameters::default(),
        "models/gliner_tokenizer.json",
        "models/gliner_medium.onnx",
    ).expect("Failed to load gliner");

    let inputs = vec![
        "What's the latest price for 'TSLA' and what are the earnings like for 'FB'? After that, order a 'lentils' for 1.57 and apply 8.0% processing tax.",
        "Search for Tmall products with keyword 'backpack' on the first page and categorize a product with title 'guacamole' and price 7.82. Include 12.0% tax.",
        "What's the latest price for 'TSLA' and what are the earnings like for 'FB'? After that, order a 'High-End Gaming Laptop' for 1283.0 and apply 12.5% processing tax.",
        "We had 10 options but bought 'vinegar' for 1708.08 with 5.5% tax on May 21st.",
        "What's the latest price for 'TSLA' and what are the earnings like for 'FB'? After that, order a 'milk' for 6.03 and apply 8.0% processing tax.",
        "Search for Tmall products with keyword 'backpack' on the first page and categorize a product with title 'bacon' and price 2240.77. Include 12.0% tax."
    ];
    let schema_fields = vec![
        ("product", "String", "the name of the product being purchased"),
        ("price", "Float", "the base price of the item before tax"),
        ("tax", "Float", "the tax or processing fee percentage rate"),
    ];
    let mut label_to_field_name = std::collections::HashMap::new();
    let mut labels = Vec::new();

    for &(name, field_type, desc) in &schema_fields {
        let name_lower = name.to_lowercase();
        
        // Only push name itself if it's not a generic word like "product" or "item"
        let is_generic = name_lower == "product" || name_lower == "item";
        if !is_generic {
            labels.push(name.to_string());
            label_to_field_name.insert(name.to_string(), name.to_string());
        }
        
        // Add description as a label
        if !desc.is_empty() {
            labels.push(desc.to_string());
            label_to_field_name.insert(desc.to_string(), name.to_string());
        }

        if field_type == "Float" || field_type == "Integer" {
            let mut field_aliases = Vec::new();
            if name_lower.contains("tax") || desc.to_lowercase().contains("tax") {
                field_aliases.push("tax rate".to_string());
                field_aliases.push("tax percentage".to_string());
                field_aliases.push("tax percent".to_string());
                field_aliases.push("sales tax".to_string());
                field_aliases.push("processing fee".to_string());
                field_aliases.push("processing fees".to_string());
                field_aliases.push("fee".to_string());
            } else if name_lower.contains("price") || desc.to_lowercase().contains("price") || name_lower.contains("cost") {
                field_aliases.push("base price".to_string());
                field_aliases.push("item price".to_string());
                field_aliases.push("cost".to_string());
            }
            for alias in field_aliases {
                if !labels.contains(&alias) {
                    labels.push(alias.clone());
                }
                label_to_field_name.insert(alias, name.to_string());
            }
        } else if field_type == "String" {
            let mut field_aliases = Vec::new();
            if name_lower.contains("product") || desc.to_lowercase().contains("product") || name_lower.contains("item") {
                field_aliases.push("purchased product".to_string());
                field_aliases.push("ordered product".to_string());
                field_aliases.push("product being categorized".to_string());
                field_aliases.push("product being purchased".to_string());
                field_aliases.push("purchased item".to_string());
            }
            for alias in field_aliases {
                if !labels.contains(&alias) {
                    labels.push(alias.clone());
                }
                label_to_field_name.insert(alias, name.to_string());
            }
        }
    }

    let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

    for input in inputs {
        println!("Input: {}", input);
        let text_input = TextInput::from_str(&[input], &label_strs).unwrap();
        let output = model.inference(text_input).unwrap();
        
        let mut candidates: std::collections::HashMap<String, (String, f32)> = std::collections::HashMap::new();
        
        for span in &output.spans[0] {
            let field_name = label_to_field_name.get(span.class()).unwrap();
            let is_exact = span.class() == field_name;
            let boost = if is_exact { 0.08 } else { 0.0 };
            let final_prob = span.probability() + boost;
            
            println!("  Span debug: [{}], class: {}, raw_prob: {}, final_prob: {} (maps to: {})", 
                     span.text(), span.class(), span.probability(), final_prob, field_name);
                     
            let should_insert = match candidates.get(field_name) {
                Some((_, existing_prob)) => final_prob > *existing_prob,
                None => true,
            };
            if should_insert {
                candidates.insert(field_name.clone(), (span.text().trim().to_string(), final_prob));
            }
        }
        
        println!("  Selected candidates: {:?}", candidates);
        println!();
    }
}
