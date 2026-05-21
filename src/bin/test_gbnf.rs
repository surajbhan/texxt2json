use structured_json_pipeline::{ExtractionSchema, FieldSchema, FieldType};
use structured_json_pipeline::layers::qwen::generate_gbnf;

fn main() {
    let schema = ExtractionSchema {
        fields: vec![
            FieldSchema {
                name: "product".to_string(),
                field_type: FieldType::String,
                description: "Name of product".to_string(),
                required: true,
            },
            FieldSchema {
                name: "price".to_string(),
                field_type: FieldType::Float,
                description: "Price of product".to_string(),
                required: true,
            },
            FieldSchema {
                name: "tax".to_string(),
                field_type: FieldType::Float,
                description: "Tax rate".to_string(),
                required: true,
            },
        ],
    };
    let gbnf = generate_gbnf(&schema);
    println!("--- GENERATED GBNF ---");
    println!("{}", gbnf);
    println!("----------------------");
}
