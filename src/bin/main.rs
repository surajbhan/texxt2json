use structured_json_pipeline::{extract_json, ExtractionSchema, FieldSchema, FieldType};
use anyhow::Result;

fn main() -> Result<()> {
    // Define a dynamic schema for transaction details
    let schema = ExtractionSchema {
        fields: vec![
            FieldSchema {
                name: "product".to_string(),
                field_type: FieldType::String,
                description: "the name of the product being purchased".to_string(),
                required: true,
            },
            FieldSchema {
                name: "price".to_string(),
                field_type: FieldType::Float,
                description: "the base price of the item before tax".to_string(),
                required: true,
            },
            FieldSchema {
                name: "tax".to_string(),
                field_type: FieldType::Float,
                description: "the tax or processing fee percentage rate".to_string(),
                required: true,
            },
        ],
    };

    println!("=============================================================");
    println!("          DEMONSTRATING DYNAMIC SCHEMA EXTRACTION            ");
    println!("=============================================================");

    // Test Case 1: Standard clean payload
    let input1 = "product 'Laptop' bought for 1200 plus 10% tax";
    println!("Input 1: \"{}\"", input1);
    let out1 = extract_json(input1, &schema);
    println!("Extraction output 1: {}\n", out1);

    // Test Case 2: Sourcing slightly messier text stream
    let input2 = "Hey, we found a box of 'Eggs'. Price details say 36.3 plus 8.0 percent processing fees.";
    println!("Input 2: \"{}\"", input2);
    let out2 = extract_json(input2, &schema);
    println!("Extraction output 2: {}", out2);
    println!("=============================================================");

    Ok(())
}
