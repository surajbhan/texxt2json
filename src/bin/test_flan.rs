use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoder = Session::builder()?
        .commit_from_file("models/flan_t5_encoder.onnx")?;
    println!("=== ENCODER INPUTS ===");
    for input in &encoder.inputs {
        println!("Name: {}, Type: {:?}", input.name, input.input_type);
    }
    println!("=== ENCODER OUTPUTS ===");
    for output in &encoder.outputs {
        println!("Name: {}, Type: {:?}", output.name, output.output_type);
    }

    let decoder = Session::builder()?
        .commit_from_file("models/flan_t5_decoder.onnx")?;
    println!("=== DECODER INPUTS ===");
    for input in &decoder.inputs {
        println!("Name: {}, Type: {:?}", input.name, input.input_type);
    }
    println!("=== DECODER OUTPUTS ===");
    for output in &decoder.outputs {
        println!("Name: {}, Type: {:?}", output.name, output.output_type);
    }

    Ok(())
}
