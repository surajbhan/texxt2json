use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    #[serde(alias = "string", alias = "String")]
    String,
    #[serde(alias = "integer", alias = "Integer", alias = "int", alias = "Int")]
    Integer,
    #[serde(alias = "float", alias = "Float", alias = "number", alias = "Number")]
    Float,
    #[serde(alias = "boolean", alias = "Boolean", alias = "bool", alias = "Bool")]
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub field_type: FieldType,
    pub description: String,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSchema {
    pub fields: Vec<FieldSchema>,
}
