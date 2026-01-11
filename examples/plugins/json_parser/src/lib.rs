// JSON Parser Plugin Example
// Demonstrates how to create a VeZ plugin

use vez_plugin::*;

pub struct JsonParserPlugin {
    metadata: PluginMetadata,
}

impl JsonParserPlugin {
    pub fn new() -> Self {
        JsonParserPlugin {
            metadata: PluginBuilder::new("json-parser".to_string())
                .version("1.0.0".to_string())
                .author("VeZ Team".to_string())
                .description("JSON parsing and serialization".to_string())
                .add_capability(PluginCapability::SyntaxExtension)
                .add_capability(PluginCapability::TypeSystem)
                .add_capability(PluginCapability::Macro)
                .build_metadata(),
        }
    }
}

impl Plugin for JsonParserPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn initialize(&mut self, context: &mut PluginContext) -> Result<()> {
        println!("JSON Parser Plugin initialized");
        
        // Register post-parse hook
        context.hooks.post_parse.push(Box::new(|module| {
            println!("Processing module: {}", module.name);
            Ok(())
        }));
        
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<()> {
        println!("JSON Parser Plugin shutting down");
        Ok(())
    }
}

impl TypePlugin for JsonParserPlugin {
    fn define_types(&self) -> Vec<TypeDefinition> {
        vec![
            // JsonValue enum
            TypeDefinition {
                name: "JsonValue".to_string(),
                kind: TypeKind::Enum(vec![
                    Variant {
                        name: "Null".to_string(),
                        fields: vec![],
                    },
                    Variant {
                        name: "Bool".to_string(),
                        fields: vec![
                            Field {
                                name: "0".to_string(),
                                ty: Type::Bool,
                            },
                        ],
                    },
                    Variant {
                        name: "Number".to_string(),
                        fields: vec![
                            Field {
                                name: "0".to_string(),
                                ty: Type::F64,
                            },
                        ],
                    },
                    Variant {
                        name: "String".to_string(),
                        fields: vec![
                            Field {
                                name: "0".to_string(),
                                ty: Type::String,
                            },
                        ],
                    },
                ]),
                methods: vec![
                    MethodSignature {
                        name: "is_null".to_string(),
                        params: vec![],
                        return_type: Type::Bool,
                    },
                    MethodSignature {
                        name: "as_bool".to_string(),
                        params: vec![],
                        return_type: Type::Option(Box::new(Type::Bool)),
                    },
                    MethodSignature {
                        name: "as_number".to_string(),
                        params: vec![],
                        return_type: Type::Option(Box::new(Type::F64)),
                    },
                ],
            },
        ]
    }
    
    fn type_check(&self, expr: &Expr, context: &TypeContext) -> Result<Type> {
        // Type checking for JSON expressions
        Ok(Type::JsonValue)
    }
    
    fn type_inference(&self, expr: &Expr, context: &TypeContext) -> Result<Type> {
        // Type inference for JSON expressions
        Ok(Type::JsonValue)
    }
}

impl MacroPlugin for JsonParserPlugin {
    fn expand_macro(&self, name: &str, args: &[Expr]) -> Result<Expr> {
        if name == "json" {
            // Expand json!(...) macro
            if args.len() != 1 {
                return Err(Error::new("json! expects 1 argument", Span::dummy()));
            }
            
            // Parse JSON at compile time
            let json_str = match &args[0] {
                Expr::Literal { value: Literal::String(s), .. } => s,
                _ => return Err(Error::new("json! expects string literal", Span::dummy())),
            };
            
            // Generate code to construct JsonValue
            Ok(parse_json_literal(json_str)?)
        } else {
            Err(Error::new(format!("Unknown macro: {}", name), Span::dummy()))
        }
    }
    
    fn macro_names(&self) -> Vec<String> {
        vec!["json".to_string()]
    }
}

// Helper function to parse JSON literal
fn parse_json_literal(json: &str) -> Result<Expr> {
    // Simplified JSON parsing
    // In production, use a proper JSON parser
    
    if json == "null" {
        Ok(Expr::Variable {
            name: "JsonValue::Null".to_string(),
            span: Span::dummy(),
        })
    } else if json == "true" || json == "false" {
        Ok(Expr::Call {
            func: Box::new(Expr::Variable {
                name: "JsonValue::Bool".to_string(),
                span: Span::dummy(),
            }),
            args: vec![Expr::Literal {
                value: Literal::Bool(json == "true"),
                span: Span::dummy(),
            }],
            span: Span::dummy(),
        })
    } else {
        // Default to string
        Ok(Expr::Call {
            func: Box::new(Expr::Variable {
                name: "JsonValue::String".to_string(),
                span: Span::dummy(),
            }),
            args: vec![Expr::Literal {
                value: Literal::String(json.to_string()),
                span: Span::dummy(),
            }],
            span: Span::dummy(),
        })
    }
}

// Export plugin
#[no_mangle]
pub extern "C" fn vez_plugin_create() -> Box<dyn Plugin> {
    Box::new(JsonParserPlugin::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_creation() {
        let plugin = JsonParserPlugin::new();
        assert_eq!(plugin.metadata().name, "json-parser");
    }
    
    #[test]
    fn test_type_definitions() {
        let plugin = JsonParserPlugin::new();
        let types = plugin.define_types();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "JsonValue");
    }
}
