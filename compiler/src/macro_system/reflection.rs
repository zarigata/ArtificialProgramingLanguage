//! Compile-time reflection system
//!
//! Provides introspection capabilities at compile time for:
//! - Struct field enumeration
//! - Type derivation generation
//! - Automatic trait implementations

use std::collections::HashMap;
use crate::parser::ast::{Struct, Enum, Function, Type as AstType, Expr, Stmt};

#[derive(Debug, Clone)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<FieldInfo>,
    pub generics: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub ty: String,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct EnumInfo {
    pub name: String,
    pub variants: Vec<VariantInfo>,
    pub generics: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VariantInfo {
    pub name: String,
    pub data: Option<String>,
    pub discriminant: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub params: Vec<ParamInfo>,
    pub return_type: String,
}

#[derive(Debug, Clone)]
pub struct ParamInfo {
    pub name: String,
    pub ty: String,
}

pub struct ReflectionRegistry {
    structs: HashMap<String, StructInfo>,
    enums: HashMap<String, EnumInfo>,
    functions: HashMap<String, FunctionInfo>,
}

impl ReflectionRegistry {
    pub fn new() -> Self {
        ReflectionRegistry {
            structs: HashMap::new(),
            enums: HashMap::new(),
            functions: HashMap::new(),
        }
    }
    
    pub fn register_struct(&mut self, s: &Struct) {
        let fields: Vec<FieldInfo> = s.fields.iter().enumerate().map(|(i, f)| {
            FieldInfo {
                name: f.name.clone(),
                ty: type_to_string(&f.ty),
                index: i,
            }
        }).collect();
        
        let info = StructInfo {
            name: s.name.clone(),
            fields,
            generics: s.generics.iter().map(|g| g.name.clone()).collect(),
        };
        
        self.structs.insert(s.name.clone(), info);
    }
    
    pub fn register_enum(&mut self, e: &crate::parser::ast::Enum) {
        let variants: Vec<VariantInfo> = e.variants.iter().enumerate().map(|(i, v)| {
            VariantInfo {
                name: v.name.clone(),
                data: v.data.as_ref().map(|t| type_to_string(t)),
                discriminant: Some(i as i64),
            }
        }).collect();
        
        let info = EnumInfo {
            name: e.name.clone(),
            variants,
            generics: e.generics.iter().map(|g| g.name.clone()).collect(),
        };
        
        self.enums.insert(e.name.clone(), info);
    }
    
    pub fn register_function(&mut self, f: &Function) {
        let params: Vec<ParamInfo> = f.params.iter().map(|p| {
            ParamInfo {
                name: p.name.clone(),
                ty: type_to_string(&p.ty),
            }
        }).collect();
        
        let info = FunctionInfo {
            name: f.name.clone(),
            params,
            return_type: f.return_type.as_ref().map(type_to_string).unwrap_or_default(),
        };
        
        self.functions.insert(f.name.clone(), info);
    }
    
    pub fn get_struct(&self, name: &str) -> Option<&StructInfo> {
        self.structs.get(name)
    }
    
    pub fn get_enum(&self, name: &str) -> Option<&EnumInfo> {
        self.enums.get(name)
    }
    
    pub fn get_function(&self, name: &str) -> Option<&FunctionInfo> {
        self.functions.get(name)
    }
    
    pub fn generate_derive_serialize(&self, struct_name: &str) -> Option<String> {
        let info = self.get_struct(struct_name)?;
        
        let mut code = format!(
            "impl Serialize for {} {{\n  fn serialize(&self) -> String {{\n    let mut result = String::new();\n",
            struct_name
        );
        
        for field in &info.fields {
            code.push_str(&format!(
                "    result.push_str(&format(\"{{:?}}\", &self.{}));\n",
                field.name
            ));
        }
        
        code.push_str("    result\n  }\n}\n");
        Some(code)
    }
    
    pub fn generate_derive_debug(&self, struct_name: &str) -> Option<String> {
        let info = self.get_struct(struct_name)?;
        
        let mut code = format!(
            "impl Debug for {} {{\n  fn fmt(&self) -> String {{\n    format!(\"{} {{",
            struct_name, struct_name
        );
        
        let fields: Vec<String> = info.fields.iter().map(|f| {
            format!("{}: {{:?}}", f.name)
        }).collect();
        
        code.push_str(&fields.join(", "));
        code.push_str("}}\", ");
        
        let field_refs: Vec<String> = info.fields.iter().map(|f| {
            format!("self.{}", f.name)
        }).collect();
        
        code.push_str(&field_refs.join(", "));
        code.push_str(")\n  }\n}\n");
        
        Some(code)
    }
    
    pub fn generate_derive_clone(&self, struct_name: &str) -> Option<String> {
        let info = self.get_struct(struct_name)?;
        
        let field_clones: Vec<String> = info.fields.iter().map(|f| {
            format!("{}: self.{}.clone()", f.name, f.name)
        }).collect();
        
        let code = format!(
            "impl Clone for {} {{\n  fn clone(&self) -> Self {{\n    Self {{\n      {}\n    }}\n  }}\n}}\n",
            struct_name,
            field_clones.join(",\n      ")
        );
        
        Some(code)
    }
    
    pub fn generate_derive_default(&self, struct_name: &str) -> Option<String> {
        let info = self.get_struct(struct_name)?;
        
        let field_defaults: Vec<String> = info.fields.iter().map(|f| {
            format!("{}: Default::default()", f.name)
        }).collect();
        
        let code = format!(
            "impl Default for {} {{\n  fn default() -> Self {{\n    Self {{\n      {}\n    }}\n  }}\n}}\n",
            struct_name,
            field_defaults.join(",\n      ")
        );
        
        Some(code)
    }
    
    pub fn generate_derive_eq(&self, struct_name: &str) -> Option<String> {
        let info = self.get_struct(struct_name)?;
        
        let comparisons: Vec<String> = info.fields.iter().map(|f| {
            format!("self.{} == other.{}", f.name, f.name)
        }).collect();
        
        let code = format!(
            "impl PartialEq for {} {{\n  fn eq(&self, other: &Self) -> bool {{\n    {}\n  }}\n}}\n",
            struct_name,
            comparisons.join(" &&\n      ")
        );
        
        Some(code)
    }
}

impl Default for ReflectionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn type_to_string(ty: &AstType) -> String {
    match ty {
        AstType::Named(n) => n.clone(),
        AstType::Generic(n, args) => {
            let args_str: Vec<String> = args.iter().map(type_to_string).collect();
            format!("{}<{}>", n, args_str.join(", "))
        }
        AstType::Reference(inner) => format!("&{}", type_to_string(inner)),
        AstType::MutableReference(inner) => format!("&mut {}", type_to_string(inner)),
        AstType::Array(inner, size) => format!("[{}; {}]", type_to_string(inner), size),
        AstType::Tuple(types) => {
            let types_str: Vec<String> = types.iter().map(type_to_string).collect();
            format!("({})", types_str.join(", "))
        }
        AstType::Function(params, ret) => {
            let params_str: Vec<String> = params.iter().map(type_to_string).collect();
            format!("fn({}) -> {}", params_str.join(", "), type_to_string(ret))
        }
        AstType::TraitObject(traits) => traits.join(" + "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ReflectionRegistry::new();
        assert!(registry.structs.is_empty());
    }
}
