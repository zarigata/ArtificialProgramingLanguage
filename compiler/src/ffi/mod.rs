//! Foreign Function Interface (FFI) for C interoperability

use std::collections::HashMap;
use std::fmt;

/// C type representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CType {
    Void,
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    Float,
    Double,
    LongDouble,
    Pointer(Box<CType>),
    ConstPointer(Box<CType>),
    Array(Box<CType>, usize),
    Struct(String, Vec<StructField>),
    Union(String, Vec<StructField>),
    Enum(String, Vec<EnumVariant>),
    Function(Box<CFunctionType>),
    SizeT,
    SSizeT,
    IntPtr,
    UIntPtr,
    PtrDiff,
    WideChar,
    Char16,
    Char32,
}

/// Struct field definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub ctype: CType,
    pub offset: Option<usize>,
}

/// Enum variant definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariant {
    pub name: String,
    pub value: i64,
}

/// C function type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CFunctionType {
    pub return_type: CType,
    pub params: Vec<CParameter>,
    pub variadic: bool,
    pub calling_convention: CallingConvention,
}

/// C function parameter
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CParameter {
    pub name: String,
    pub ctype: CType,
}

/// Calling convention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallingConvention {
    C,
    StdCall,
    FastCall,
    VectorCall,
    SystemV,
    Windows,
}

impl Default for CallingConvention {
    fn default() -> Self {
        CallingConvention::C
    }
}

impl CType {
    /// Get the size of this type in bytes (platform-dependent defaults)
    pub fn size(&self) -> usize {
        match self {
            CType::Void => 0,
            CType::Char | CType::SignedChar | CType::UnsignedChar => 1,
            CType::Short | CType::UnsignedShort => 2,
            CType::Int | CType::UnsignedInt => 4,
            CType::Long | CType::UnsignedLong => 8,
            CType::LongLong | CType::UnsignedLongLong => 8,
            CType::Float => 4,
            CType::Double => 8,
            CType::LongDouble => 16,
            CType::Pointer(_) | CType::ConstPointer(_) => 8,
            CType::Array(inner, count) => inner.size() * count,
            CType::Struct(_, fields) => {
                let mut size = 0;
                let mut max_align = 1;
                for field in fields {
                    let field_size = field.ctype.size();
                    let field_align = field.ctype.align();
                    size = (size + field_align - 1) / field_align * field_align + field_size;
                    max_align = max_align.max(field_align);
                }
                (size + max_align - 1) / max_align * max_align
            }
            CType::Union(_, fields) => {
                let mut max_size = 0;
                let mut max_align = 1;
                for field in fields {
                    let field_size = field.ctype.size();
                    let field_align = field.ctype.align();
                    max_size = max_size.max(field_size);
                    max_align = max_align.max(field_align);
                }
                (max_size + max_align - 1) / max_align * max_align
            }
            CType::Enum(_, _) => 4,
            CType::Function(_) => 8,
            CType::SizeT | CType::SSizeT | CType::UIntPtr | CType::IntPtr => 8,
            CType::PtrDiff => 8,
            CType::WideChar => 4,
            CType::Char16 => 2,
            CType::Char32 => 4,
        }
    }
    
    /// Get the alignment of this type in bytes
    pub fn align(&self) -> usize {
        match self {
            CType::Void => 1,
            CType::Char | CType::SignedChar | CType::UnsignedChar => 1,
            CType::Short | CType::UnsignedShort => 2,
            CType::Int | CType::UnsignedInt => 4,
            CType::Long | CType::UnsignedLong => 8,
            CType::LongLong | CType::UnsignedLongLong => 8,
            CType::Float => 4,
            CType::Double => 8,
            CType::LongDouble => 16,
            CType::Pointer(_) | CType::ConstPointer(_) => 8,
            CType::Array(inner, _) => inner.align(),
            CType::Struct(_, fields) => {
                let mut max_align = 1;
                for field in fields {
                    max_align = max_align.max(field.ctype.align());
                }
                max_align
            }
            CType::Union(_, fields) => {
                let mut max_align = 1;
                for field in fields {
                    max_align = max_align.max(field.ctype.align());
                }
                max_align
            }
            CType::Enum(_, _) => 4,
            CType::Function(_) => 8,
            CType::SizeT | CType::SSizeT | CType::UIntPtr | CType::IntPtr => 8,
            CType::PtrDiff => 8,
            CType::WideChar => 4,
            CType::Char16 => 2,
            CType::Char32 => 4,
        }
    }
    
    /// Check if this type is a primitive
    pub fn is_primitive(&self) -> bool {
        matches!(self, 
            CType::Void | CType::Char | CType::SignedChar | CType::UnsignedChar |
            CType::Short | CType::UnsignedShort | CType::Int | CType::UnsignedInt |
            CType::Long | CType::UnsignedLong | CType::LongLong | CType::UnsignedLongLong |
            CType::Float | CType::Double | CType::LongDouble
        )
    }
    
    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self,
            CType::Char | CType::SignedChar | CType::UnsignedChar |
            CType::Short | CType::UnsignedShort | CType::Int | CType::UnsignedInt |
            CType::Long | CType::UnsignedLong | CType::LongLong | CType::UnsignedLongLong |
            CType::SizeT | CType::SSizeT | CType::IntPtr | CType::UIntPtr | CType::PtrDiff
        )
    }
    
    /// Check if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self, CType::Float | CType::Double | CType::LongDouble)
    }
    
    /// Check if this is a pointer type
    pub fn is_pointer(&self) -> bool {
        matches!(self, CType::Pointer(_) | CType::ConstPointer(_))
    }
    
    /// Convert to VeZ type string
    pub fn to_vez_type(&self) -> String {
        match self {
            CType::Void => "void".to_string(),
            CType::Char => "c_char".to_string(),
            CType::SignedChar => "c_schar".to_string(),
            CType::UnsignedChar => "c_uchar".to_string(),
            CType::Short => "c_short".to_string(),
            CType::UnsignedShort => "c_ushort".to_string(),
            CType::Int => "c_int".to_string(),
            CType::UnsignedInt => "c_uint".to_string(),
            CType::Long => "c_long".to_string(),
            CType::UnsignedLong => "c_ulong".to_string(),
            CType::LongLong => "i64".to_string(),
            CType::UnsignedLongLong => "u64".to_string(),
            CType::Float => "f32".to_string(),
            CType::Double => "f64".to_string(),
            CType::LongDouble => "c_longdouble".to_string(),
            CType::Pointer(inner) => format!("*mut {}", inner.to_vez_type()),
            CType::ConstPointer(inner) => format!("*const {}", inner.to_vez_type()),
            CType::Array(inner, size) => format!("[{}; {}]", inner.to_vez_type(), size),
            CType::Struct(name, _) => name.clone(),
            CType::Union(name, _) => format!("union {}", name),
            CType::Enum(name, _) => name.clone(),
            CType::Function(f) => format!("extern fn({})", f.to_vez_signature()),
            CType::SizeT => "usize".to_string(),
            CType::SSizeT => "isize".to_string(),
            CType::IntPtr => "isize".to_string(),
            CType::UIntPtr => "usize".to_string(),
            CType::PtrDiff => "isize".to_string(),
            CType::WideChar => "c_wchar".to_string(),
            CType::Char16 => "c_char16".to_string(),
            CType::Char32 => "c_char32".to_string(),
        }
    }
}

impl CFunctionType {
    pub fn to_vez_signature(&self) -> String {
        let params: Vec<String> = self.params.iter()
            .map(|p| format!("{}: {}", p.name, p.ctype.to_vez_type()))
            .collect();
        
        let mut sig = params.join(", ");
        
        if self.variadic {
            if !sig.is_empty() {
                sig.push_str(", ");
            }
            sig.push_str("...");
        }
        
        sig.push_str(&format!(" -> {}", self.return_type.to_vez_type()));
        sig
    }
}

impl fmt::Display for CType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CType::Void => write!(f, "void"),
            CType::Char => write!(f, "char"),
            CType::SignedChar => write!(f, "signed char"),
            CType::UnsignedChar => write!(f, "unsigned char"),
            CType::Short => write!(f, "short"),
            CType::UnsignedShort => write!(f, "unsigned short"),
            CType::Int => write!(f, "int"),
            CType::UnsignedInt => write!(f, "unsigned int"),
            CType::Long => write!(f, "long"),
            CType::UnsignedLong => write!(f, "unsigned long"),
            CType::LongLong => write!(f, "long long"),
            CType::UnsignedLongLong => write!(f, "unsigned long long"),
            CType::Float => write!(f, "float"),
            CType::Double => write!(f, "double"),
            CType::LongDouble => write!(f, "long double"),
            CType::Pointer(inner) => write!(f, "{} *", inner),
            CType::ConstPointer(inner) => write!(f, "const {} *", inner),
            CType::Array(inner, size) => write!(f, "{}[{}]", inner, size),
            CType::Struct(name, _) => write!(f, "struct {}", name),
            CType::Union(name, _) => write!(f, "union {}", name),
            CType::Enum(name, _) => write!(f, "enum {}", name),
            CType::Function(func) => write!(f, "{}", func),
            CType::SizeT => write!(f, "size_t"),
            CType::SSizeT => write!(f, "ssize_t"),
            CType::IntPtr => write!(f, "intptr_t"),
            CType::UIntPtr => write!(f, "uintptr_t"),
            CType::PtrDiff => write!(f, "ptrdiff_t"),
            CType::WideChar => write!(f, "wchar_t"),
            CType::Char16 => write!(f, "char16_t"),
            CType::Char32 => write!(f, "char32_t"),
        }
    }
}

impl fmt::Display for CFunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: Vec<String> = self.params.iter()
            .map(|p| format!("{} {}", p.ctype, p.name))
            .collect();
        
        write!(f, "{} ({})", self.return_type, params.join(", "))
    }
}

/// FFI function declaration
#[derive(Debug, Clone)]
pub struct FFIFunction {
    pub name: String,
    pub c_name: String,
    pub signature: CFunctionType,
    pub library: Option<String>,
    pub documentation: Option<String>,
}

/// FFI struct declaration
#[derive(Debug, Clone)]
pub struct FFIStruct {
    pub name: String,
    pub c_name: String,
    pub fields: Vec<StructField>,
    pub packed: bool,
    pub documentation: Option<String>,
}

/// FFI enum declaration
#[derive(Debug, Clone)]
pub struct FFIEnum {
    pub name: String,
    pub c_name: String,
    pub variants: Vec<EnumVariant>,
    pub documentation: Option<String>,
}

/// FFI constant declaration
#[derive(Debug, Clone)]
pub struct FFIConstant {
    pub name: String,
    pub c_name: String,
    pub ctype: CType,
    pub value: String,
}

/// FFI callback type
#[derive(Debug, Clone)]
pub struct FFICallback {
    pub name: String,
    pub signature: CFunctionType,
}

/// FFI binding module
#[derive(Debug, Clone, Default)]
pub struct FFIBindings {
    pub functions: Vec<FFIFunction>,
    pub structs: Vec<FFIStruct>,
    pub enums: Vec<FFIEnum>,
    pub constants: Vec<FFIConstant>,
    pub callbacks: Vec<FFICallback>,
    pub libraries: Vec<String>,
}

impl FFIBindings {
    pub fn new() -> Self {
        FFIBindings::default()
    }
    
    pub fn add_function(&mut self, func: FFIFunction) {
        if let Some(lib) = &func.library {
            if !self.libraries.contains(lib) {
                self.libraries.push(lib.clone());
            }
        }
        self.functions.push(func);
    }
    
    pub fn add_struct(&mut self, s: FFIStruct) {
        self.structs.push(s);
    }
    
    pub fn add_enum(&mut self, e: FFIEnum) {
        self.enums.push(e);
    }
    
    pub fn add_constant(&mut self, c: FFIConstant) {
        self.constants.push(c);
    }
    
    pub fn add_callback(&mut self, cb: FFICallback) {
        self.callbacks.push(cb);
    }
    
    /// Generate VeZ FFI bindings
    pub fn generate_vez(&self) -> String {
        let mut output = String::new();
        
        output.push_str("// Auto-generated FFI bindings\n\n");
        
        // External libraries
        for lib in &self.libraries {
            output.push_str(&format!("#[link(name = \"{}\")]\n", lib));
        }
        output.push_str("extern \"C\" {\n");
        
        // Functions
        for func in &self.functions {
            if let Some(doc) = &func.documentation {
                output.push_str(&format!("    /// {}\n", doc));
            }
            output.push_str(&format!("    pub fn {}(", func.name));
            
            let params: Vec<String> = func.signature.params.iter()
                .map(|p| format!("{}: {}", p.name, p.ctype.to_vez_type()))
                .collect();
            
            output.push_str(&params.join(", "));
            output.push_str(&format!(") -> {};\n\n", func.signature.return_type.to_vez_type()));
        }
        
        output.push_str("}\n\n");
        
        // Structs
        for s in &self.structs {
            if let Some(doc) = &s.documentation {
                output.push_str(&format!("/// {}\n", doc));
            }
            
            let repr = if s.packed { "packed" } else { "C" };
            output.push_str(&format!("#[repr({})]\n", repr));
            output.push_str(&format!("pub struct {} {{\n", s.name));
            
            for field in &s.fields {
                output.push_str(&format!("    pub {}: {},\n", field.name, field.ctype.to_vez_type()));
            }
            
            output.push_str("}\n\n");
        }
        
        // Enums
        for e in &self.enums {
            if let Some(doc) = &e.documentation {
                output.push_str(&format!("/// {}\n", doc));
            }
            output.push_str("#[repr(C)]\n");
            output.push_str(&format!("pub enum {} {{\n", e.name));
            
            for variant in &e.variants {
                output.push_str(&format!("    {} = {},\n", variant.name, variant.value));
            }
            
            output.push_str("}\n\n");
        }
        
        // Constants
        for c in &self.constants {
            output.push_str(&format!("pub const {}: {} = {};\n", c.name, c.ctype.to_vez_type(), c.value));
        }
        
        output
    }
    
    /// Generate C header file
    pub fn generate_header(&self) -> String {
        let mut output = String::new();
        
        output.push_str("/* Auto-generated C header */\n\n");
        output.push_str("#ifndef VEZ_FFI_H\n");
        output.push_str("#define VEZ_FFI_H\n\n");
        output.push_str("#include <stdint.h>\n#include <stddef.h>\n\n");
        
        // Forward declarations
        output.push_str("/* Forward declarations */\n");
        for s in &self.structs {
            output.push_str(&format!("struct {};\n", s.c_name));
        }
        for e in &self.enums {
            output.push_str(&format!("enum {};\n", e.c_name));
        }
        output.push_str("\n");
        
        // Enums
        for e in &self.enums {
            output.push_str(&format!("enum {} {{\n", e.c_name));
            for variant in &e.variants {
                output.push_str(&format!("    {} = {},\n", variant.name, variant.value));
            }
            output.push_str("};\n\n");
        }
        
        // Structs
        for s in &self.structs {
            output.push_str(&format!("struct {} {{\n", s.c_name));
            for field in &s.fields {
                output.push_str(&format!("    {} {};\n", field.ctype, field.name));
            }
            output.push_str("};\n\n");
        }
        
        // Function declarations
        output.push_str("/* Function declarations */\n");
        for func in &self.functions {
            let params: Vec<String> = func.signature.params.iter()
                .map(|p| format!("{} {}", p.ctype, p.name))
                .collect();
            
            output.push_str(&format!("{} {}({});\n", 
                func.signature.return_type, 
                func.c_name, 
                params.join(", ")
            ));
        }
        output.push_str("\n");
        
        // Constants
        output.push_str("/* Constants */\n");
        for c in &self.constants {
            output.push_str(&format!("#define {} ({})\n", c.c_name, c.value));
        }
        output.push_str("\n");
        
        output.push_str("#endif /* VEZ_FFI_H */\n");
        output
    }
}

/// C header parser
pub struct HeaderParser {
    definitions: FFIBindings,
    type_aliases: HashMap<String, CType>,
}

impl HeaderParser {
    pub fn new() -> Self {
        let mut parser = HeaderParser {
            definitions: FFIBindings::new(),
            type_aliases: HashMap::new(),
        };
        parser.init_standard_types();
        parser
    }
    
    fn init_standard_types(&mut self) {
        let standard_types = [
            ("size_t", CType::SizeT),
            ("ssize_t", CType::SSizeT),
            ("intptr_t", CType::IntPtr),
            ("uintptr_t", CType::UIntPtr),
            ("ptrdiff_t", CType::PtrDiff),
            ("wchar_t", CType::WideChar),
            ("char16_t", CType::Char16),
            ("char32_t", CType::Char32),
            ("int8_t", CType::SignedChar),
            ("uint8_t", CType::UnsignedChar),
            ("int16_t", CType::Short),
            ("uint16_t", CType::UnsignedShort),
            ("int32_t", CType::Int),
            ("uint32_t", CType::UnsignedInt),
            ("int64_t", CType::LongLong),
            ("uint64_t", CType::UnsignedLongLong),
        ];
        
        for (name, ctype) in standard_types {
            self.type_aliases.insert(name.to_string(), ctype);
        }
    }
    
    /// Parse a C header file
    pub fn parse(&mut self, content: &str) -> Result<FFIBindings, String> {
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;
        
        while i < lines.len() {
            let line = lines[i].trim();
            
            if line.starts_with("//") || line.starts_with("/*") || line.is_empty() {
                i += 1;
                continue;
            }
            
            if line.starts_with("typedef") {
                self.parse_typedef(line)?;
            } else if line.starts_with("struct") {
                let (def, consumed) = self.parse_struct(&lines[i..])?;
                self.definitions.add_struct(def);
                i += consumed;
                continue;
            } else if line.starts_with("enum") {
                let (def, consumed) = self.parse_enum(&lines[i..])?;
                self.definitions.add_enum(def);
                i += consumed;
                continue;
            } else if line.starts_with("#define") {
                if let Some(c) = self.parse_define(line)? {
                    self.definitions.add_constant(c);
                }
            }
            
            i += 1;
        }
        
        Ok(self.definitions.clone())
    }
    
    fn parse_typedef(&mut self, line: &str) -> Result<(), String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() < 3 {
            return Err(format!("Invalid typedef: {}", line));
        }
        
        let new_name = parts.last().unwrap().trim_end_matches(';');
        let old_type = parts[1..parts.len()-1].join(" ");
        
        let ctype = self.parse_type_str(&old_type)?;
        self.type_aliases.insert(new_name.to_string(), ctype);
        
        Ok(())
    }
    
    fn parse_struct(&self, lines: &[&str]) -> Result<(FFIStruct, usize), String> {
        let first = lines[0].trim();
        let name = first.split_whitespace()
            .nth(1)
            .unwrap_or("anonymous")
            .trim_end_matches('{')
            .to_string();
        
        let mut fields = Vec::new();
        let mut i = 1;
        
        while i < lines.len() && !lines[i].contains('}') {
            let line = lines[i].trim();
            
            if !line.is_empty() && !line.starts_with("//") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let field_name = parts.last().unwrap().trim_end_matches(';').to_string();
                    let type_str = parts[..parts.len()-1].join(" ");
                    let ctype = self.parse_type_str(&type_str)?;
                    
                    fields.push(StructField {
                        name: field_name,
                        ctype,
                        offset: None,
                    });
                }
            }
            i += 1;
        }
        
        i += 1;
        
        Ok((FFIStruct {
            name: name.clone(),
            c_name: name,
            fields,
            packed: false,
            documentation: None,
        }, i))
    }
    
    fn parse_enum(&self, lines: &[&str]) -> Result<(FFIEnum, usize), String> {
        let first = lines[0].trim();
        let name = first.split_whitespace()
            .nth(1)
            .unwrap_or("anonymous")
            .trim_end_matches('{')
            .to_string();
        
        let mut variants = Vec::new();
        let mut i = 1;
        let mut value: i64 = 0;
        
        while i < lines.len() && !lines[i].contains('}') {
            let line = lines[i].trim();
            
            if !line.is_empty() && !line.starts_with("//") {
                let line = line.trim_end_matches(',');
                
                if line.contains('=') {
                    let parts: Vec<&str> = line.split('=').collect();
                    let variant_name = parts[0].trim();
                    if let Ok(v) = parts[1].trim().parse::<i64>() {
                        value = v;
                    }
                    variants.push(EnumVariant {
                        name: variant_name.to_string(),
                        value,
                    });
                } else {
                    variants.push(EnumVariant {
                        name: line.to_string(),
                        value,
                    });
                }
                value += 1;
            }
            i += 1;
        }
        
        i += 1;
        
        Ok((FFIEnum {
            name: name.clone(),
            c_name: name,
            variants,
            documentation: None,
        }, i))
    }
    
    fn parse_define(&self, line: &str) -> Result<Option<FFIConstant>, String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() < 3 {
            return Ok(None);
        }
        
        let name = parts[1].to_string();
        let value = parts[2..].join(" ");
        
        let ctype = if value.starts_with('"') {
            CType::Pointer(Box::new(CType::Char))
        } else if value.contains('.') {
            CType::Double
        } else {
            CType::Int
        };
        
        Ok(Some(FFIConstant {
            name: name.clone(),
            c_name: name,
            ctype,
            value,
        }))
    }
    
    fn parse_type_str(&self, s: &str) -> Result<CType, String> {
        let s = s.trim();
        
        if let Some(ctype) = self.type_aliases.get(s) {
            return Ok(ctype.clone());
        }
        
        match s {
            "void" => Ok(CType::Void),
            "char" => Ok(CType::Char),
            "signed char" => Ok(CType::SignedChar),
            "unsigned char" => Ok(CType::UnsignedChar),
            "short" | "short int" => Ok(CType::Short),
            "unsigned short" | "unsigned short int" => Ok(CType::UnsignedShort),
            "int" | "signed" | "signed int" => Ok(CType::Int),
            "unsigned" | "unsigned int" => Ok(CType::UnsignedInt),
            "long" | "long int" => Ok(CType::Long),
            "unsigned long" | "unsigned long int" => Ok(CType::UnsignedLong),
            "long long" | "long long int" => Ok(CType::LongLong),
            "unsigned long long" | "unsigned long long int" => Ok(CType::UnsignedLongLong),
            "float" => Ok(CType::Float),
            "double" => Ok(CType::Double),
            "long double" => Ok(CType::LongDouble),
            _ => {
                if s.ends_with('*') {
                    let inner = s.trim_end_matches('*').trim();
                    let inner_type = self.parse_type_str(inner)?;
                    if s.starts_with("const") {
                        Ok(CType::ConstPointer(Box::new(inner_type)))
                    } else {
                        Ok(CType::Pointer(Box::new(inner_type)))
                    }
                } else {
                    Ok(CType::Struct(s.to_string(), Vec::new()))
                }
            }
        }
    }
}

impl Default for HeaderParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ctype_size() {
        assert_eq!(CType::Char.size(), 1);
        assert_eq!(CType::Int.size(), 4);
        assert_eq!(CType::Pointer(Box::new(CType::Void)).size(), 8);
    }
    
    #[test]
    fn test_ctype_to_vez() {
        assert_eq!(CType::Int.to_vez_type(), "c_int");
        assert_eq!(CType::Pointer(Box::new(CType::Char)).to_vez_type(), "*mut c_char");
    }
    
    #[test]
    fn test_ffi_bindings() {
        let mut bindings = FFIBindings::new();
        
        bindings.add_function(FFIFunction {
            name: "puts".to_string(),
            c_name: "puts".to_string(),
            signature: CFunctionType {
                return_type: CType::Int,
                params: vec![CParameter {
                    name: "s".to_string(),
                    ctype: CType::ConstPointer(Box::new(CType::Char)),
                }],
                variadic: false,
                calling_convention: CallingConvention::C,
            },
            library: Some("c".to_string()),
            documentation: Some("Write string to stdout"),
        });
        
        let code = bindings.generate_vez();
        assert!(code.contains("pub fn puts"));
    }
    
    #[test]
    fn test_header_parser() {
        let mut parser = HeaderParser::new();
        
        let header = r#"
            typedef int myint;
            
            struct Point {
                int x;
                int y;
            };
            
            enum Color {
                Red,
                Green,
                Blue
            };
        "#;
        
        let bindings = parser.parse(header).unwrap();
        
        assert!(!bindings.structs.is_empty());
        assert!(!bindings.enums.is_empty());
    }
}
