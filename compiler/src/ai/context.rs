//! Context extraction for AI code generation

use std::collections::HashMap;

/// Position in source code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

/// Semantic context extracted for AI code generation
#[derive(Debug, Clone)]
pub struct AiContext {
    pub signatures: Vec<FunctionSignature>,
    pub types: Vec<TypeDefinition>,
    pub imports: Vec<ImportInfo>,
    pub scope: ScopeContext,
    pub suggestions: Vec<CompletionSuggestion>,
}

/// Function signature for AI understanding
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub return_type: String,
    pub doc: Option<String>,
    pub visibility: Visibility,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub annotations: Vec<String>,
}

/// Type definition for AI understanding
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub name: String,
    pub kind: TypeKind,
    pub fields: Vec<FieldInfo>,
    pub methods: Vec<FunctionSignature>,
    pub generics: Vec<String>,
    pub doc: Option<String>,
}

/// Kind of type
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Struct,
    Enum,
    Trait,
    TypeAlias,
}

/// Field information
#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub ty: String,
    pub visibility: Visibility,
}

/// Visibility level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Visibility {
    Public,
    Private,
    Crate,
    Super,
}

/// Import information
#[derive(Debug, Clone)]
pub struct ImportInfo {
    pub module: String,
    pub items: Vec<String>,
    pub alias: Option<String>,
}

/// Scope context at cursor position
#[derive(Debug, Clone, Default)]
pub struct ScopeContext {
    pub local_vars: Vec<VarInfo>,
    pub in_function: Option<String>,
    pub in_struct: Option<String>,
    pub in_impl: Option<String>,
    pub indent_level: usize,
}

/// Variable information
#[derive(Debug, Clone)]
pub struct VarInfo {
    pub name: String,
    pub ty: String,
    pub mutable: bool,
}

/// Completion suggestion
#[derive(Debug, Clone)]
pub struct CompletionSuggestion {
    pub text: String,
    pub kind: CompletionKind,
    pub detail: Option<String>,
    pub score: f32,
}

/// Kind of completion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompletionKind {
    Function,
    Variable,
    Type,
    Keyword,
    Module,
    Field,
    Method,
}

impl AiContext {
    pub fn new() -> Self {
        AiContext {
            signatures: Vec::new(),
            types: Vec::new(),
            imports: Vec::new(),
            scope: ScopeContext::default(),
            suggestions: Vec::new(),
        }
    }

    pub fn to_prompt_context(&self) -> String {
        let mut ctx = String::new();
        
        if !self.imports.is_empty() {
            ctx.push_str("Available imports:\n");
            for imp in &self.imports {
                ctx.push_str(&format!("  use {}::{{{}}}\n", imp.module, imp.items.join(", ")));
            }
            ctx.push('\n');
        }
        
        if !self.types.is_empty() {
            ctx.push_str("Available types:\n");
            for ty in &self.types {
                ctx.push_str(&format!("  {} {}\n", 
                    match ty.kind {
                        TypeKind::Struct => "struct",
                        TypeKind::Enum => "enum",
                        TypeKind::Trait => "trait",
                        TypeKind::TypeAlias => "type",
                    },
                    ty.name
                ));
                if !ty.fields.is_empty() {
                    for f in &ty.fields {
                        ctx.push_str(&format!("    {}: {}\n", f.name, f.ty));
                    }
                }
            }
            ctx.push('\n');
        }
        
        if !self.signatures.is_empty() {
            ctx.push_str("Available functions:\n");
            for sig in &self.signatures {
                let params: Vec<String> = sig.params.iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect();
                ctx.push_str(&format!("  fn {}({}) -> {}\n", 
                    sig.name, 
                    params.join(", "),
                    sig.return_type
                ));
            }
        }
        
        ctx
    }
}

/// Extract AI context from source at position
pub fn extract_context(source: &str, cursor: Position) -> AiContext {
    let mut ctx = AiContext::new();
    
    let lines: Vec<&str> = source.lines().collect();
    let line_before = if cursor.line > 0 { lines.get(cursor.line - 1) } else { None };
    let current_line = lines.get(cursor.line).unwrap_or(&"");
    
    ctx.scope.indent_level = current_line.chars().take_while(|c| c.is_whitespace()).count();
    
    if let Some(prev) = line_before {
        if prev.trim().starts_with("fn ") || prev.trim().starts_with("def ") {
            if let Some(name) = extract_fn_name(prev) {
                ctx.scope.in_function = Some(name);
            }
        }
    }
    
    let mut in_function = false;
    for line in &lines[..cursor.line.min(lines.len())] {
        let trimmed = line.trim();
        
        if trimmed.starts_with("use ") || trimmed.starts_with("import ") {
            ctx.imports.push(parse_import(trimmed));
        }
        
        if trimmed.starts_with("fn ") || trimmed.starts_with("def ") {
            if let Some(sig) = parse_function_signature(trimmed) {
                ctx.signatures.push(sig);
            }
            in_function = true;
        }
        
        if trimmed.starts_with("struct ") {
            if let Some(ty) = parse_struct_def(trimmed) {
                ctx.types.push(ty);
            }
        }
        
        if trimmed.starts_with("let ") || trimmed.starts_with("var ") {
            if let Some(var) = parse_var_def(trimmed) {
                ctx.scope.local_vars.push(var);
            }
        }
    }
    
    ctx.suggestions = generate_suggestions(&ctx, current_line);
    
    ctx
}

fn extract_fn_name(line: &str) -> Option<String> {
    let line = line.trim();
    let start = if line.starts_with("fn ") { 3 } 
                else if line.starts_with("def ") { 4 } 
                else { return None };
    
    let rest = &line[start..];
    let end = rest.find('(').unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn parse_import(line: &str) -> ImportInfo {
    let line = line.trim();
    let line = line.strip_prefix("use ").unwrap_or(line);
    let line = line.strip_prefix("import ").unwrap_or(line);
    let line = line.trim_end_matches(';');
    
    if let Some(pos) = line.find("::") {
        let module = line[..pos].to_string();
        let items_str = &line[pos + 2..];
        let items: Vec<String> = items_str
            .trim_matches(|c| c == '{' || c == '}')
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        ImportInfo { module, items, alias: None }
    } else {
        ImportInfo {
            module: line.to_string(),
            items: vec!["*".to_string()],
            alias: None,
        }
    }
}

fn parse_function_signature(line: &str) -> Option<FunctionSignature> {
    let line = line.trim();
    let (prefix, rest) = if line.starts_with("fn ") { ("fn ", &line[3..]) }
                         else if line.starts_with("def ") { ("def ", &line[4..]) }
                         else { return None };
    
    let name_end = rest.find('(')?;
    let name = rest[..name_end].trim().to_string();
    
    let params_start = name_end;
    let params_end = rest.find(')').unwrap_or(rest.len());
    let params_str = &rest[params_start + 1..params_end];
    
    let params: Vec<(String, String)> = params_str
        .split(',')
        .filter_map(|p| {
            let parts: Vec<&str> = p.trim().split(':').collect();
            if parts.len() == 2 {
                Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
            } else {
                None
            }
        })
        .collect();
    
    let return_type = if let Some(pos) = rest.find("->") {
        rest[pos + 2..].split('{').next().unwrap_or("()").trim().to_string()
    } else {
        "()".to_string()
    };
    
    Some(FunctionSignature {
        name,
        params,
        return_type,
        doc: None,
        visibility: Visibility::Public,
        is_async: prefix.contains("async"),
        is_unsafe: prefix.contains("unsafe"),
        annotations: Vec::new(),
    })
}

fn parse_struct_def(line: &str) -> Option<TypeDefinition> {
    let line = line.trim();
    let rest = line.strip_prefix("struct ")?;
    let name = rest.split(':').next()?.trim().to_string();
    
    Some(TypeDefinition {
        name,
        kind: TypeKind::Struct,
        fields: Vec::new(),
        methods: Vec::new(),
        generics: Vec::new(),
        doc: None,
    })
}

fn parse_var_def(line: &str) -> Option<VarInfo> {
    let line = line.trim();
    let rest = line.strip_prefix("let ")?;
    let rest = rest.strip_prefix("var ")?;
    
    let mutable = rest.starts_with("mut ");
    let rest = if mutable { &rest[4..] } else { rest };
    
    let parts: Vec<&str> = rest.split(':').collect();
    if parts.len() >= 2 {
        Some(VarInfo {
            name: parts[0].trim().to_string(),
            ty: parts[1].split('=').next()?.trim().to_string(),
            mutable,
        })
    } else {
        let name = rest.split('=').next()?.trim().to_string();
        Some(VarInfo {
            name,
            ty: "unknown".to_string(),
            mutable,
        })
    }
}

fn generate_suggestions(ctx: &AiContext, current_line: &str) -> Vec<CompletionSuggestion> {
    let mut suggestions = Vec::new();
    let prefix = extract_word_before_cursor(current_line);
    
    for sig in &ctx.signatures {
        if sig.name.starts_with(&prefix) || prefix.is_empty() {
            suggestions.push(CompletionSuggestion {
                text: sig.name.clone(),
                kind: CompletionKind::Function,
                detail: Some(format!("fn({}) -> {}", 
                    sig.params.iter().map(|(n, t)| format!("{}: {}", n, t)).collect::<Vec<_>>().join(", "),
                    sig.return_type
                )),
                score: 0.9,
            });
        }
    }
    
    for var in &ctx.scope.local_vars {
        if var.name.starts_with(&prefix) || prefix.is_empty() {
            suggestions.push(CompletionSuggestion {
                text: var.name.clone(),
                kind: CompletionKind::Variable,
                detail: Some(var.ty.clone()),
                score: 0.8,
            });
        }
    }
    
    for ty in &ctx.types {
        if ty.name.starts_with(&prefix) || prefix.is_empty() {
            suggestions.push(CompletionSuggestion {
                text: ty.name.clone(),
                kind: CompletionKind::Type,
                detail: Some(format!("{:?}", ty.kind)),
                score: 0.85,
            });
        }
    }
    
    suggestions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    suggestions.truncate(20);
    
    suggestions
}

fn extract_word_before_cursor(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut word = String::new();
    
    for c in chars.iter().rev() {
        if c.is_alphanumeric() || *c == '_' {
            word.insert(0, *c);
        } else {
            break;
        }
    }
    
    word
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_context() {
        let source = r#"
use std::collections::Vec

struct Point:
    x: f64
    y: f64

fn distance(p1: Point, p2: Point) -> f64:
    let dx = p1.x - p2.x
    let dy = p1.y - p2.y
    return sqrt(dx * dx + dy * dy)
"#;
        
        let ctx = extract_context(source, Position { line: 10, column: 0 });
        
        assert!(!ctx.imports.is_empty());
        assert!(!ctx.types.is_empty());
        assert!(!ctx.signatures.is_empty());
    }
    
    #[test]
    fn test_parse_function_signature() {
        let sig = parse_function_signature("fn add(a: i32, b: i32) -> i32").unwrap();
        assert_eq!(sig.name, "add");
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.return_type, "i32");
    }
    
    #[test]
    fn test_to_prompt_context() {
        let mut ctx = AiContext::new();
        ctx.signatures.push(FunctionSignature {
            name: "foo".to_string(),
            params: vec![("x".to_string(), "i32".to_string())],
            return_type: "i32".to_string(),
            doc: None,
            visibility: Visibility::Public,
            is_async: false,
            is_unsafe: false,
            annotations: vec![],
        });
        
        let prompt = ctx.to_prompt_context();
        assert!(prompt.contains("foo"));
    }
}
