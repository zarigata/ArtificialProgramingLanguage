//! Enhanced error diagnostics with intelligent suggestions
//!
//! This module provides advanced error diagnostics including:
//! - Spell checking and identifier suggestions
//! - Type mismatch suggestions with common fixes
//! - Import suggestions for missing modules
//! - Pattern matching suggestions
//! - Error recovery helpers

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Suggestion engine for providing intelligent fixes
pub struct SuggestionEngine {
    /// Known identifiers in scope
    identifiers: HashSet<String>,
    /// Known types
    types: HashSet<String>,
    /// Known functions with their signatures
    functions: HashMap<String, FunctionSignature>,
    /// Known modules
    modules: HashSet<String>,
    /// Common mistakes and their corrections
    common_mistakes: HashMap<String, String>,
    /// Maximum edit distance for suggestions
    max_distance: usize,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<(String, String)>, // (name, type)
    pub return_type: Option<String>,
}

impl SuggestionEngine {
    pub fn new() -> Self {
        let mut engine = SuggestionEngine {
            identifiers: HashSet::new(),
            types: HashSet::new(),
            functions: HashMap::new(),
            modules: HashSet::new(),
            common_mistakes: HashMap::new(),
            max_distance: 2,
        };
        
        engine.init_builtin_types();
        engine.init_builtin_functions();
        engine.init_common_mistakes();
        engine
    }
    
    fn init_builtin_types(&mut self) {
        let types = [
            // Primitive types
            "int", "i8", "i16", "i32", "i64", "i128", "isize",
            "uint", "u8", "u16", "u32", "u64", "u128", "usize",
            "float", "f32", "f64",
            "bool", "char", "string", "str",
            // Common types
            "Option", "Result", "Vec", "String",
            "Box", "Rc", "Arc",
            "Cell", "RefCell",
            "HashMap", "HashSet", "BTreeMap", "BTreeSet",
            "LinkedList", "VecDeque",
            "Duration", "DateTime", "Date", "Time",
            "Path", "PathBuf",
            "Cow", "Mutex", "RwLock",
            // AI-native types
            "Tensor", "NdArray", "GpuBuffer",
            "Region", "Scope",
        ];
        
        for t in types {
            self.types.insert(t.to_string());
        }
    }
    
    fn init_builtin_functions(&mut self) {
        let functions = [
            ("print", vec![("value", "any")], None),
            ("println", vec![("value", "any")], None),
            ("format", vec![("template", "string"), ("args", "...")], Some("string")),
            ("len", vec![("collection", "Collection")], Some("int")),
            ("range", vec![("start", "int"), ("end", "int")], Some("Range")),
            ("min", vec![("a", "T"), ("b", "T")], Some("T")),
            ("max", vec![("a", "T"), ("b", "T")], Some("T")),
            ("abs", vec![("x", "number")], Some("number")),
            ("sqrt", vec![("x", "float")], Some("float")),
            ("pow", vec![("base", "float"), ("exp", "float")], Some("float")),
            ("sin", vec![("x", "float")], Some("float")),
            ("cos", vec![("x", "float")], Some("float")),
            ("tan", vec![("x", "float")], Some("float")),
            ("log", vec![("x", "float")], Some("float")),
            ("exp", vec![("x", "float")], Some("float")),
            ("floor", vec![("x", "float")], Some("int")),
            ("ceil", vec![("x", "float")], Some("int")),
            ("round", vec![("x", "float")], Some("int")),
            ("to_string", vec![("value", "T")], Some("string")),
            ("to_int", vec![("s", "string")], Some("int")),
            ("to_float", vec![("s", "string")], Some("float")),
            ("parse", vec![("s", "string")], Some("T")),
            ("clone", vec![("value", "T")], Some("T")),
            ("copy", vec![("value", "T")], Some("T")),
            ("default", vec![], Some("T")),
            ("hash", vec![("value", "T")], Some("u64")),
            ("type_name", vec![("value", "T")], Some("string")),
        ];
        
        for (name, params, ret) in functions {
            let sig = FunctionSignature {
                name: name.to_string(),
                params: params.iter()
                    .map(|(n, t)| (n.to_string(), t.to_string()))
                    .collect(),
                return_type: ret.map(|r| r.to_string()),
            };
            self.functions.insert(name.to_string(), sig);
        }
    }
    
    fn init_common_mistakes(&mut self) {
        let mistakes = [
            // Typos
            ("prnt", "print"),
            ("prinf", "print"),
            ("prntln", "println"),
            ("lenght", "length"),
            ("lengh", "length"),
            ("legnth", "length"),
            ("retrun", "return"),
            ("functon", "function"),
            ("funciton", "function"),
            ("funtion", "function"),
            ("functin", "function"),
            ("defalt", "default"),
            ("defualt", "default"),
            ("conts", "const"),
            ("conts", "const"),
            ("imoprt", "import"),
            ("improt", "import"),
            ("inlcude", "include"),
            ("publc", "public"),
            ("pbulic", "public"),
            ("priavte", "private"),
            ("structrue", "struct"),
            ("struc", "struct"),
            ("strcut", "struct"),
            ("calss", "class"),
            ("clas", "class"),
            ("enmu", "enum"),
            ("enun", "enum"),
            ("iterface", "interface"),
            ("inteface", "interface"),
            ("impliment", "implement"),
            ("implemnt", "implement"),
            ("exteds", "extends"),
            ("extneds", "extends"),
            ("impelments", "implements"),
            
            // Common type confusion
            ("String", "string"),
            ("Int", "int"),
            ("Float", "float"),
            ("Bool", "bool"),
            
            // Method name typos
            ("toStirng", "to_string"),
            ("toStrnig", "to_string"),
            ("to_strng", "to_string"),
            ("pust", "push"),
            ("pusth", "push"),
            ("poop", "pop"),
            ("remaove", "remove"),
            ("remoev", "remove"),
            ("inseret", "insert"),
            ("insret", "insert"),
            ("appned", "append"),
            ("apend", "append"),
            ("conatins", "contains"),
            ("contians", "contains"),
            ("lcoate", "locate"),
            ("lcoate", "locate"),
            ("fnd", "find"),
            ("fiilter", "filter"),
            ("flter", "filter"),
            ("maping", "mapping"),
            ("mappnig", "mapping"),
            ("redcuce", "reduce"),
            ("redue", "reduce"),
            ("sorrt", "sort"),
            ("srot", "sort"),
            ("reveerse", "reverse"),
            ("revese", "reverse"),
        ];
        
        for (wrong, correct) in mistakes {
            self.common_mistakes.insert(wrong.to_string(), correct.to_string());
        }
    }
    
    /// Add an identifier to the known set
    pub fn add_identifier(&mut self, name: &str) {
        self.identifiers.insert(name.to_string());
    }
    
    /// Add a type to the known set
    pub fn add_type(&mut self, name: &str) {
        self.types.insert(name.to_string());
    }
    
    /// Add a function signature
    pub fn add_function(&mut self, sig: FunctionSignature) {
        self.functions.insert(sig.name.clone(), sig);
    }
    
    /// Add a module to the known set
    pub fn add_module(&mut self, name: &str) {
        self.modules.insert(name.to_string());
    }
    
    /// Get suggestions for an undefined identifier
    pub fn suggest_identifier(&self, name: &str) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        
        // Check common mistakes first
        if let Some(correct) = self.common_mistakes.get(name) {
            suggestions.push(Suggestion {
                message: format!("did you mean `{}`?", correct),
                replacement: Some(correct.clone()),
                confidence: 0.95,
                kind: SuggestionKind::Typo,
            });
        }
        
        // Check case-insensitive match
        let name_lower = name.to_lowercase();
        for id in &self.identifiers {
            if id.to_lowercase() == name_lower && id != name {
                suggestions.push(Suggestion {
                    message: format!("did you mean `{}`? (note: names are case-sensitive)", id),
                    replacement: Some(id.clone()),
                    confidence: 0.9,
                    kind: SuggestionKind::CaseMismatch,
                });
            }
        }
        
        // Find similar identifiers using edit distance
        for id in &self.identifiers {
            let dist = levenshtein_distance(name, id);
            if dist > 0 && dist <= self.max_distance {
                suggestions.push(Suggestion {
                    message: format!("did you mean `{}`?", id),
                    replacement: Some(id.clone()),
                    confidence: 1.0 - (dist as f64 / (name.len().max(id.len()) as f64)),
                    kind: SuggestionKind::SimilarName,
                });
            }
        }
        
        // Sort by confidence and limit results
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        suggestions.truncate(3);
        
        suggestions
    }
    
    /// Get suggestions for an undefined type
    pub fn suggest_type(&self, name: &str) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        
        // Check common type mistakes
        let type_corrections = [
            ("String", ("string", "primitive type `string`")),
            ("Int", ("int", "primitive type `int`")),
            ("Float", ("float", "primitive type `float`")),
            ("Bool", ("bool", "primitive type `bool`")),
            ("str", ("string", "owned string type `string`")),
        ];
        
        for (wrong, (correct, explanation)) in type_corrections {
            if name == wrong {
                suggestions.push(Suggestion {
                    message: format!("use `{}` instead ({})", correct, explanation),
                    replacement: Some(correct.to_string()),
                    confidence: 0.95,
                    kind: SuggestionKind::TypeConvention,
                });
            }
        }
        
        // Find similar types
        for t in &self.types {
            let dist = levenshtein_distance(name, t);
            if dist > 0 && dist <= self.max_distance {
                suggestions.push(Suggestion {
                    message: format!("did you mean `{}`?", t),
                    replacement: Some(t.clone()),
                    confidence: 1.0 - (dist as f64 / (name.len().max(t.len()) as f64)),
                    kind: SuggestionKind::SimilarType,
                });
            }
        }
        
        // Suggest importing if it might be from a module
        for module in &self.modules {
            suggestions.push(Suggestion {
                message: format!("if `{}` is a type from module `{}`, try importing it", name, module),
                replacement: Some(format!("use {}::{};", module, name)),
                confidence: 0.5,
                kind: SuggestionKind::MissingImport,
            });
        }
        
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        suggestions.truncate(3);
        
        suggestions
    }
    
    /// Get suggestions for a type mismatch
    pub fn suggest_type_conversion(&self, expected: &str, found: &str) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        
        // Common type conversion patterns
        let conversions: Vec<(&str, &str, &str)> = vec![
            ("int", "float", "use `x as float` or `float(x)`"),
            ("float", "int", "use `x as int` or `int(x)`"),
            ("string", "int", "use `int.parse(s)` or `to_int(s)`"),
            ("string", "float", "use `float.parse(s)` or `to_float(s)`"),
            ("int", "string", "use `x.to_string()` or `str(x)`"),
            ("float", "string", "use `x.to_string()` or `str(x)`"),
            ("bool", "string", "use `x.to_string()`"),
            ("&str", "string", "use `s.to_string()` or `String::from(s)`"),
            ("string", "&str", "use `&s` or `s.as_str()`"),
            ("Vec<T>", "&[T]", "use `&v` or `v.as_slice()`"),
            ("&[T]", "Vec<T>", "use `s.to_vec()` or `s.to_owned()`"),
            ("Option<T>", "T", "use `opt.unwrap()` or `opt?`"),
            ("Result<T, E>", "T", "use `res?` or `res.unwrap()`"),
        ];
        
        for (exp, fnd, hint) in conversions {
            if (expected == exp && found == fnd) || 
               (expected.contains(exp) && found.contains(fnd)) {
                suggestions.push(Suggestion {
                    message: hint.to_string(),
                    replacement: None,
                    confidence: 0.9,
                    kind: SuggestionKind::TypeConversion,
                });
            }
        }
        
        // Suggest deref if applicable
        if found.starts_with('&') && expected == &found[1..] {
            suggestions.push(Suggestion {
                message: "try dereferencing with `*`".to_string(),
                replacement: Some("*value".to_string()),
                confidence: 0.85,
                kind: SuggestionKind::Dereference,
            });
        }
        
        // Suggest reference if applicable
        if expected.starts_with('&') && &expected[1..] == found {
            suggestions.push(Suggestion {
                message: "try adding a reference with `&`".to_string(),
                replacement: Some("&value".to_string()),
                confidence: 0.85,
                kind: SuggestionKind::Reference,
            });
        }
        
        suggestions
    }
    
    /// Get suggestions for missing imports
    pub fn suggest_import(&self, name: &str) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        
        // Standard library module mappings
        let std_modules: HashMap<&str, &str> = [
            ("Vec", "std::collections"),
            ("HashMap", "std::collections"),
            ("HashSet", "std::collections"),
            ("String", "std::string"),
            ("Box", "std::box"),
            ("Rc", "std::rc"),
            ("Arc", "std::sync"),
            ("Option", "std::option"),
            ("Result", "std::result"),
            ("Duration", "std::time"),
            ("DateTime", "std::datetime"),
            ("Path", "std::path"),
            ("File", "std::fs"),
            ("Json", "std::json"),
            ("StringBuilder", "std::string"),
        ].iter().cloned().collect();
        
        if let Some(module) = std_modules.get(name) {
            suggestions.push(Suggestion {
                message: format!("`{}` is defined in `{}`", name, module),
                replacement: Some(format!("use {}::{};", module, name)),
                confidence: 0.9,
                kind: SuggestionKind::MissingImport,
            });
        }
        
        // Check all modules for the identifier
        for module in &self.modules {
            suggestions.push(Suggestion {
                message: format!("check if `{}` is available in `{}`", name, module),
                replacement: Some(format!("use {}::*;", module)),
                confidence: 0.5,
                kind: SuggestionKind::MissingImport,
            });
        }
        
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        suggestions.truncate(3);
        
        suggestions
    }
    
    /// Get suggestions for function call errors
    pub fn suggest_function_fix(&self, name: &str, args_count: usize) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        
        // Check common mistakes
        if let Some(correct) = self.common_mistakes.get(name) {
            suggestions.push(Suggestion {
                message: format!("did you mean `{}`?", correct),
                replacement: Some(correct.clone()),
                confidence: 0.95,
                kind: SuggestionKind::Typo,
            });
        }
        
        // Check function signature
        if let Some(sig) = self.functions.get(name) {
            let expected_count = sig.params.len();
            
            if args_count != expected_count {
                let param_names: Vec<&str> = sig.params.iter()
                    .map(|(n, _)| n.as_str())
                    .collect();
                
                suggestions.push(Suggestion {
                    message: format!(
                        "`{}` expects {} argument(s): {}",
                        name,
                        expected_count,
                        param_names.join(", ")
                    ),
                    replacement: None,
                    confidence: 0.9,
                    kind: SuggestionKind::ArityMismatch,
                });
            }
        }
        
        // Find similar function names
        for func_name in self.functions.keys() {
            let dist = levenshtein_distance(name, func_name);
            if dist > 0 && dist <= self.max_distance {
                if let Some(sig) = self.functions.get(func_name) {
                    let param_count = sig.params.len();
                    if param_count == args_count {
                        suggestions.push(Suggestion {
                            message: format!(
                                "did you mean `{}`? (takes {} argument(s))",
                                func_name, param_count
                            ),
                            replacement: Some(func_name.clone()),
                            confidence: 1.0 - (dist as f64 / (name.len().max(func_name.len()) as f64)),
                            kind: SuggestionKind::SimilarFunction,
                        });
                    }
                }
            }
        }
        
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        suggestions.truncate(3);
        
        suggestions
    }
    
    /// Get suggestions for pattern matching errors
    pub fn suggest_pattern(&self, pattern: &str, ty: &str) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        
        // Suggest common patterns
        if ty == "Option" || ty == "Option<T>" {
            suggestions.push(Suggestion {
                message: "for Option, use `Some(x)` or `None`".to_string(),
                replacement: None,
                confidence: 0.9,
                kind: SuggestionKind::PatternSuggestion,
            });
        }
        
        if ty == "Result" || ty == "Result<T, E>" {
            suggestions.push(Suggestion {
                message: "for Result, use `Ok(x)` or `Err(e)`".to_string(),
                replacement: None,
                confidence: 0.9,
                kind: SuggestionKind::PatternSuggestion,
            });
        }
        
        if ty == "bool" {
            suggestions.push(Suggestion {
                message: "for bool, use `true` or `false`".to_string(),
                replacement: None,
                confidence: 0.9,
                kind: SuggestionKind::PatternSuggestion,
            });
        }
        
        suggestions
    }
}

impl Default for SuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// A single suggestion for fixing an error
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub message: String,
    pub replacement: Option<String>,
    pub confidence: f64,
    pub kind: SuggestionKind,
}

impl fmt::Display for Suggestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(repl) = &self.replacement {
            write!(f, " (try: `{}`)", repl)?;
        }
        Ok(())
    }
}

/// The kind of suggestion being offered
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionKind {
    /// A typo correction
    Typo,
    /// Case sensitivity issue
    CaseMismatch,
    /// Similar identifier found
    SimilarName,
    /// Similar type found
    SimilarType,
    /// Similar function found
    SimilarFunction,
    /// Type conversion suggestion
    TypeConversion,
    /// Type convention suggestion
    TypeConvention,
    /// Dereference suggestion
    Dereference,
    /// Reference suggestion
    Reference,
    /// Missing import
    MissingImport,
    /// Wrong number of arguments
    ArityMismatch,
    /// Pattern matching suggestion
    PatternSuggestion,
    /// Syntax correction
    SyntaxCorrection,
    /// General help
    GeneralHelp,
}

/// Compute Levenshtein distance between two strings
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }
    
    let mut current = vec![0; b_len + 1];
    
    for j in 0..=b_len {
        current[j] = j;
    }
    
    for i in 1..=a_len {
        let mut previous = current[0];
        current[0] = i;
        
        for j in 1..=b_len {
            let old = current[j];
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            
            current[j] = (previous + cost)
                .min(current[j] + 1)
                .min(current[j - 1] + 1);
            
            previous = old;
        }
    }
    
    current[b_len]
}

/// Enhanced error message with formatted code snippets
pub struct ErrorFormatter {
    source_lines: Vec<String>,
    max_context_lines: usize,
}

impl ErrorFormatter {
    pub fn new(source: &str) -> Self {
        ErrorFormatter {
            source_lines: source.lines().map(|s| s.to_string()).collect(),
            max_context_lines: 3,
        }
    }
    
    /// Format an error with source context
    pub fn format_error(
        &self,
        line: usize,
        column: usize,
        end_line: Option<usize>,
        end_column: Option<usize>,
        message: &str,
        label: Option<&str>,
    ) -> String {
        let mut output = String::new();
        
        // Calculate the range to show
        let start = line.saturating_sub(self.max_context_lines).max(1);
        let end = end_line.unwrap_or(line).min(self.source_lines.len());
        
        // Calculate line number width
        let line_num_width = end.to_string().len();
        
        // Show each line
        for l in start..=end {
            let line_str = self.source_lines.get(l - 1)
                .map(|s| s.as_str())
                .unwrap_or("");
            
            // Line number
            output.push_str(&format!(" {:>width$} | ", l, width = line_num_width));
            output.push_str(line_str);
            output.push('\n');
            
            // Underline on error line
            if l == line {
                output.push_str(&format!(" {:>width$} | ", "", width = line_num_width));
                
                let col = column.saturating_sub(1);
                for _ in 0..col {
                    output.push(' ');
                }
                
                // Draw underline
                let err_len = if let (Some(el), Some(ec)) = (end_line, end_column) {
                    if el == line {
                        ec.saturating_sub(column).max(1)
                    } else {
                        line_str.len().saturating_sub(col).max(1)
                    }
                } else {
                    1
                };
                
                for _ in 0..err_len {
                    output.push('^');
                }
                
                if let Some(lbl) = label {
                    output.push_str(&format!(" {}", lbl));
                }
                
                output.push('\n');
            }
        }
        
        output
    }
}

/// Error codes and their descriptions
pub const ERROR_CODES: &[(u32, &str, &str)] = &[
    (1, "E0001", "Undefined variable or identifier"),
    (2, "E0002", "Type mismatch"),
    (3, "E0003", "Undefined function"),
    (4, "E0004", "Invalid syntax"),
    (5, "E0005", "Borrow checker error"),
    (6, "E0006", "Use of moved value"),
    (7, "E0007", "Lifetime error"),
    (8, "E0008", "Duplicate definition"),
    (9, "E0009", "Invalid character"),
    (10, "E0010", "Unterminated string"),
    (11, "E0011", "Invalid escape sequence"),
    (12, "E0012", "Invalid number literal"),
    (13, "E0013", "Unexpected token"),
    (14, "E0014", "Expected token"),
    (15, "E0015", "Invalid type"),
    (16, "E0016", "Missing import"),
    (17, "E0017", "Wrong number of arguments"),
    (18, "E0018", "Pattern not exhaustive"),
    (19, "E0019", "Unreachable pattern"),
    (20, "E0020", "Cyclic dependency"),
    (21, "E0021", "Recursion limit exceeded"),
    (22, "E0022", "Type parameter error"),
    (23, "E0023", "Trait bound not satisfied"),
    (24, "E0024", "Ambiguous associated type"),
    (25, "E0025", "Overflow during compilation"),
];

/// Get error code description
pub fn get_error_description(code: u32) -> Option<&'static str> {
    ERROR_CODES.iter()
        .find(|(c, _, _)| *c == code)
        .map(|(_, _, desc)| *desc)
}

/// Get error code name
pub fn get_error_code_name(code: u32) -> String {
    format!("E{:04}", code)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_suggestion_engine_identifier() {
        let mut engine = SuggestionEngine::new();
        engine.add_identifier("hello");
        engine.add_identifier("world");
        
        let suggestions = engine.suggest_identifier("helo");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].message.contains("hello"));
    }
    
    #[test]
    fn test_suggestion_engine_typo() {
        let engine = SuggestionEngine::new();
        
        let suggestions = engine.suggest_identifier("prnt");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].replacement == Some("print".to_string()));
    }
    
    #[test]
    fn test_type_conversion_suggestions() {
        let engine = SuggestionEngine::new();
        
        let suggestions = engine.suggest_type_conversion("float", "int");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].message.contains("as float"));
    }
    
    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("hello", "world"), 4);
    }
    
    #[test]
    fn test_error_formatter() {
        let source = "line 1\nline 2\nline 3\nline 4\nline 5";
        let formatter = ErrorFormatter::new(source);
        
        let output = formatter.format_error(3, 3, None, None, "test error", Some("here"));
        assert!(output.contains("line 3"));
        assert!(output.contains("^"));
    }
    
    #[test]
    fn test_error_codes() {
        assert_eq!(get_error_description(1), Some("Undefined variable or identifier"));
        assert_eq!(get_error_description(999), None);
        assert_eq!(get_error_code_name(1), "E0001");
    }
}
