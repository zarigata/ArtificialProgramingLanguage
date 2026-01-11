// VeZ Language Server Protocol (LSP)
// Provides IDE integration for VeZ

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// LSP types
#[derive(Debug, Clone)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

#[derive(Debug, Clone)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

#[derive(Debug, Clone)]
pub struct Location {
    pub uri: String,
    pub range: Range,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: DiagnosticSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Copy)]
pub enum DiagnosticSeverity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

#[derive(Debug, Clone)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionItemKind,
    pub detail: Option<String>,
    pub documentation: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum CompletionItemKind {
    Function = 3,
    Variable = 6,
    Struct = 7,
    Enum = 13,
    Keyword = 14,
    Module = 9,
}

// Language server
pub struct LanguageServer {
    documents: Arc<Mutex<HashMap<String, Document>>>,
    symbols: Arc<Mutex<SymbolTable>>,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub uri: String,
    pub content: String,
    pub version: i32,
}

#[derive(Debug)]
pub struct SymbolTable {
    symbols: HashMap<String, Symbol>,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub location: Location,
    pub ty: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum SymbolKind {
    Function,
    Variable,
    Struct,
    Enum,
    Trait,
    Module,
}

impl LanguageServer {
    pub fn new() -> Self {
        LanguageServer {
            documents: Arc::new(Mutex::new(HashMap::new())),
            symbols: Arc::new(Mutex::new(SymbolTable::new())),
        }
    }
    
    // Document lifecycle
    pub fn did_open(&self, uri: String, content: String) {
        let doc = Document {
            uri: uri.clone(),
            content: content.clone(),
            version: 0,
        };
        
        self.documents.lock().unwrap().insert(uri.clone(), doc);
        self.analyze_document(&uri, &content);
    }
    
    pub fn did_change(&self, uri: String, content: String, version: i32) {
        if let Some(doc) = self.documents.lock().unwrap().get_mut(&uri) {
            doc.content = content.clone();
            doc.version = version;
        }
        
        self.analyze_document(&uri, &content);
    }
    
    pub fn did_close(&self, uri: String) {
        self.documents.lock().unwrap().remove(&uri);
    }
    
    // Analysis
    fn analyze_document(&self, uri: &str, content: &str) {
        // Parse document and extract symbols
        let symbols = self.extract_symbols(content);
        
        // Update symbol table
        let mut table = self.symbols.lock().unwrap();
        for symbol in symbols {
            table.insert(symbol.name.clone(), symbol);
        }
        
        // Run diagnostics
        let diagnostics = self.check_document(content);
        self.publish_diagnostics(uri, diagnostics);
    }
    
    fn extract_symbols(&self, content: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        
        // Simple pattern matching for symbols
        for (line_num, line) in content.lines().enumerate() {
            if line.starts_with("fn ") {
                if let Some(name) = self.extract_function_name(line) {
                    symbols.push(Symbol {
                        name,
                        kind: SymbolKind::Function,
                        location: Location {
                            uri: String::new(),
                            range: Range {
                                start: Position { line: line_num as u32, character: 0 },
                                end: Position { line: line_num as u32, character: line.len() as u32 },
                            },
                        },
                        ty: None,
                    });
                }
            } else if line.starts_with("struct ") {
                if let Some(name) = self.extract_struct_name(line) {
                    symbols.push(Symbol {
                        name,
                        kind: SymbolKind::Struct,
                        location: Location {
                            uri: String::new(),
                            range: Range {
                                start: Position { line: line_num as u32, character: 0 },
                                end: Position { line: line_num as u32, character: line.len() as u32 },
                            },
                        },
                        ty: None,
                    });
                }
            }
        }
        
        symbols
    }
    
    fn extract_function_name(&self, line: &str) -> Option<String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "fn" {
            let name = parts[1].trim_end_matches('(');
            Some(name.to_string())
        } else {
            None
        }
    }
    
    fn extract_struct_name(&self, line: &str) -> Option<String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "struct" {
            Some(parts[1].trim_end_matches('{').to_string())
        } else {
            None
        }
    }
    
    fn check_document(&self, content: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        
        // Simple checks
        for (line_num, line) in content.lines().enumerate() {
            // Check for common issues
            if line.contains("TODO") {
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position { line: line_num as u32, character: 0 },
                        end: Position { line: line_num as u32, character: line.len() as u32 },
                    },
                    severity: DiagnosticSeverity::Information,
                    message: "TODO comment".to_string(),
                });
            }
            
            if line.contains("unwrap()") {
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position { line: line_num as u32, character: 0 },
                        end: Position { line: line_num as u32, character: line.len() as u32 },
                    },
                    severity: DiagnosticSeverity::Warning,
                    message: "Consider using proper error handling instead of unwrap()".to_string(),
                });
            }
        }
        
        diagnostics
    }
    
    fn publish_diagnostics(&self, uri: &str, diagnostics: Vec<Diagnostic>) {
        // In production, send to client via JSON-RPC
        println!("Diagnostics for {}: {} issues", uri, diagnostics.len());
    }
    
    // Code completion
    pub fn completion(&self, uri: &str, position: Position) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        
        // Add keywords
        for keyword in &["fn", "let", "mut", "if", "else", "match", "loop", "while", "for", "return", "struct", "enum", "trait", "impl", "use", "pub"] {
            items.push(CompletionItem {
                label: keyword.to_string(),
                kind: CompletionItemKind::Keyword,
                detail: None,
                documentation: None,
            });
        }
        
        // Add symbols from symbol table
        let table = self.symbols.lock().unwrap();
        for symbol in table.symbols.values() {
            items.push(CompletionItem {
                label: symbol.name.clone(),
                kind: match symbol.kind {
                    SymbolKind::Function => CompletionItemKind::Function,
                    SymbolKind::Variable => CompletionItemKind::Variable,
                    SymbolKind::Struct => CompletionItemKind::Struct,
                    SymbolKind::Enum => CompletionItemKind::Enum,
                    _ => CompletionItemKind::Variable,
                },
                detail: symbol.ty.clone(),
                documentation: None,
            });
        }
        
        items
    }
    
    // Go to definition
    pub fn goto_definition(&self, uri: &str, position: Position) -> Option<Location> {
        // Find symbol at position
        let doc = self.documents.lock().unwrap();
        let content = doc.get(uri)?;
        
        let word = self.word_at_position(&content.content, position)?;
        
        // Look up in symbol table
        let table = self.symbols.lock().unwrap();
        let symbol = table.get(&word)?;
        
        Some(symbol.location.clone())
    }
    
    // Hover information
    pub fn hover(&self, uri: &str, position: Position) -> Option<String> {
        let doc = self.documents.lock().unwrap();
        let content = doc.get(uri)?;
        
        let word = self.word_at_position(&content.content, position)?;
        
        let table = self.symbols.lock().unwrap();
        let symbol = table.get(&word)?;
        
        Some(format!("{:?}: {}", symbol.kind, symbol.name))
    }
    
    // Rename symbol
    pub fn rename(&self, uri: &str, position: Position, new_name: String) -> Vec<Location> {
        let mut locations = Vec::new();
        
        // Find all references to symbol
        let doc = self.documents.lock().unwrap();
        if let Some(content) = doc.get(uri) {
            let word = self.word_at_position(&content.content, position);
            if let Some(word) = word {
                // Find all occurrences
                for (line_num, line) in content.content.lines().enumerate() {
                    if line.contains(&word) {
                        locations.push(Location {
                            uri: uri.to_string(),
                            range: Range {
                                start: Position { line: line_num as u32, character: 0 },
                                end: Position { line: line_num as u32, character: line.len() as u32 },
                            },
                        });
                    }
                }
            }
        }
        
        locations
    }
    
    fn word_at_position(&self, content: &str, position: Position) -> Option<String> {
        let lines: Vec<&str> = content.lines().collect();
        if position.line as usize >= lines.len() {
            return None;
        }
        
        let line = lines[position.line as usize];
        let chars: Vec<char> = line.chars().collect();
        
        if position.character as usize >= chars.len() {
            return None;
        }
        
        // Find word boundaries
        let mut start = position.character as usize;
        let mut end = position.character as usize;
        
        while start > 0 && chars[start - 1].is_alphanumeric() {
            start -= 1;
        }
        
        while end < chars.len() && chars[end].is_alphanumeric() {
            end += 1;
        }
        
        Some(chars[start..end].iter().collect())
    }
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            symbols: HashMap::new(),
        }
    }
    
    pub fn insert(&mut self, name: String, symbol: Symbol) {
        self.symbols.insert(name, symbol);
    }
    
    pub fn get(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }
}

fn main() {
    println!("VeZ Language Server starting...");
    
    let server = LanguageServer::new();
    
    // In production, this would handle JSON-RPC messages
    println!("Language server ready");
}
