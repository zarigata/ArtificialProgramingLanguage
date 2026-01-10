//! Symbol table and name resolution

use std::collections::HashMap;

/// Symbol identifier
pub type SymbolId = usize;

/// Scope identifier
pub type ScopeId = usize;

/// Symbol table for name resolution
pub struct SymbolTable {
    scopes: Vec<Scope>,
    current_scope: ScopeId,
}

/// A lexical scope
pub struct Scope {
    parent: Option<ScopeId>,
    symbols: HashMap<String, SymbolId>,
}

/// A symbol in the symbol table
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Variable,
    Function,
    Type,
    Trait,
    Module,
}

impl SymbolTable {
    pub fn new() -> Self {
        let root_scope = Scope {
            parent: None,
            symbols: HashMap::new(),
        };
        
        SymbolTable {
            scopes: vec![root_scope],
            current_scope: 0,
        }
    }
    
    pub fn enter_scope(&mut self) -> ScopeId {
        let new_scope = Scope {
            parent: Some(self.current_scope),
            symbols: HashMap::new(),
        };
        
        let scope_id = self.scopes.len();
        self.scopes.push(new_scope);
        self.current_scope = scope_id;
        scope_id
    }
    
    pub fn exit_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope].parent {
            self.current_scope = parent;
        }
    }
    
    pub fn insert(&mut self, name: String, kind: SymbolKind) -> SymbolId {
        let symbol_id = self.scopes.len(); // Simple ID generation
        self.scopes[self.current_scope].symbols.insert(name, symbol_id);
        symbol_id
    }
    
    pub fn lookup(&self, name: &str) -> Option<SymbolId> {
        let mut current = Some(self.current_scope);
        
        while let Some(scope_id) = current {
            if let Some(&symbol_id) = self.scopes[scope_id].symbols.get(name) {
                return Some(symbol_id);
            }
            current = self.scopes[scope_id].parent;
        }
        
        None
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
