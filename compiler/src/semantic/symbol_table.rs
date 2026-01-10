//! Symbol table for semantic analysis

use std::collections::HashMap;
use crate::parser::{Type, GenericParam};
use crate::span::Span;

/// Symbol kind
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    Variable,
    Function,
    Struct,
    Enum,
    Trait,
    TypeAlias,
    Module,
    GenericParam,
    EnumVariant,
}

/// Symbol information
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub ty: Option<Type>,
    pub span: Span,
    pub scope_id: ScopeId,
    pub is_mutable: bool,
    pub generic_params: Vec<GenericParam>,
}

impl Symbol {
    pub fn new(name: String, kind: SymbolKind, span: Span, scope_id: ScopeId) -> Self {
        Symbol {
            name,
            kind,
            ty: None,
            span,
            scope_id,
            is_mutable: false,
            generic_params: Vec::new(),
        }
    }
    
    pub fn with_type(mut self, ty: Type) -> Self {
        self.ty = Some(ty);
        self
    }
    
    pub fn with_generics(mut self, generics: Vec<GenericParam>) -> Self {
        self.generic_params = generics;
        self
    }
    
    pub fn mutable(mut self) -> Self {
        self.is_mutable = true;
        self
    }
}

/// Scope identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(usize);

impl ScopeId {
    pub fn new(id: usize) -> Self {
        ScopeId(id)
    }
    
    pub fn root() -> Self {
        ScopeId(0)
    }
}

/// Scope information
#[derive(Debug, Clone)]
pub struct Scope {
    pub id: ScopeId,
    pub parent: Option<ScopeId>,
    pub symbols: HashMap<String, Symbol>,
    pub children: Vec<ScopeId>,
}

impl Scope {
    pub fn new(id: ScopeId, parent: Option<ScopeId>) -> Self {
        Scope {
            id,
            parent,
            symbols: HashMap::new(),
            children: Vec::new(),
        }
    }
    
    pub fn insert(&mut self, symbol: Symbol) -> Option<Symbol> {
        self.symbols.insert(symbol.name.clone(), symbol)
    }
    
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }
    
    pub fn contains(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }
}

/// Symbol table managing all scopes
#[derive(Debug, Clone)]
pub struct SymbolTable {
    scopes: Vec<Scope>,
    current_scope: ScopeId,
    next_scope_id: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        let root_scope = Scope::new(ScopeId::root(), None);
        SymbolTable {
            scopes: vec![root_scope],
            current_scope: ScopeId::root(),
            next_scope_id: 1,
        }
    }
    
    /// Enter a new scope
    pub fn enter_scope(&mut self) -> ScopeId {
        let new_scope_id = ScopeId::new(self.next_scope_id);
        self.next_scope_id += 1;
        
        let new_scope = Scope::new(new_scope_id, Some(self.current_scope));
        self.scopes.push(new_scope);
        
        // Add to parent's children
        if let Some(parent) = self.get_scope_mut(self.current_scope) {
            parent.children.push(new_scope_id);
        }
        
        self.current_scope = new_scope_id;
        new_scope_id
    }
    
    /// Exit current scope
    pub fn exit_scope(&mut self) {
        if let Some(parent) = self.get_scope(self.current_scope).and_then(|s| s.parent) {
            self.current_scope = parent;
        }
    }
    
    /// Get current scope ID
    pub fn current_scope_id(&self) -> ScopeId {
        self.current_scope
    }
    
    /// Insert symbol in current scope
    pub fn insert(&mut self, symbol: Symbol) -> Result<(), String> {
        let scope = self.get_scope_mut(self.current_scope)
            .ok_or_else(|| "Invalid scope".to_string())?;
        
        if scope.contains(&symbol.name) {
            return Err(format!("Symbol '{}' already defined in this scope", symbol.name));
        }
        
        scope.insert(symbol);
        Ok(())
    }
    
    /// Lookup symbol in current scope and parent scopes
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        let mut current = Some(self.current_scope);
        
        while let Some(scope_id) = current {
            if let Some(scope) = self.get_scope(scope_id) {
                if let Some(symbol) = scope.lookup(name) {
                    return Some(symbol);
                }
                current = scope.parent;
            } else {
                break;
            }
        }
        
        None
    }
    
    /// Lookup symbol only in current scope
    pub fn lookup_current(&self, name: &str) -> Option<&Symbol> {
        self.get_scope(self.current_scope)
            .and_then(|scope| scope.lookup(name))
    }
    
    /// Get scope by ID
    pub fn get_scope(&self, id: ScopeId) -> Option<&Scope> {
        self.scopes.iter().find(|s| s.id == id)
    }
    
    /// Get mutable scope by ID
    fn get_scope_mut(&mut self, id: ScopeId) -> Option<&mut Scope> {
        self.scopes.iter_mut().find(|s| s.id == id)
    }
    
    /// Get all symbols in current scope
    pub fn current_scope_symbols(&self) -> Vec<&Symbol> {
        self.get_scope(self.current_scope)
            .map(|scope| scope.symbols.values().collect())
            .unwrap_or_default()
    }
    
    /// Get all visible symbols (current + parent scopes)
    pub fn visible_symbols(&self) -> Vec<&Symbol> {
        let mut symbols = Vec::new();
        let mut current = Some(self.current_scope);
        
        while let Some(scope_id) = current {
            if let Some(scope) = self.get_scope(scope_id) {
                symbols.extend(scope.symbols.values());
                current = scope.parent;
            } else {
                break;
            }
        }
        
        symbols
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Position;

    fn test_span() -> Span {
        Span::new(Position::new(0, 0, 0), Position::new(0, 0, 0))
    }

    #[test]
    fn test_symbol_table_creation() {
        let table = SymbolTable::new();
        assert_eq!(table.current_scope_id(), ScopeId::root());
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut table = SymbolTable::new();
        let symbol = Symbol::new(
            "x".to_string(),
            SymbolKind::Variable,
            test_span(),
            table.current_scope_id(),
        );
        
        table.insert(symbol).unwrap();
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_none());
    }

    #[test]
    fn test_duplicate_symbol() {
        let mut table = SymbolTable::new();
        let symbol1 = Symbol::new(
            "x".to_string(),
            SymbolKind::Variable,
            test_span(),
            table.current_scope_id(),
        );
        let symbol2 = Symbol::new(
            "x".to_string(),
            SymbolKind::Variable,
            test_span(),
            table.current_scope_id(),
        );
        
        table.insert(symbol1).unwrap();
        assert!(table.insert(symbol2).is_err());
    }

    #[test]
    fn test_scope_hierarchy() {
        let mut table = SymbolTable::new();
        
        // Insert in root scope
        let symbol1 = Symbol::new(
            "x".to_string(),
            SymbolKind::Variable,
            test_span(),
            table.current_scope_id(),
        );
        table.insert(symbol1).unwrap();
        
        // Enter new scope
        table.enter_scope();
        
        // Insert in nested scope
        let symbol2 = Symbol::new(
            "y".to_string(),
            SymbolKind::Variable,
            test_span(),
            table.current_scope_id(),
        );
        table.insert(symbol2).unwrap();
        
        // Should see both x and y
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_some());
        
        // Exit scope
        table.exit_scope();
        
        // Should only see x
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_none());
    }

    #[test]
    fn test_shadowing() {
        let mut table = SymbolTable::new();
        
        // Insert x in root scope
        let symbol1 = Symbol::new(
            "x".to_string(),
            SymbolKind::Variable,
            test_span(),
            table.current_scope_id(),
        );
        table.insert(symbol1).unwrap();
        
        // Enter new scope and shadow x
        table.enter_scope();
        let symbol2 = Symbol::new(
            "x".to_string(),
            SymbolKind::Variable,
            test_span(),
            table.current_scope_id(),
        );
        table.insert(symbol2).unwrap();
        
        // Should find the shadowing x
        let found = table.lookup("x").unwrap();
        assert_eq!(found.scope_id, table.current_scope_id());
    }
}
