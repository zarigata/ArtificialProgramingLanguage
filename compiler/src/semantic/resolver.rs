//! Name resolution - builds symbol table from AST

use crate::parser::*;
use crate::error::{Error, ErrorKind, Result};
use crate::span::Span;
use super::symbol_table::{SymbolTable, Symbol, SymbolKind};

/// Name resolver - walks AST and builds symbol table
pub struct Resolver {
    symbol_table: SymbolTable,
    errors: Vec<Error>,
}

impl Resolver {
    pub fn new() -> Self {
        Resolver {
            symbol_table: SymbolTable::new(),
            errors: Vec::new(),
        }
    }
    
    /// Resolve a program and return the symbol table
    pub fn resolve(mut self, program: &Program) -> Result<SymbolTable> {
        self.visit_program(program);
        
        if self.errors.is_empty() {
            Ok(self.symbol_table)
        } else {
            Err(self.errors.into_iter().next().unwrap())
        }
    }
    
    /// Get all errors encountered
    pub fn errors(&self) -> &[Error] {
        &self.errors
    }
    
    fn add_error(&mut self, error: Error) {
        self.errors.push(error);
    }
    
    fn visit_program(&mut self, program: &Program) {
        for item in &program.items {
            self.visit_item(item);
        }
    }
    
    fn visit_item(&mut self, item: &Item) {
        match item {
            Item::Function(func) => self.visit_function(func),
            Item::Struct(struct_def) => self.visit_struct(struct_def),
            Item::Enum(enum_def) => self.visit_enum(enum_def),
            Item::Trait(trait_def) => self.visit_trait(trait_def),
            Item::Impl(impl_block) => self.visit_impl(impl_block),
            Item::Use(use_path) => self.visit_use(use_path),
            Item::Mod(name, items) => self.visit_mod(name, items),
        }
    }
    
    fn visit_function(&mut self, func: &Function) {
        // Register function in current scope
        let scope_id = self.symbol_table.current_scope_id();
        let symbol = Symbol::new(
            func.name.clone(),
            SymbolKind::Function,
            Span::default(),
            scope_id,
        ).with_generics(func.generics.clone());
        
        if let Err(e) = self.symbol_table.insert(symbol) {
            self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
            return;
        }
        
        // Enter function scope
        self.symbol_table.enter_scope();
        
        // Register generic parameters
        for generic in &func.generics {
            let scope_id = self.symbol_table.current_scope_id();
            let symbol = Symbol::new(
                generic.name.clone(),
                SymbolKind::GenericParam,
                Span::default(),
                scope_id,
            );
            
            if let Err(e) = self.symbol_table.insert(symbol) {
                self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
            }
        }
        
        // Register parameters
        for param in &func.params {
            let scope_id = self.symbol_table.current_scope_id();
            let symbol = Symbol::new(
                param.name.clone(),
                SymbolKind::Variable,
                Span::default(),
                scope_id,
            ).with_type(param.ty.clone());
            
            if let Err(e) = self.symbol_table.insert(symbol) {
                self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
            }
        }
        
        // Visit function body
        for stmt in &func.body {
            self.visit_stmt(stmt);
        }
        
        // Exit function scope
        self.symbol_table.exit_scope();
    }
    
    fn visit_struct(&mut self, struct_def: &Struct) {
        let scope_id = self.symbol_table.current_scope_id();
        let symbol = Symbol::new(
            struct_def.name.clone(),
            SymbolKind::Struct,
            Span::default(),
            scope_id,
        ).with_generics(struct_def.generics.clone());
        
        if let Err(e) = self.symbol_table.insert(symbol) {
            self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
        }
    }
    
    fn visit_enum(&mut self, enum_def: &Enum) {
        let scope_id = self.symbol_table.current_scope_id();
        let symbol = Symbol::new(
            enum_def.name.clone(),
            SymbolKind::Enum,
            Span::default(),
            scope_id,
        ).with_generics(enum_def.generics.clone());
        
        if let Err(e) = self.symbol_table.insert(symbol) {
            self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
            return;
        }
        
        // Register enum variants
        for variant in &enum_def.variants {
            let scope_id = self.symbol_table.current_scope_id();
            let symbol = Symbol::new(
                variant.name.clone(),
                SymbolKind::EnumVariant,
                Span::default(),
                scope_id,
            );
            
            if let Err(e) = self.symbol_table.insert(symbol) {
                self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
            }
        }
    }
    
    fn visit_trait(&mut self, trait_def: &Trait) {
        let scope_id = self.symbol_table.current_scope_id();
        let symbol = Symbol::new(
            trait_def.name.clone(),
            SymbolKind::Trait,
            Span::default(),
            scope_id,
        ).with_generics(trait_def.generics.clone());
        
        if let Err(e) = self.symbol_table.insert(symbol) {
            self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
        }
    }
    
    fn visit_impl(&mut self, impl_block: &Impl) {
        // Enter impl scope
        self.symbol_table.enter_scope();
        
        // Register generic parameters
        for generic in &impl_block.generics {
            let scope_id = self.symbol_table.current_scope_id();
            let symbol = Symbol::new(
                generic.name.clone(),
                SymbolKind::GenericParam,
                Span::default(),
                scope_id,
            );
            
            if let Err(e) = self.symbol_table.insert(symbol) {
                self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
            }
        }
        
        // Visit impl items
        for item in &impl_block.items {
            match item {
                ImplItem::Function(func) => self.visit_function(func),
                ImplItem::Type(_, _) => {
                    // Type aliases handled separately
                }
            }
        }
        
        // Exit impl scope
        self.symbol_table.exit_scope();
    }
    
    fn visit_use(&mut self, _use_path: &UsePath) {
        // Use statements handled in later phase
    }
    
    fn visit_mod(&mut self, name: &str, items: &[Item]) {
        let scope_id = self.symbol_table.current_scope_id();
        let symbol = Symbol::new(
            name.to_string(),
            SymbolKind::Module,
            Span::default(),
            scope_id,
        );
        
        if let Err(e) = self.symbol_table.insert(symbol) {
            self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
            return;
        }
        
        // Enter module scope
        self.symbol_table.enter_scope();
        
        for item in items {
            self.visit_item(item);
        }
        
        // Exit module scope
        self.symbol_table.exit_scope();
    }
    
    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(name, ty, init) => {
                // Visit initializer first (if present)
                if let Some(expr) = init {
                    self.visit_expr(expr);
                }
                
                // Register variable
                let scope_id = self.symbol_table.current_scope_id();
                let mut symbol = Symbol::new(
                    name.clone(),
                    SymbolKind::Variable,
                    Span::default(),
                    scope_id,
                );
                
                if let Some(ty) = ty {
                    symbol = symbol.with_type(ty.clone());
                }
                
                if let Err(e) = self.symbol_table.insert(symbol) {
                    self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
                }
            }
            Stmt::Expr(expr) => {
                self.visit_expr(expr);
            }
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    self.visit_expr(expr);
                }
            }
        }
    }
    
    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Literal(_) => {}
            Expr::Ident(name) => {
                // Check if identifier is defined
                if self.symbol_table.lookup(name).is_none() {
                    self.add_error(Error::new(
                        ErrorKind::UndefinedSymbol,
                        format!("Undefined symbol: {}", name),
                    ));
                }
            }
            Expr::Binary(left, _, right) => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            Expr::Unary(_, expr) => {
                self.visit_expr(expr);
            }
            Expr::Call(func, args) => {
                self.visit_expr(func);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::MethodCall(obj, _, args) => {
                self.visit_expr(obj);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::Field(obj, _) => {
                self.visit_expr(obj);
            }
            Expr::Index(arr, idx) => {
                self.visit_expr(arr);
                self.visit_expr(idx);
            }
            Expr::Block(stmts) => {
                self.symbol_table.enter_scope();
                for stmt in stmts {
                    self.visit_stmt(stmt);
                }
                self.symbol_table.exit_scope();
            }
            Expr::If(cond, then_branch, else_branch) => {
                self.visit_expr(cond);
                self.visit_expr(then_branch);
                if let Some(else_expr) = else_branch {
                    self.visit_expr(else_expr);
                }
            }
            Expr::Match(scrutinee, arms) => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.symbol_table.enter_scope();
                    self.visit_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                    self.symbol_table.exit_scope();
                }
            }
            Expr::Loop(body) => {
                self.symbol_table.enter_scope();
                self.visit_expr(body);
                self.symbol_table.exit_scope();
            }
            Expr::While(cond, body) => {
                self.visit_expr(cond);
                self.symbol_table.enter_scope();
                self.visit_expr(body);
                self.symbol_table.exit_scope();
            }
            Expr::For(var, iter, body) => {
                self.visit_expr(iter);
                self.symbol_table.enter_scope();
                
                // Register loop variable
                let scope_id = self.symbol_table.current_scope_id();
                let symbol = Symbol::new(
                    var.clone(),
                    SymbolKind::Variable,
                    Span::default(),
                    scope_id,
                );
                
                if let Err(e) = self.symbol_table.insert(symbol) {
                    self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
                }
                
                self.visit_expr(body);
                self.symbol_table.exit_scope();
            }
            Expr::Break(value) => {
                if let Some(expr) = value {
                    self.visit_expr(expr);
                }
            }
            Expr::Continue => {}
            Expr::Return(value) => {
                if let Some(expr) = value {
                    self.visit_expr(expr);
                }
            }
            Expr::Array(elements) => {
                for elem in elements {
                    self.visit_expr(elem);
                }
            }
            Expr::Tuple(elements) => {
                for elem in elements {
                    self.visit_expr(elem);
                }
            }
            Expr::StructLiteral(_, fields) => {
                for (_, expr) in fields {
                    self.visit_expr(expr);
                }
            }
        }
    }
    
    fn visit_pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Wildcard => {}
            Pattern::Literal(_) => {}
            Pattern::Ident(name) => {
                // Register pattern variable
                let scope_id = self.symbol_table.current_scope_id();
                let symbol = Symbol::new(
                    name.clone(),
                    SymbolKind::Variable,
                    Span::default(),
                    scope_id,
                );
                
                if let Err(e) = self.symbol_table.insert(symbol) {
                    self.add_error(Error::new(ErrorKind::DuplicateSymbol, e));
                }
            }
            Pattern::Tuple(patterns) => {
                for pat in patterns {
                    self.visit_pattern(pat);
                }
            }
            Pattern::Struct(_, fields) => {
                for (_, pat) in fields {
                    self.visit_pattern(pat);
                }
            }
            Pattern::Enum(_, patterns) => {
                for pat in patterns {
                    self.visit_pattern(pat);
                }
            }
            Pattern::Or(patterns) => {
                for pat in patterns {
                    self.visit_pattern(pat);
                }
            }
        }
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse_and_resolve(source: &str) -> Result<SymbolTable> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let resolver = Resolver::new();
        resolver.resolve(&program)
    }

    #[test]
    fn test_simple_function() {
        let source = "fn main() { }";
        let table = parse_and_resolve(source).unwrap();
        assert!(table.lookup("main").is_some());
    }

    #[test]
    fn test_function_parameters() {
        let source = "fn add(a: i32, b: i32) -> i32 { a }";
        let table = parse_and_resolve(source).unwrap();
        assert!(table.lookup("add").is_some());
    }

    #[test]
    fn test_let_binding() {
        let source = "fn main() { let x = 42; }";
        let _table = parse_and_resolve(source).unwrap();
    }

    #[test]
    fn test_undefined_variable() {
        let source = "fn main() { x; }";
        let result = parse_and_resolve(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_struct_definition() {
        let source = "struct Point { x: f64, y: f64 }";
        let table = parse_and_resolve(source).unwrap();
        assert!(table.lookup("Point").is_some());
    }

    #[test]
    fn test_enum_definition() {
        let source = "enum Option { Some(i32), None }";
        let table = parse_and_resolve(source).unwrap();
        assert!(table.lookup("Option").is_some());
        assert!(table.lookup("Some").is_some());
        assert!(table.lookup("None").is_some());
    }

    #[test]
    fn test_generic_function() {
        let source = "fn identity<T>(x: T) -> T { x }";
        let _table = parse_and_resolve(source).unwrap();
    }

    #[test]
    fn test_nested_scopes() {
        let source = r#"
            fn main() {
                let x = 1;
                {
                    let y = 2;
                }
            }
        "#;
        let _table = parse_and_resolve(source).unwrap();
    }

    #[test]
    fn test_duplicate_function() {
        let source = r#"
            fn foo() { }
            fn foo() { }
        "#;
        let result = parse_and_resolve(source);
        assert!(result.is_err());
    }
}
