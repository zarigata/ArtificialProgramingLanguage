//! Main borrow checker implementation

use crate::parser::*;
use crate::error::{Error, ErrorKind, Result};
use crate::semantic::SymbolTable;
use super::lifetime::{LifetimeEnv, LifetimeId};
use super::ownership::{OwnershipTracker, MoveChecker};
use std::collections::HashMap;

/// Borrow checker state
pub struct BorrowChecker {
    symbol_table: SymbolTable,
    lifetime_env: LifetimeEnv,
    ownership_tracker: OwnershipTracker,
    move_checker: MoveChecker,
    /// Map expressions to their inferred lifetimes
    expr_lifetimes: HashMap<usize, LifetimeId>,
    errors: Vec<Error>,
}

impl BorrowChecker {
    pub fn new(symbol_table: SymbolTable) -> Self {
        BorrowChecker {
            symbol_table,
            lifetime_env: LifetimeEnv::new(),
            ownership_tracker: OwnershipTracker::new(),
            move_checker: MoveChecker::new(),
            expr_lifetimes: HashMap::new(),
            errors: Vec::new(),
        }
    }
    
    /// Check a program for borrow errors
    pub fn check_program(&mut self, program: &Program) -> Result<()> {
        for item in &program.items {
            self.check_item(item)?;
        }
        
        // Solve lifetime constraints
        self.lifetime_env.solve_constraints()?;
        
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.remove(0))
        }
    }
    
    fn check_item(&mut self, item: &Item) -> Result<()> {
        match item {
            Item::Function(func) => self.check_function(func),
            Item::Impl(impl_block) => {
                for item in &impl_block.items {
                    if let ImplItem::Function(func) = item {
                        self.check_function(func)?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    
    fn check_function(&mut self, func: &Function) -> Result<()> {
        // Create fresh ownership tracker for function scope
        let mut func_tracker = OwnershipTracker::new();
        
        // Register parameters as owned
        for param in &func.params {
            func_tracker.register(param.name.clone());
        }
        
        // Save current tracker and use function tracker
        let saved_tracker = std::mem::replace(&mut self.ownership_tracker, func_tracker);
        
        // Check function body
        for stmt in &func.body {
            self.check_stmt(stmt)?;
        }
        
        // Restore tracker
        self.ownership_tracker = saved_tracker;
        
        Ok(())
    }
    
    fn check_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let(name, _ty, init) => {
                if let Some(expr) = init {
                    self.check_expr(expr)?;
                }
                self.ownership_tracker.register(name.clone());
                Ok(())
            }
            Stmt::Expr(expr) => {
                self.check_expr(expr)?;
                Ok(())
            }
            Stmt::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    self.check_expr(expr)?;
                }
                Ok(())
            }
        }
    }
    
    fn check_expr(&mut self, expr: &Expr) -> Result<()> {
        match expr {
            Expr::Literal(_) => Ok(()),
            
            Expr::Ident(name) => {
                // Check if variable is available (not moved)
                if !self.ownership_tracker.is_available(name) {
                    return Err(Error::new(
                        ErrorKind::BorrowError,
                        format!("Use of moved value '{}'", name)
                    ));
                }
                
                // For non-Copy types, mark as moved
                // (simplified - would need type information)
                Ok(())
            }
            
            Expr::Binary(left, _op, right) => {
                self.check_expr(left)?;
                self.check_expr(right)?;
                Ok(())
            }
            
            Expr::Unary(op, operand) => {
                match op {
                    UnOp::Ref => {
                        // Taking a reference - mark as borrowed
                        if let Expr::Ident(name) = operand.as_ref() {
                            self.ownership_tracker.mark_borrowed_shared(name)?;
                        }
                        self.check_expr(operand)?;
                    }
                    UnOp::Deref => {
                        // Dereferencing - check the reference
                        self.check_expr(operand)?;
                    }
                    _ => {
                        self.check_expr(operand)?;
                    }
                }
                Ok(())
            }
            
            Expr::Call(func, args) => {
                self.check_expr(func)?;
                for arg in args {
                    self.check_expr(arg)?;
                    
                    // If argument is a variable, it might be moved
                    if let Expr::Ident(name) = arg {
                        // Would need type info to determine if this is a move
                        // For now, assume non-Copy types move
                        let _ = name; // Placeholder
                    }
                }
                Ok(())
            }
            
            Expr::MethodCall(obj, _method, args) => {
                self.check_expr(obj)?;
                for arg in args {
                    self.check_expr(arg)?;
                }
                Ok(())
            }
            
            Expr::Field(obj, _field) => {
                self.check_expr(obj)?;
                Ok(())
            }
            
            Expr::Index(arr, idx) => {
                self.check_expr(arr)?;
                self.check_expr(idx)?;
                Ok(())
            }
            
            Expr::Block(stmts) => {
                for stmt in stmts {
                    self.check_stmt(stmt)?;
                }
                Ok(())
            }
            
            Expr::If(cond, then_branch, else_branch) => {
                self.check_expr(cond)?;
                self.check_expr(then_branch)?;
                if let Some(else_expr) = else_branch {
                    self.check_expr(else_expr)?;
                }
                Ok(())
            }
            
            Expr::Match(scrutinee, arms) => {
                self.check_expr(scrutinee)?;
                
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard)?;
                    }
                    self.check_expr(&arm.body)?;
                }
                Ok(())
            }
            
            Expr::Loop(body) => {
                self.check_expr(body)?;
                Ok(())
            }
            
            Expr::While(cond, body) => {
                self.check_expr(cond)?;
                self.check_expr(body)?;
                Ok(())
            }
            
            Expr::For(_var, iter, body) => {
                self.check_expr(iter)?;
                self.check_expr(body)?;
                Ok(())
            }
            
            Expr::Break(value) => {
                if let Some(expr) = value {
                    self.check_expr(expr)?;
                }
                Ok(())
            }
            
            Expr::Continue => Ok(()),
            
            Expr::Return(value) => {
                if let Some(expr) = value {
                    self.check_expr(expr)?;
                }
                Ok(())
            }
            
            Expr::Array(elements) => {
                for elem in elements {
                    self.check_expr(elem)?;
                }
                Ok(())
            }
            
            Expr::Tuple(elements) => {
                for elem in elements {
                    self.check_expr(elem)?;
                }
                Ok(())
            }
            
            Expr::StructLiteral(_name, fields) => {
                for (_field_name, expr) in fields {
                    self.check_expr(expr)?;
                }
                Ok(())
            }
        }
    }
    
    /// Get all errors encountered
    pub fn errors(&self) -> &[Error] {
        &self.errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::semantic::Resolver;

    fn borrow_check_source(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let resolver = Resolver::new();
        let symbol_table = resolver.resolve(&program)?;
        
        let mut checker = BorrowChecker::new(symbol_table);
        checker.check_program(&program)
    }

    #[test]
    fn test_simple_function() {
        let source = "fn main() { let x = 42; }";
        assert!(borrow_check_source(source).is_ok());
    }

    #[test]
    fn test_variable_use() {
        let source = r#"
            fn main() {
                let x = 42;
                let y = x;
            }
        "#;
        assert!(borrow_check_source(source).is_ok());
    }

    #[test]
    fn test_reference() {
        let source = r#"
            fn main() {
                let x = 42;
                let y = &x;
            }
        "#;
        assert!(borrow_check_source(source).is_ok());
    }

    #[test]
    fn test_multiple_borrows() {
        let source = r#"
            fn main() {
                let x = 42;
                let y = &x;
                let z = &x;
            }
        "#;
        assert!(borrow_check_source(source).is_ok());
    }
}
