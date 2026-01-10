//! Type checking and inference

use crate::parser::*;
use crate::error::{Error, ErrorKind, Result};
use super::symbol_table::SymbolTable;
use super::type_env::{TypeEnv, InferredType, Substitution, Unifier, TypeVar};
use std::collections::HashMap;

/// Type checker with inference
pub struct TypeChecker {
    symbol_table: SymbolTable,
    type_env: TypeEnv,
    constraints: Vec<(InferredType, InferredType)>,
    expr_types: HashMap<usize, InferredType>,
    next_expr_id: usize,
}

impl TypeChecker {
    pub fn new(symbol_table: SymbolTable) -> Self {
        TypeChecker {
            symbol_table,
            type_env: TypeEnv::new(),
            constraints: Vec::new(),
            expr_types: HashMap::new(),
            next_expr_id: 0,
        }
    }
    
    /// Type check a program
    pub fn check_program(&mut self, program: &Program) -> Result<()> {
        // First pass: collect all type signatures
        for item in &program.items {
            self.collect_signatures(item)?;
        }
        
        // Second pass: check function bodies
        for item in &program.items {
            self.check_item(item)?;
        }
        
        // Solve constraints
        let subst = self.solve_constraints()?;
        
        // Apply substitution to all inferred types
        self.apply_substitution(&subst);
        
        Ok(())
    }
    
    fn collect_signatures(&mut self, item: &Item) -> Result<()> {
        match item {
            Item::Function(func) => {
                // Build function type
                let param_types: Vec<InferredType> = func.params.iter()
                    .map(|p| InferredType::from_ast(&p.ty))
                    .collect();
                
                let ret_type = func.return_type.as_ref()
                    .map(InferredType::from_ast)
                    .unwrap_or_else(|| {
                        InferredType::Concrete(Type::Named("()".to_string()))
                    });
                
                let func_type = InferredType::Function(param_types, Box::new(ret_type));
                self.type_env.bind(func.name.clone(), func_type);
            }
            Item::Struct(_) | Item::Enum(_) | Item::Trait(_) => {
                // Type constructors handled separately
            }
            Item::Impl(impl_block) => {
                for item in &impl_block.items {
                    if let ImplItem::Function(func) = item {
                        self.collect_signatures(&Item::Function(func.clone()))?;
                    }
                }
            }
            Item::Use(_) | Item::Mod(_, _) => {}
        }
        Ok(())
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
        // Enter function scope
        let mut func_env = self.type_env.child();
        
        // Bind parameters
        for param in &func.params {
            let param_type = InferredType::from_ast(&param.ty);
            func_env.bind(param.name.clone(), param_type);
        }
        
        // Save current env and switch to function env
        let saved_env = std::mem::replace(&mut self.type_env, func_env);
        
        // Check function body
        let mut body_type = InferredType::Concrete(Type::Named("()".to_string()));
        for stmt in &func.body {
            body_type = self.check_stmt(stmt)?;
        }
        
        // Check return type matches
        if let Some(ret_ty) = &func.return_type {
            let expected = InferredType::from_ast(ret_ty);
            self.add_constraint(body_type, expected);
        }
        
        // Restore environment
        self.type_env = saved_env;
        
        Ok(())
    }
    
    fn check_stmt(&mut self, stmt: &Stmt) -> Result<InferredType> {
        match stmt {
            Stmt::Let(name, ty_opt, init) => {
                let var_type = if let Some(init_expr) = init {
                    let inferred = self.infer_expr(init_expr)?;
                    
                    if let Some(declared_ty) = ty_opt {
                        let expected = InferredType::from_ast(declared_ty);
                        self.add_constraint(inferred.clone(), expected);
                    }
                    
                    inferred
                } else if let Some(declared_ty) = ty_opt {
                    InferredType::from_ast(declared_ty)
                } else {
                    return Err(Error::new(
                        ErrorKind::TypeError,
                        "Let binding must have type annotation or initializer"
                    ));
                };
                
                self.type_env.bind(name.clone(), var_type);
                Ok(InferredType::Concrete(Type::Named("()".to_string())))
            }
            Stmt::Expr(expr) => self.infer_expr(expr),
            Stmt::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    self.infer_expr(expr)
                } else {
                    Ok(InferredType::Concrete(Type::Named("()".to_string())))
                }
            }
        }
    }
    
    fn infer_expr(&mut self, expr: &Expr) -> Result<InferredType> {
        let ty = match expr {
            Expr::Literal(lit) => self.infer_literal(lit),
            
            Expr::Ident(name) => {
                self.type_env.lookup(name)
                    .cloned()
                    .ok_or_else(|| Error::new(
                        ErrorKind::UndefinedSymbol,
                        format!("Undefined variable: {}", name)
                    ))?
            }
            
            Expr::Binary(left, op, right) => {
                let left_ty = self.infer_expr(left)?;
                let right_ty = self.infer_expr(right)?;
                
                // For now, require both operands to have the same type
                self.add_constraint(left_ty.clone(), right_ty.clone());
                
                // Result type depends on operator
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        left_ty
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        InferredType::Concrete(Type::Named("bool".to_string()))
                    }
                    BinOp::And | BinOp::Or => {
                        // Require boolean operands
                        let bool_ty = InferredType::Concrete(Type::Named("bool".to_string()));
                        self.add_constraint(left_ty, bool_ty.clone());
                        self.add_constraint(right_ty, bool_ty.clone());
                        bool_ty
                    }
                }
            }
            
            Expr::Unary(op, operand) => {
                let operand_ty = self.infer_expr(operand)?;
                
                match op {
                    UnOp::Neg => operand_ty,
                    UnOp::Not => {
                        let bool_ty = InferredType::Concrete(Type::Named("bool".to_string()));
                        self.add_constraint(operand_ty, bool_ty.clone());
                        bool_ty
                    }
                    UnOp::Deref => {
                        // Dereference: &T -> T
                        let inner_var = InferredType::Var(self.type_env.fresh_var());
                        self.add_constraint(
                            operand_ty,
                            InferredType::Reference(Box::new(inner_var.clone()))
                        );
                        inner_var
                    }
                    UnOp::Ref => {
                        // Reference: T -> &T
                        InferredType::Reference(Box::new(operand_ty))
                    }
                }
            }
            
            Expr::Call(func, args) => {
                let func_ty = self.infer_expr(func)?;
                let arg_types: Result<Vec<_>> = args.iter()
                    .map(|arg| self.infer_expr(arg))
                    .collect();
                let arg_types = arg_types?;
                
                // Create fresh type variable for return type
                let ret_var = InferredType::Var(self.type_env.fresh_var());
                
                // Constrain function type
                let expected_func_ty = InferredType::Function(arg_types, Box::new(ret_var.clone()));
                self.add_constraint(func_ty, expected_func_ty);
                
                ret_var
            }
            
            Expr::MethodCall(_, _, _) => {
                // Method calls require trait resolution
                InferredType::Var(self.type_env.fresh_var())
            }
            
            Expr::Field(obj, _field) => {
                let _obj_ty = self.infer_expr(obj)?;
                // Field access requires struct type information
                InferredType::Var(self.type_env.fresh_var())
            }
            
            Expr::Index(arr, idx) => {
                let arr_ty = self.infer_expr(arr)?;
                let idx_ty = self.infer_expr(idx)?;
                
                // Index must be integer
                let int_ty = InferredType::Concrete(Type::Named("usize".to_string()));
                self.add_constraint(idx_ty, int_ty);
                
                // Array type: [T; N] -> T
                let elem_var = InferredType::Var(self.type_env.fresh_var());
                self.add_constraint(
                    arr_ty,
                    InferredType::Array(Box::new(elem_var.clone()), 0) // Size unknown
                );
                
                elem_var
            }
            
            Expr::Block(stmts) => {
                let mut last_ty = InferredType::Concrete(Type::Named("()".to_string()));
                for stmt in stmts {
                    last_ty = self.check_stmt(stmt)?;
                }
                last_ty
            }
            
            Expr::If(cond, then_branch, else_branch) => {
                let cond_ty = self.infer_expr(cond)?;
                let bool_ty = InferredType::Concrete(Type::Named("bool".to_string()));
                self.add_constraint(cond_ty, bool_ty);
                
                let then_ty = self.infer_expr(then_branch)?;
                
                if let Some(else_expr) = else_branch {
                    let else_ty = self.infer_expr(else_expr)?;
                    self.add_constraint(then_ty.clone(), else_ty);
                    then_ty
                } else {
                    InferredType::Concrete(Type::Named("()".to_string()))
                }
            }
            
            Expr::Match(scrutinee, arms) => {
                let _scrutinee_ty = self.infer_expr(scrutinee)?;
                
                if arms.is_empty() {
                    return Ok(InferredType::Concrete(Type::Named("()".to_string())));
                }
                
                // All arms must have the same type
                let first_arm_ty = self.infer_expr(&arms[0].body)?;
                
                for arm in &arms[1..] {
                    let arm_ty = self.infer_expr(&arm.body)?;
                    self.add_constraint(first_arm_ty.clone(), arm_ty);
                }
                
                first_arm_ty
            }
            
            Expr::Loop(body) => {
                self.infer_expr(body)?;
                // Loop never returns normally
                InferredType::Concrete(Type::Named("!".to_string()))
            }
            
            Expr::While(cond, body) => {
                let cond_ty = self.infer_expr(cond)?;
                let bool_ty = InferredType::Concrete(Type::Named("bool".to_string()));
                self.add_constraint(cond_ty, bool_ty);
                
                self.infer_expr(body)?;
                InferredType::Concrete(Type::Named("()".to_string()))
            }
            
            Expr::For(_var, iter, body) => {
                self.infer_expr(iter)?;
                self.infer_expr(body)?;
                InferredType::Concrete(Type::Named("()".to_string()))
            }
            
            Expr::Break(value) => {
                if let Some(expr) = value {
                    self.infer_expr(expr)?;
                }
                InferredType::Concrete(Type::Named("!".to_string()))
            }
            
            Expr::Continue => {
                InferredType::Concrete(Type::Named("!".to_string()))
            }
            
            Expr::Return(value) => {
                if let Some(expr) = value {
                    self.infer_expr(expr)?;
                }
                InferredType::Concrete(Type::Named("!".to_string()))
            }
            
            Expr::Array(elements) => {
                if elements.is_empty() {
                    let elem_var = InferredType::Var(self.type_env.fresh_var());
                    return Ok(InferredType::Array(Box::new(elem_var), 0));
                }
                
                let first_ty = self.infer_expr(&elements[0])?;
                
                for elem in &elements[1..] {
                    let elem_ty = self.infer_expr(elem)?;
                    self.add_constraint(first_ty.clone(), elem_ty);
                }
                
                InferredType::Array(Box::new(first_ty), elements.len())
            }
            
            Expr::Tuple(elements) => {
                let types: Result<Vec<_>> = elements.iter()
                    .map(|e| self.infer_expr(e))
                    .collect();
                InferredType::Tuple(types?)
            }
            
            Expr::StructLiteral(name, _fields) => {
                // Struct literal requires struct definition lookup
                InferredType::Concrete(Type::Named(name.clone()))
            }
        };
        
        Ok(ty)
    }
    
    fn infer_literal(&self, lit: &Literal) -> InferredType {
        match lit {
            Literal::Int(_) => InferredType::Concrete(Type::Named("i32".to_string())),
            Literal::Float(_) => InferredType::Concrete(Type::Named("f64".to_string())),
            Literal::String(_) => InferredType::Concrete(Type::Named("String".to_string())),
            Literal::Char(_) => InferredType::Concrete(Type::Named("char".to_string())),
            Literal::Bool(_) => InferredType::Concrete(Type::Named("bool".to_string())),
        }
    }
    
    fn add_constraint(&mut self, t1: InferredType, t2: InferredType) {
        self.constraints.push((t1, t2));
    }
    
    fn solve_constraints(&self) -> Result<Substitution> {
        let mut subst = Substitution::new();
        
        for (t1, t2) in &self.constraints {
            let s1 = subst.apply(t1);
            let s2 = subst.apply(t2);
            
            let new_subst = Unifier::unify(&s1, &s2)?;
            subst = subst.compose(&new_subst);
        }
        
        Ok(subst)
    }
    
    fn apply_substitution(&mut self, subst: &Substitution) {
        for ty in self.expr_types.values_mut() {
            *ty = subst.apply(ty);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::semantic::Resolver;

    fn type_check_source(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let resolver = Resolver::new();
        let symbol_table = resolver.resolve(&program)?;
        
        let mut checker = TypeChecker::new(symbol_table);
        checker.check_program(&program)
    }

    #[test]
    fn test_simple_function() {
        let source = "fn main() { }";
        assert!(type_check_source(source).is_ok());
    }

    #[test]
    fn test_let_binding() {
        let source = "fn main() { let x: i32 = 42; }";
        assert!(type_check_source(source).is_ok());
    }

    #[test]
    fn test_arithmetic() {
        let source = "fn main() { let x = 1 + 2; }";
        assert!(type_check_source(source).is_ok());
    }

    #[test]
    fn test_function_call() {
        let source = r#"
            fn add(a: i32, b: i32) -> i32 { a }
            fn main() { let x = add(1, 2); }
        "#;
        assert!(type_check_source(source).is_ok());
    }

    #[test]
    fn test_if_expression() {
        let source = r#"
            fn main() {
                let x = if true { 1 } else { 2 };
            }
        "#;
        assert!(type_check_source(source).is_ok());
    }

    #[test]
    fn test_array_literal() {
        let source = "fn main() { let arr = [1, 2, 3]; }";
        assert!(type_check_source(source).is_ok());
    }
}
