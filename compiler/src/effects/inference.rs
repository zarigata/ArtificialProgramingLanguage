//! Effect inference

use std::collections::HashMap;
use crate::parser::ast::{Expr, Stmt, Function as AstFunction, UnOp};
use crate::ir::ssa::Function as IrFunction;
use crate::error::Result;
use super::effect::{EffectSet, EffectKind};

pub struct EffectInference {
    context: HashMap<String, EffectSet>,
}

impl EffectInference {
    pub fn new() -> Self {
        let mut context = HashMap::new();
        
        context.insert("print".to_string(), Self::io_write_effects());
        context.insert("println".to_string(), Self::io_write_effects());
        context.insert("read".to_string(), Self::io_read_effects());
        context.insert("write".to_string(), Self::io_write_effects());
        context.insert("malloc".to_string(), Self::alloc_effects());
        context.insert("free".to_string(), Self::free_effects());
        context.insert("spawn".to_string(), Self::async_spawn_effects());
        
        EffectInference { context }
    }
    
    fn io_read_effects() -> EffectSet {
        let mut set = EffectSet::new();
        set.insert(EffectKind::IoRead);
        set
    }
    
    fn io_write_effects() -> EffectSet {
        let mut set = EffectSet::new();
        set.insert(EffectKind::IoWrite);
        set
    }
    
    fn alloc_effects() -> EffectSet {
        let mut set = EffectSet::new();
        set.insert(EffectKind::Allocate);
        set
    }
    
    fn free_effects() -> EffectSet {
        let mut set = EffectSet::new();
        set.insert(EffectKind::Deallocate);
        set
    }
    
    fn async_spawn_effects() -> EffectSet {
        let mut set = EffectSet::new();
        set.insert(EffectKind::AsyncSpawn);
        set
    }
    
    pub fn infer_function(&mut self, func: &AstFunction) -> Result<EffectSet> {
        let mut effects = EffectSet::new();
        
        for stmt in &func.body {
            let stmt_effects = self.infer_stmt(stmt)?;
            effects.extend(&stmt_effects);
        }
        
        if effects.is_empty() {
            effects.insert(EffectKind::Pure);
        }
        
        self.context.insert(func.name.clone(), effects.clone());
        
        Ok(effects)
    }
    
    pub fn infer_stmt(&self, stmt: &Stmt) -> Result<EffectSet> {
        match stmt {
            Stmt::Let(_, _, Some(expr)) => self.infer_expr(expr),
            Stmt::Let(_, _, None) => Ok(EffectSet::pure()),
            Stmt::Expr(expr) => self.infer_expr(expr),
            Stmt::Return(Some(expr)) => self.infer_expr(expr),
            Stmt::Return(None) => Ok(EffectSet::pure()),
        }
    }
    
    pub fn infer_expr(&self, expr: &Expr) -> Result<EffectSet> {
        match expr {
            Expr::Literal(_) => Ok(EffectSet::pure()),
            Expr::Ident(_) => Ok(EffectSet::pure()),
            
            Expr::Binary(lhs, _, rhs) => {
                let mut effects = self.infer_expr(lhs)?;
                effects.extend(&self.infer_expr(rhs)?);
                Ok(effects)
            }
            
            Expr::Unary(op, operand) => {
                let mut effects = self.infer_expr(operand)?;
                if matches!(op, UnOp::Deref) {
                    effects.insert(EffectKind::StateRead);
                }
                Ok(effects)
            }
            
            Expr::Call(func, args) => {
                let mut effects = EffectSet::new();
                
                if let Expr::Ident(name) = func.as_ref() {
                    if let Some(func_effects) = self.context.get(name) {
                        effects.extend(func_effects);
                    } else {
                        effects.insert(EffectKind::Unknown);
                    }
                } else {
                    effects.extend(&self.infer_expr(func)?);
                }
                
                for arg in args {
                    effects.extend(&self.infer_expr(arg)?);
                }
                
                Ok(effects)
            }
            
            Expr::MethodCall(obj, _, args) => {
                let mut effects = self.infer_expr(obj)?;
                for arg in args {
                    effects.extend(&self.infer_expr(arg)?);
                }
                effects.insert(EffectKind::Unknown);
                Ok(effects)
            }
            
            Expr::Field(obj, _) => {
                let mut effects = self.infer_expr(obj)?;
                effects.insert(EffectKind::StateRead);
                Ok(effects)
            }
            
            Expr::Index(obj, index) => {
                let mut effects = self.infer_expr(obj)?;
                effects.extend(&self.infer_expr(index)?);
                effects.insert(EffectKind::StateRead);
                Ok(effects)
            }
            
            Expr::Block(stmts) => {
                let mut effects = EffectSet::new();
                for stmt in stmts {
                    effects.extend(&self.infer_stmt(stmt)?);
                }
                if effects.is_empty() {
                    effects.insert(EffectKind::Pure);
                }
                Ok(effects)
            }
            
            Expr::If(cond, then_block, else_block) => {
                let mut effects = self.infer_expr(cond)?;
                effects.extend(&self.infer_expr(then_block)?);
                if let Some(else_expr) = else_block {
                    effects.extend(&self.infer_expr(else_expr)?);
                }
                Ok(effects)
            }
            
            Expr::Match(scrutinee, arms) => {
                let mut effects = self.infer_expr(scrutinee)?;
                for arm in arms {
                    effects.extend(&self.infer_expr(&arm.body)?);
                }
                Ok(effects)
            }
            
            Expr::Loop(body) => self.infer_expr(body),
            
            Expr::While(cond, body) => {
                let mut effects = self.infer_expr(cond)?;
                effects.extend(&self.infer_expr(body)?);
                Ok(effects)
            }
            
            Expr::For(_, iter, body) => {
                let mut effects = self.infer_expr(iter)?;
                effects.extend(&self.infer_expr(body)?);
                Ok(effects)
            }
            
            Expr::Array(elements) => {
                let mut effects = EffectSet::new();
                for elem in elements {
                    effects.extend(&self.infer_expr(elem)?);
                }
                if !effects.is_empty() {
                    effects.insert(EffectKind::Allocate);
                }
                Ok(effects)
            }
            
            Expr::Tuple(elements) => {
                let mut effects = EffectSet::new();
                for elem in elements {
                    effects.extend(&self.infer_expr(elem)?);
                }
                Ok(effects)
            }
            
            Expr::StructLiteral(_, fields) => {
                let mut effects = EffectSet::new();
                for (_, value) in fields {
                    effects.extend(&self.infer_expr(value)?);
                }
                effects.insert(EffectKind::Allocate);
                Ok(effects)
            }
            
            Expr::Break(_) | Expr::Continue => Ok(EffectSet::pure()),
            
            Expr::Return(opt_expr) => {
                match opt_expr {
                    Some(expr) => self.infer_expr(expr),
                    None => Ok(EffectSet::pure()),
                }
            }
        }
    }
    
    pub fn infer_ir_function(&self, func: &IrFunction) -> Result<EffectSet> {
        let mut effects = EffectSet::new();
        
        for block in &func.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    crate::ir::instructions::Instruction::Load { .. } => {
                        effects.insert(EffectKind::StateRead);
                    }
                    crate::ir::instructions::Instruction::Store { .. } => {
                        effects.insert(EffectKind::StateWrite);
                    }
                    crate::ir::instructions::Instruction::Alloca { .. } => {
                        effects.insert(EffectKind::Allocate);
                    }
                    crate::ir::instructions::Instruction::Call { func: func_val, .. } => {
                        if let Some(crate::ir::ssa::Value::Global(name, _)) = 
                            func.values.get(func_val.0) {
                            if let Some(func_effects) = self.context.get(name) {
                                effects.extend(func_effects);
                            } else {
                                effects.insert(EffectKind::Unknown);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        
        if effects.is_empty() {
            effects.insert(EffectKind::Pure);
        }
        
        Ok(effects)
    }
    
    pub fn get_function_effects(&self, name: &str) -> Option<&EffectSet> {
        self.context.get(name)
    }
    
    pub fn register_function(&mut self, name: String, effects: EffectSet) {
        self.context.insert(name, effects);
    }
}

impl Default for EffectInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_creation() {
        let inference = EffectInference::new();
        assert!(inference.get_function_effects("print").is_some());
    }

    #[test]
    fn test_builtin_effects() {
        let inference = EffectInference::new();
        let print_effects = inference.get_function_effects("print").unwrap();
        assert!(print_effects.has_io());
    }
}
