// VeZ Compile-Time Evaluation Engine
// Executes code at compile time for constant folding and metaprogramming

use crate::parser::ast::*;
use crate::semantic::types::Type;
use crate::error::{Error, Result};
use std::collections::HashMap;

// Compile-time value
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<ConstValue>),
    Struct(HashMap<String, ConstValue>),
    Function(String),
}

impl ConstValue {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ConstValue::Int(n) => Some(*n),
            _ => None,
        }
    }
    
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConstValue::Float(f) => Some(*f),
            _ => None,
        }
    }
    
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConstValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

// Compile-time evaluator
pub struct ConstEvaluator {
    constants: HashMap<String, ConstValue>,
    functions: HashMap<String, Function>,
}

impl ConstEvaluator {
    pub fn new() -> Self {
        ConstEvaluator {
            constants: HashMap::new(),
            functions: HashMap::new(),
        }
    }
    
    // Evaluate expression at compile time
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<ConstValue> {
        match expr {
            Expr::Literal { value, .. } => self.eval_literal(value),
            
            Expr::Variable { name, .. } => {
                self.constants.get(name)
                    .cloned()
                    .ok_or_else(|| Error::new(
                        format!("Undefined constant: {}", name),
                        Span::dummy()
                    ))
            }
            
            Expr::Binary { op, left, right, .. } => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                self.eval_binary_op(*op, &left_val, &right_val)
            }
            
            Expr::Unary { op, operand, .. } => {
                let val = self.eval_expr(operand)?;
                self.eval_unary_op(*op, &val)
            }
            
            Expr::Call { func, args, .. } => {
                self.eval_call(func, args)
            }
            
            Expr::If { condition, then_branch, else_branch, .. } => {
                let cond_val = self.eval_expr(condition)?;
                if cond_val.as_bool().unwrap_or(false) {
                    self.eval_expr(then_branch)
                } else if let Some(else_expr) = else_branch {
                    self.eval_expr(else_expr)
                } else {
                    Ok(ConstValue::Bool(false))
                }
            }
            
            Expr::Array { elements, .. } => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expr(elem)?);
                }
                Ok(ConstValue::Array(values))
            }
            
            _ => Err(Error::new(
                "Expression cannot be evaluated at compile time",
                Span::dummy()
            )),
        }
    }
    
    fn eval_literal(&self, lit: &Literal) -> Result<ConstValue> {
        match lit {
            Literal::Int(n) => Ok(ConstValue::Int(*n)),
            Literal::Float(f) => Ok(ConstValue::Float(*f)),
            Literal::Bool(b) => Ok(ConstValue::Bool(*b)),
            Literal::String(s) => Ok(ConstValue::String(s.clone())),
            Literal::Char(c) => Ok(ConstValue::Int(*c as i64)),
        }
    }
    
    fn eval_binary_op(&self, op: BinaryOp, left: &ConstValue, right: &ConstValue) -> Result<ConstValue> {
        match (left, right) {
            (ConstValue::Int(l), ConstValue::Int(r)) => {
                let result = match op {
                    BinaryOp::Add => ConstValue::Int(l + r),
                    BinaryOp::Sub => ConstValue::Int(l - r),
                    BinaryOp::Mul => ConstValue::Int(l * r),
                    BinaryOp::Div => {
                        if *r == 0 {
                            return Err(Error::new("Division by zero", Span::dummy()));
                        }
                        ConstValue::Int(l / r)
                    }
                    BinaryOp::Mod => ConstValue::Int(l % r),
                    BinaryOp::Eq => ConstValue::Bool(l == r),
                    BinaryOp::Ne => ConstValue::Bool(l != r),
                    BinaryOp::Lt => ConstValue::Bool(l < r),
                    BinaryOp::Le => ConstValue::Bool(l <= r),
                    BinaryOp::Gt => ConstValue::Bool(l > r),
                    BinaryOp::Ge => ConstValue::Bool(l >= r),
                    BinaryOp::BitAnd => ConstValue::Int(l & r),
                    BinaryOp::BitOr => ConstValue::Int(l | r),
                    BinaryOp::BitXor => ConstValue::Int(l ^ r),
                    BinaryOp::Shl => ConstValue::Int(l << r),
                    BinaryOp::Shr => ConstValue::Int(l >> r),
                    _ => return Err(Error::new("Unsupported binary operation", Span::dummy())),
                };
                Ok(result)
            }
            
            (ConstValue::Float(l), ConstValue::Float(r)) => {
                let result = match op {
                    BinaryOp::Add => ConstValue::Float(l + r),
                    BinaryOp::Sub => ConstValue::Float(l - r),
                    BinaryOp::Mul => ConstValue::Float(l * r),
                    BinaryOp::Div => ConstValue::Float(l / r),
                    BinaryOp::Eq => ConstValue::Bool(l == r),
                    BinaryOp::Ne => ConstValue::Bool(l != r),
                    BinaryOp::Lt => ConstValue::Bool(l < r),
                    BinaryOp::Le => ConstValue::Bool(l <= r),
                    BinaryOp::Gt => ConstValue::Bool(l > r),
                    BinaryOp::Ge => ConstValue::Bool(l >= r),
                    _ => return Err(Error::new("Unsupported binary operation", Span::dummy())),
                };
                Ok(result)
            }
            
            (ConstValue::Bool(l), ConstValue::Bool(r)) => {
                let result = match op {
                    BinaryOp::And => ConstValue::Bool(*l && *r),
                    BinaryOp::Or => ConstValue::Bool(*l || *r),
                    BinaryOp::Eq => ConstValue::Bool(l == r),
                    BinaryOp::Ne => ConstValue::Bool(l != r),
                    _ => return Err(Error::new("Unsupported binary operation", Span::dummy())),
                };
                Ok(result)
            }
            
            _ => Err(Error::new("Type mismatch in binary operation", Span::dummy())),
        }
    }
    
    fn eval_unary_op(&self, op: UnaryOp, val: &ConstValue) -> Result<ConstValue> {
        match val {
            ConstValue::Int(n) => {
                let result = match op {
                    UnaryOp::Neg => ConstValue::Int(-n),
                    UnaryOp::Not => ConstValue::Bool(*n == 0),
                    UnaryOp::BitNot => ConstValue::Int(!n),
                    _ => return Err(Error::new("Unsupported unary operation", Span::dummy())),
                };
                Ok(result)
            }
            
            ConstValue::Float(f) => {
                let result = match op {
                    UnaryOp::Neg => ConstValue::Float(-f),
                    _ => return Err(Error::new("Unsupported unary operation", Span::dummy())),
                };
                Ok(result)
            }
            
            ConstValue::Bool(b) => {
                let result = match op {
                    UnaryOp::Not => ConstValue::Bool(!b),
                    _ => return Err(Error::new("Unsupported unary operation", Span::dummy())),
                };
                Ok(result)
            }
            
            _ => Err(Error::new("Type mismatch in unary operation", Span::dummy())),
        }
    }
    
    fn eval_call(&mut self, func: &Expr, args: &[Expr]) -> Result<ConstValue> {
        // Get function name
        let func_name = match func {
            Expr::Variable { name, .. } => name,
            _ => return Err(Error::new("Invalid function call", Span::dummy())),
        };
        
        // Evaluate arguments
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.eval_expr(arg)?);
        }
        
        // Call built-in functions
        match func_name.as_str() {
            "abs" => self.builtin_abs(&arg_values),
            "min" => self.builtin_min(&arg_values),
            "max" => self.builtin_max(&arg_values),
            "pow" => self.builtin_pow(&arg_values),
            "sqrt" => self.builtin_sqrt(&arg_values),
            _ => Err(Error::new(
                format!("Unknown compile-time function: {}", func_name),
                Span::dummy()
            )),
        }
    }
    
    fn builtin_abs(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 1 {
            return Err(Error::new("abs expects 1 argument", Span::dummy()));
        }
        
        match &args[0] {
            ConstValue::Int(n) => Ok(ConstValue::Int(n.abs())),
            ConstValue::Float(f) => Ok(ConstValue::Float(f.abs())),
            _ => Err(Error::new("abs expects numeric argument", Span::dummy())),
        }
    }
    
    fn builtin_min(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 2 {
            return Err(Error::new("min expects 2 arguments", Span::dummy()));
        }
        
        match (&args[0], &args[1]) {
            (ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(*a.min(b))),
            (ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a.min(*b))),
            _ => Err(Error::new("min expects numeric arguments", Span::dummy())),
        }
    }
    
    fn builtin_max(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 2 {
            return Err(Error::new("max expects 2 arguments", Span::dummy()));
        }
        
        match (&args[0], &args[1]) {
            (ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(*a.max(b))),
            (ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a.max(*b))),
            _ => Err(Error::new("max expects numeric arguments", Span::dummy())),
        }
    }
    
    fn builtin_pow(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 2 {
            return Err(Error::new("pow expects 2 arguments", Span::dummy()));
        }
        
        match (&args[0], &args[1]) {
            (ConstValue::Float(base), ConstValue::Float(exp)) => {
                Ok(ConstValue::Float(base.powf(*exp)))
            }
            (ConstValue::Int(base), ConstValue::Int(exp)) => {
                if *exp < 0 {
                    return Err(Error::new("Negative exponent for integer pow", Span::dummy()));
                }
                Ok(ConstValue::Int(base.pow(*exp as u32)))
            }
            _ => Err(Error::new("pow expects numeric arguments", Span::dummy())),
        }
    }
    
    fn builtin_sqrt(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 1 {
            return Err(Error::new("sqrt expects 1 argument", Span::dummy()));
        }
        
        match &args[0] {
            ConstValue::Float(f) => {
                if *f < 0.0 {
                    return Err(Error::new("sqrt of negative number", Span::dummy()));
                }
                Ok(ConstValue::Float(f.sqrt()))
            }
            _ => Err(Error::new("sqrt expects float argument", Span::dummy())),
        }
    }
    
    // Define a compile-time constant
    pub fn define_const(&mut self, name: String, value: ConstValue) {
        self.constants.insert(name, value);
    }
    
    // Register a function for compile-time execution
    pub fn register_function(&mut self, func: Function) {
        self.functions.insert(func.name.clone(), func);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_eval_literal() {
        let mut eval = ConstEvaluator::new();
        
        let expr = Expr::Literal {
            value: Literal::Int(42),
            span: Span::dummy(),
        };
        
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(result, ConstValue::Int(42));
    }
    
    #[test]
    fn test_eval_binary_add() {
        let mut eval = ConstEvaluator::new();
        
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Literal {
                value: Literal::Int(10),
                span: Span::dummy(),
            }),
            right: Box::new(Expr::Literal {
                value: Literal::Int(32),
                span: Span::dummy(),
            }),
            span: Span::dummy(),
        };
        
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(result, ConstValue::Int(42));
    }
    
    #[test]
    fn test_builtin_abs() {
        let mut eval = ConstEvaluator::new();
        let args = vec![ConstValue::Int(-42)];
        let result = eval.builtin_abs(&args).unwrap();
        assert_eq!(result, ConstValue::Int(42));
    }
}
