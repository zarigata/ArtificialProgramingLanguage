// VeZ Compile-Time Evaluation Engine
// Executes code at compile time for constant folding and metaprogramming

use crate::parser::ast::{Function, Literal, Expr, BinOp, UnOp};
use crate::span::Span;
use crate::error::{Error, Result, ErrorKind};
use crate::parser::ast::Type;
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
            Expr::Literal(value) => self.eval_literal(value),
            
            Expr::Ident(name) => {
                self.constants.get(name)
                    .cloned()
                    .ok_or_else(|| Error::new(
                        ErrorKind::UndefinedSymbol,
                        format!("Undefined constant: {}", name)
                    ).with_span(Span::dummy()))
            }
            
            Expr::Binary(left, op, right) => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                self.eval_binary_op(*op, &left_val, &right_val)
            }
            
            Expr::Unary(op, operand) => {
                let val = self.eval_expr(operand)?;
                self.eval_unary_op(*op, &val)
            }
            
            Expr::Call(func, args) => {
                self.eval_call(func, args)
            }
            
            Expr::If(condition, then_branch, else_branch) => {
                let cond_val = self.eval_expr(condition)?;
                if cond_val.as_bool().unwrap_or(false) {
                    self.eval_expr(then_branch)
                } else if let Some(else_expr) = else_branch {
                    self.eval_expr(else_expr)
                } else {
                    Ok(ConstValue::Bool(false))
                }
            }
            
            Expr::Array(elements) => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expr(elem)?);
                }
                Ok(ConstValue::Array(values))
            }
            
            _ => Err(Error::new(
                ErrorKind::InvalidSyntax,
                "Expression cannot be evaluated at compile time".to_string()
            ).with_span(Span::dummy())),
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
    
    fn eval_binary_op(&self, op: BinOp, left: &ConstValue, right: &ConstValue) -> Result<ConstValue> {
        match (left, right) {
            (ConstValue::Int(l), ConstValue::Int(r)) => {
                let result = match op {
                    BinOp::Add => ConstValue::Int(l + r),
                    BinOp::Sub => ConstValue::Int(l - r),
                    BinOp::Mul => ConstValue::Int(l * r),
                    BinOp::Div => {
                        if *r == 0 {
                            return Err(Error::new(ErrorKind::InternalError, "Division by zero".to_string()).with_span(Span::dummy()));
                        }
                        ConstValue::Int(l / r)
                    }
                    BinOp::Mod => ConstValue::Int(l % r),
                    BinOp::Eq => ConstValue::Bool(l == r),
                    BinOp::Ne => ConstValue::Bool(l != r),
                    BinOp::Lt => ConstValue::Bool(l < r),
                    BinOp::Le => ConstValue::Bool(l <= r),
                    BinOp::Gt => ConstValue::Bool(l > r),
                    BinOp::Ge => ConstValue::Bool(l >= r),
                    _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Unsupported binary operation".to_string()).with_span(Span::dummy())),
                };
                Ok(result)
            }
            
            (ConstValue::Float(l), ConstValue::Float(r)) => {
                let result = match op {
                    BinOp::Add => ConstValue::Float(l + r),
                    BinOp::Sub => ConstValue::Float(l - r),
                    BinOp::Mul => ConstValue::Float(l * r),
                    BinOp::Div => ConstValue::Float(l / r),
                    BinOp::Eq => ConstValue::Bool(l == r),
                    BinOp::Ne => ConstValue::Bool(l != r),
                    BinOp::Lt => ConstValue::Bool(l < r),
                    BinOp::Le => ConstValue::Bool(l <= r),
                    BinOp::Gt => ConstValue::Bool(l > r),
                    BinOp::Ge => ConstValue::Bool(l >= r),
                    _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Unsupported binary operation".to_string()).with_span(Span::dummy())),
                };
                Ok(result)
            }
            
            (ConstValue::Bool(l), ConstValue::Bool(r)) => {
                let result = match op {
                    BinOp::And => ConstValue::Bool(*l && *r),
                    BinOp::Or => ConstValue::Bool(*l || *r),
                    BinOp::Eq => ConstValue::Bool(l == r),
                    BinOp::Ne => ConstValue::Bool(l != r),
                    _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Unsupported binary operation".to_string()).with_span(Span::dummy())),
                };
                Ok(result)
            }
            
            _ => Err(Error::new(ErrorKind::TypeMismatch, "Type mismatch in binary operation".to_string()).with_span(Span::dummy())),
        }
    }
    
    fn eval_unary_op(&self, op: UnOp, val: &ConstValue) -> Result<ConstValue> {
        match val {
            ConstValue::Int(n) => {
                let result = match op {
                    UnOp::Neg => ConstValue::Int(-n),
                    UnOp::Not => ConstValue::Bool(*n == 0),
                    _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Unsupported unary operation".to_string()).with_span(Span::dummy())),
                };
                Ok(result)
            }
            
            ConstValue::Float(f) => {
                let result = match op {
                    UnOp::Neg => ConstValue::Float(-f),
                    _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Unsupported unary operation".to_string()).with_span(Span::dummy())),
                };
                Ok(result)
            }
            
            ConstValue::Bool(b) => {
                let result = match op {
                    UnOp::Not => ConstValue::Bool(!b),
                    _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Unsupported unary operation".to_string()).with_span(Span::dummy())),
                };
                Ok(result)
            }
            
            _ => Err(Error::new(ErrorKind::TypeMismatch, "Type mismatch in unary operation".to_string()).with_span(Span::dummy())),
        }
    }
    
    fn eval_call(&mut self, func: &Expr, args: &[Expr]) -> Result<ConstValue> {
        // Get function name
        let func_name = match func {
            Expr::Ident(name) => name,
            _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Invalid function call".to_string()).with_span(Span::dummy())),
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
                ErrorKind::UndefinedSymbol,
                format!("Unknown compile-time function: {}", func_name)
            ).with_span(Span::dummy())),
        }
    }
    
    fn builtin_abs(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 1 {
            return Err(Error::new(ErrorKind::InvalidSyntax, "abs expects 1 argument".to_string()).with_span(Span::dummy()));
        }
        
        match &args[0] {
            ConstValue::Int(n) => Ok(ConstValue::Int(n.abs())),
            ConstValue::Float(f) => Ok(ConstValue::Float(f.abs())),
            _ => Err(Error::new(ErrorKind::TypeMismatch, "abs expects numeric argument".to_string()).with_span(Span::dummy())),
        }
    }
    
    fn builtin_min(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 2 {
            return Err(Error::new(ErrorKind::InvalidSyntax, "min expects 2 arguments".to_string()).with_span(Span::dummy()));
        }
        
        match (&args[0], &args[1]) {
            (ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(*a.min(b))),
            (ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a.min(*b))),
            _ => Err(Error::new(ErrorKind::TypeMismatch, "min expects numeric arguments".to_string()).with_span(Span::dummy())),
        }
    }
    
    fn builtin_max(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 2 {
            return Err(Error::new(ErrorKind::InvalidSyntax, "max expects 2 arguments".to_string()).with_span(Span::dummy()));
        }
        
        match (&args[0], &args[1]) {
            (ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(*a.max(b))),
            (ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a.max(*b))),
            _ => Err(Error::new(ErrorKind::TypeMismatch, "max expects numeric arguments".to_string()).with_span(Span::dummy())),
        }
    }
    
    fn builtin_pow(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 2 {
            return Err(Error::new(ErrorKind::InvalidSyntax, "pow expects 2 arguments".to_string()).with_span(Span::dummy()));
        }
        
        match (&args[0], &args[1]) {
            (ConstValue::Float(base), ConstValue::Float(exp)) => {
                Ok(ConstValue::Float(base.powf(*exp)))
            }
            (ConstValue::Int(base), ConstValue::Int(exp)) => {
                if *exp < 0 {
                    return Err(Error::new(ErrorKind::InvalidSyntax, "Negative exponent for integer pow".to_string()).with_span(Span::dummy()));
                }
                Ok(ConstValue::Int(base.pow(*exp as u32)))
            }
            _ => Err(Error::new(ErrorKind::TypeMismatch, "pow expects numeric arguments".to_string()).with_span(Span::dummy())),
        }
    }
    
    fn builtin_sqrt(&self, args: &[ConstValue]) -> Result<ConstValue> {
        if args.len() != 1 {
            return Err(Error::new(ErrorKind::InvalidSyntax, "sqrt expects 1 argument".to_string()).with_span(Span::dummy()));
        }
        
        match &args[0] {
            ConstValue::Float(f) => {
                if *f < 0.0 {
                    return Err(Error::new(ErrorKind::InvalidSyntax, "sqrt of negative number".to_string()).with_span(Span::dummy()));
                }
                Ok(ConstValue::Float(f.sqrt()))
            }
            _ => Err(Error::new(ErrorKind::TypeMismatch, "sqrt expects float argument".to_string()).with_span(Span::dummy())),
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
        
        let expr = Expr::Literal(Literal::Int(42));
        
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(result, ConstValue::Int(42));
    }
    
    #[test]
    fn test_eval_binary_add() {
        let mut eval = ConstEvaluator::new();
        
        let expr = Expr::Binary(
            Box::new(Expr::Literal(Literal::Int(10))),
            BinOp::Add,
            Box::new(Expr::Literal(Literal::Int(32))),
        );
        
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
