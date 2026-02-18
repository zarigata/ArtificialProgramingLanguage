//! Constant folding optimization pass

use crate::ir::ssa::{Module, Function as IrFunction, Value, Constant};
use crate::ir::instructions::{Instruction, BinaryOp, UnaryOp};
use crate::error::Result;
use super::pass_manager::OptimizationPass;

/// Constant folding pass
pub struct ConstantFolding {
    changed: bool,
}

impl ConstantFolding {
    pub fn new() -> Self {
        ConstantFolding { changed: false }
    }
    
    /// Fold a binary operation on constants
    fn fold_binary(&self, op: BinaryOp, lhs: &Constant, rhs: &Constant) -> Option<Constant> {
        match (lhs, rhs) {
            (Constant::Int(l, ty), Constant::Int(r, _)) => {
                let result = match op {
                    BinaryOp::Add => l.wrapping_add(*r),
                    BinaryOp::Sub => l.wrapping_sub(*r),
                    BinaryOp::Mul => l.wrapping_mul(*r),
                    BinaryOp::Div if *r != 0 => l / r,
                    BinaryOp::Rem if *r != 0 => l % r,
                    BinaryOp::And => l & r,
                    BinaryOp::Or => l | r,
                    BinaryOp::Xor => l ^ r,
                    BinaryOp::Shl => l << (r & 127),
                    BinaryOp::Shr => l >> (r & 127),
                    BinaryOp::Eq => return Some(Constant::Bool(l == r)),
                    BinaryOp::Ne => return Some(Constant::Bool(l != r)),
                    BinaryOp::Lt => return Some(Constant::Bool(l < r)),
                    BinaryOp::Le => return Some(Constant::Bool(l <= r)),
                    BinaryOp::Gt => return Some(Constant::Bool(l > r)),
                    BinaryOp::Ge => return Some(Constant::Bool(l >= r)),
                    _ => return None,
                };
                Some(Constant::Int(result, ty.clone()))
            }
            (Constant::Float(l, ty), Constant::Float(r, _)) => {
                let result = match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => l / r,
                    BinaryOp::Eq => return Some(Constant::Bool(l == r)),
                    BinaryOp::Ne => return Some(Constant::Bool(l != r)),
                    BinaryOp::Lt => return Some(Constant::Bool(l < r)),
                    BinaryOp::Le => return Some(Constant::Bool(l <= r)),
                    BinaryOp::Gt => return Some(Constant::Bool(l > r)),
                    BinaryOp::Ge => return Some(Constant::Bool(l >= r)),
                    _ => return None,
                };
                Some(Constant::Float(result, ty.clone()))
            }
            (Constant::Bool(l), Constant::Bool(r)) => {
                let result = match op {
                    BinaryOp::And => *l && *r,
                    BinaryOp::Or => *l || *r,
                    BinaryOp::Xor => *l ^ *r,
                    BinaryOp::Eq => l == r,
                    BinaryOp::Ne => l != r,
                    _ => return None,
                };
                Some(Constant::Bool(result))
            }
            _ => None,
        }
    }
    
    /// Fold a unary operation on a constant
    fn fold_unary(&self, op: UnaryOp, operand: &Constant) -> Option<Constant> {
        match operand {
            Constant::Int(val, ty) => {
                let result = match op {
                    UnaryOp::Neg => -val,
                    UnaryOp::Not => !val,
                };
                Some(Constant::Int(result, ty.clone()))
            }
            Constant::Float(val, ty) => {
                let result = match op {
                    UnaryOp::Neg => -val,
                    _ => return None,
                };
                Some(Constant::Float(result, ty.clone()))
            }
            Constant::Bool(val) => {
                let result = match op {
                    UnaryOp::Not => !val,
                    _ => return None,
                };
                Some(Constant::Bool(result))
            }
            _ => None,
        }
    }
    
    /// Apply algebraic simplifications
    fn simplify_binary(&self, op: BinaryOp, lhs: &Value, rhs: &Value) -> Option<Value> {
        // x + 0 = x
        if let (BinaryOp::Add, Value::Constant(Constant::Int(0, _))) = (op, rhs) {
            return Some(lhs.clone());
        }
        
        // 0 + x = x
        if let (BinaryOp::Add, Value::Constant(Constant::Int(0, _))) = (op, lhs) {
            return Some(rhs.clone());
        }
        
        // x * 0 = 0
        if let (BinaryOp::Mul, Value::Constant(c @ Constant::Int(0, _))) = (op, rhs) {
            return Some(Value::Constant(c.clone()));
        }
        
        // 0 * x = 0
        if let (BinaryOp::Mul, Value::Constant(c @ Constant::Int(0, _))) = (op, lhs) {
            return Some(Value::Constant(c.clone()));
        }
        
        // x * 1 = x
        if let (BinaryOp::Mul, Value::Constant(Constant::Int(1, _))) = (op, rhs) {
            return Some(lhs.clone());
        }
        
        // 1 * x = x
        if let (BinaryOp::Mul, Value::Constant(Constant::Int(1, _))) = (op, lhs) {
            return Some(rhs.clone());
        }
        
        // x - 0 = x
        if let (BinaryOp::Sub, Value::Constant(Constant::Int(0, _))) = (op, rhs) {
            return Some(lhs.clone());
        }
        
        // x / 1 = x
        if let (BinaryOp::Div, Value::Constant(Constant::Int(1, _))) = (op, rhs) {
            return Some(lhs.clone());
        }
        
        None
    }
}

impl Default for ConstantFolding {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "Constant Folding"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        self.changed = false;
        
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, function: &mut IrFunction) -> Result<bool> {
        self.changed = false;
        
        // Collect replacements to avoid borrow conflicts
        let mut replacements = Vec::new();
        
        for block in &function.blocks {
            for (value_id, inst) in &block.instructions {
                match inst {
                    Instruction::Binary { op, lhs, rhs, ty: _ } => {
                        // Try to get constant values
                        let lhs_val = function.get_value(*lhs);
                        let rhs_val = function.get_value(*rhs);
                        
                        if let (Some(Value::Constant(l)), Some(Value::Constant(r))) = (lhs_val, rhs_val) {
                            // Both operands are constants - fold them
                            if let Some(result) = self.fold_binary(*op, l, r) {
                                replacements.push((*value_id, Value::Constant(result)));
                            }
                        } else if let (Some(l), Some(r)) = (lhs_val, rhs_val) {
                            // Try algebraic simplification
                            if let Some(simplified) = self.simplify_binary(*op, l, r) {
                                replacements.push((*value_id, simplified));
                            }
                        }
                    }
                    Instruction::Unary { op, operand, ty: _ } => {
                        // Try to get constant value
                        if let Some(Value::Constant(c)) = function.get_value(*operand) {
                            // Operand is constant - fold it
                            if let Some(result) = self.fold_unary(*op, c) {
                                replacements.push((*value_id, Value::Constant(result)));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        
        // Apply replacements
        if !replacements.is_empty() {
            self.changed = true;
            for (value_id, new_value) in replacements {
                function.values[value_id.0] = new_value;
            }
        }
        
        Ok(self.changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ssa::ValueId;

    #[test]
    fn test_fold_add() {
        let cf = ConstantFolding::new();
        let lhs = Constant::Int(5, IrType::I32);
        let rhs = Constant::Int(3, IrType::I32);
        
        let result = cf.fold_binary(BinaryOp::Add, &lhs, &rhs);
        assert_eq!(result, Some(Constant::Int(8, IrType::I32)));
    }

    #[test]
    fn test_fold_mul() {
        let cf = ConstantFolding::new();
        let lhs = Constant::Int(4, IrType::I32);
        let rhs = Constant::Int(7, IrType::I32);
        
        let result = cf.fold_binary(BinaryOp::Mul, &lhs, &rhs);
        assert_eq!(result, Some(Constant::Int(28, IrType::I32)));
    }

    #[test]
    fn test_fold_comparison() {
        let cf = ConstantFolding::new();
        let lhs = Constant::Int(5, IrType::I32);
        let rhs = Constant::Int(3, IrType::I32);
        
        let result = cf.fold_binary(BinaryOp::Gt, &lhs, &rhs);
        assert_eq!(result, Some(Constant::Bool(true)));
    }

    #[test]
    fn test_fold_neg() {
        let cf = ConstantFolding::new();
        let operand = Constant::Int(42, IrType::I32);
        
        let result = cf.fold_unary(UnaryOp::Neg, &operand);
        assert_eq!(result, Some(Constant::Int(-42, IrType::I32)));
    }

    #[test]
    fn test_fold_not() {
        let cf = ConstantFolding::new();
        let operand = Constant::Bool(true);
        
        let result = cf.fold_unary(UnaryOp::Not, &operand);
        assert_eq!(result, Some(Constant::Bool(false)));
    }

    #[test]
    fn test_simplify_add_zero() {
        let cf = ConstantFolding::new();
        let x = Value::Parameter(0, IrType::I32);
        let zero = Value::Constant(Constant::Int(0, IrType::I32));
        
        let result = cf.simplify_binary(BinaryOp::Add, &x, &zero);
        assert!(result.is_some());
    }

    #[test]
    fn test_simplify_mul_one() {
        let cf = ConstantFolding::new();
        let x = Value::Parameter(0, IrType::I32);
        let one = Value::Constant(Constant::Int(1, IrType::I32));
        
        let result = cf.simplify_binary(BinaryOp::Mul, &x, &one);
        assert!(result.is_some());
    }
}
