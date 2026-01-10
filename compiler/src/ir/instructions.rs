//! IR instructions

use super::types::IrType;
use super::ssa::ValueId;
use std::fmt;

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    Shr,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// IR instruction
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// Binary operation: result = lhs op rhs
    Binary {
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: IrType,
    },
    
    /// Unary operation: result = op operand
    Unary {
        op: UnaryOp,
        operand: ValueId,
        ty: IrType,
    },
    
    /// Load from memory: result = *ptr
    Load {
        ptr: ValueId,
        ty: IrType,
    },
    
    /// Store to memory: *ptr = value
    Store {
        ptr: ValueId,
        value: ValueId,
    },
    
    /// Allocate stack memory
    Alloca {
        ty: IrType,
    },
    
    /// Get element pointer (GEP)
    GetElementPtr {
        ptr: ValueId,
        indices: Vec<ValueId>,
        ty: IrType,
    },
    
    /// Function call
    Call {
        func: ValueId,
        args: Vec<ValueId>,
        ty: IrType,
    },
    
    /// Return from function
    Return {
        value: Option<ValueId>,
    },
    
    /// Conditional branch
    Branch {
        cond: ValueId,
        then_block: usize,
        else_block: usize,
    },
    
    /// Unconditional jump
    Jump {
        target: usize,
    },
    
    /// Phi node for SSA
    Phi {
        incoming: Vec<(ValueId, usize)>, // (value, block_id)
        ty: IrType,
    },
    
    /// Type cast
    Cast {
        value: ValueId,
        from_ty: IrType,
        to_ty: IrType,
    },
    
    /// Select: result = cond ? true_val : false_val
    Select {
        cond: ValueId,
        true_val: ValueId,
        false_val: ValueId,
        ty: IrType,
    },
}

impl Instruction {
    /// Get the type of the instruction's result
    pub fn result_type(&self) -> Option<IrType> {
        match self {
            Instruction::Binary { ty, .. } => Some(ty.clone()),
            Instruction::Unary { ty, .. } => Some(ty.clone()),
            Instruction::Load { ty, .. } => Some(ty.clone()),
            Instruction::Store { .. } => None,
            Instruction::Alloca { ty } => Some(IrType::Pointer(Box::new(ty.clone()))),
            Instruction::GetElementPtr { ty, .. } => Some(IrType::Pointer(Box::new(ty.clone()))),
            Instruction::Call { ty, .. } => Some(ty.clone()),
            Instruction::Return { .. } => None,
            Instruction::Branch { .. } => None,
            Instruction::Jump { .. } => None,
            Instruction::Phi { ty, .. } => Some(ty.clone()),
            Instruction::Cast { to_ty, .. } => Some(to_ty.clone()),
            Instruction::Select { ty, .. } => Some(ty.clone()),
        }
    }
    
    /// Check if instruction is a terminator
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Return { .. } | Instruction::Branch { .. } | Instruction::Jump { .. }
        )
    }
    
    /// Get used values
    pub fn used_values(&self) -> Vec<ValueId> {
        match self {
            Instruction::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
            Instruction::Unary { operand, .. } => vec![*operand],
            Instruction::Load { ptr, .. } => vec![*ptr],
            Instruction::Store { ptr, value } => vec![*ptr, *value],
            Instruction::Alloca { .. } => vec![],
            Instruction::GetElementPtr { ptr, indices, .. } => {
                let mut vals = vec![*ptr];
                vals.extend(indices);
                vals
            }
            Instruction::Call { func, args, .. } => {
                let mut vals = vec![*func];
                vals.extend(args);
                vals
            }
            Instruction::Return { value } => value.iter().copied().collect(),
            Instruction::Branch { cond, .. } => vec![*cond],
            Instruction::Jump { .. } => vec![],
            Instruction::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
            Instruction::Cast { value, .. } => vec![*value],
            Instruction::Select { cond, true_val, false_val, .. } => {
                vec![*cond, *true_val, *false_val]
            }
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "add"),
            BinaryOp::Sub => write!(f, "sub"),
            BinaryOp::Mul => write!(f, "mul"),
            BinaryOp::Div => write!(f, "div"),
            BinaryOp::Rem => write!(f, "rem"),
            BinaryOp::And => write!(f, "and"),
            BinaryOp::Or => write!(f, "or"),
            BinaryOp::Xor => write!(f, "xor"),
            BinaryOp::Shl => write!(f, "shl"),
            BinaryOp::Shr => write!(f, "shr"),
            BinaryOp::Eq => write!(f, "eq"),
            BinaryOp::Ne => write!(f, "ne"),
            BinaryOp::Lt => write!(f, "lt"),
            BinaryOp::Le => write!(f, "le"),
            BinaryOp::Gt => write!(f, "gt"),
            BinaryOp::Ge => write!(f, "ge"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "neg"),
            UnaryOp::Not => write!(f, "not"),
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Binary { op, lhs, rhs, .. } => {
                write!(f, "{} v{}, v{}", op, lhs.0, rhs.0)
            }
            Instruction::Unary { op, operand, .. } => {
                write!(f, "{} v{}", op, operand.0)
            }
            Instruction::Load { ptr, .. } => {
                write!(f, "load v{}", ptr.0)
            }
            Instruction::Store { ptr, value } => {
                write!(f, "store v{}, v{}", ptr.0, value.0)
            }
            Instruction::Alloca { ty } => {
                write!(f, "alloca {}", ty)
            }
            Instruction::GetElementPtr { ptr, indices, .. } => {
                write!(f, "gep v{}", ptr.0)?;
                for idx in indices {
                    write!(f, ", v{}", idx.0)?;
                }
                Ok(())
            }
            Instruction::Call { func, args, .. } => {
                write!(f, "call v{}", func.0)?;
                for arg in args {
                    write!(f, ", v{}", arg.0)?;
                }
                Ok(())
            }
            Instruction::Return { value } => {
                if let Some(val) = value {
                    write!(f, "ret v{}", val.0)
                } else {
                    write!(f, "ret void")
                }
            }
            Instruction::Branch { cond, then_block, else_block } => {
                write!(f, "br v{}, bb{}, bb{}", cond.0, then_block, else_block)
            }
            Instruction::Jump { target } => {
                write!(f, "jmp bb{}", target)
            }
            Instruction::Phi { incoming, .. } => {
                write!(f, "phi")?;
                for (i, (val, block)) in incoming.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " [v{}, bb{}]", val.0, block)?;
                }
                Ok(())
            }
            Instruction::Cast { value, from_ty, to_ty } => {
                write!(f, "cast v{} from {} to {}", value.0, from_ty, to_ty)
            }
            Instruction::Select { cond, true_val, false_val, .. } => {
                write!(f, "select v{}, v{}, v{}", cond.0, true_val.0, false_val.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_instruction() {
        let inst = Instruction::Binary {
            op: BinaryOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: IrType::I32,
        };
        
        assert_eq!(inst.result_type(), Some(IrType::I32));
        assert!(!inst.is_terminator());
        assert_eq!(inst.used_values(), vec![ValueId(0), ValueId(1)]);
    }

    #[test]
    fn test_return_instruction() {
        let inst = Instruction::Return {
            value: Some(ValueId(0)),
        };
        
        assert_eq!(inst.result_type(), None);
        assert!(inst.is_terminator());
    }

    #[test]
    fn test_phi_instruction() {
        let inst = Instruction::Phi {
            incoming: vec![(ValueId(0), 0), (ValueId(1), 1)],
            ty: IrType::I32,
        };
        
        assert_eq!(inst.result_type(), Some(IrType::I32));
        assert!(!inst.is_terminator());
    }
}
