//! Common subexpression elimination optimization pass

use crate::ir::ssa::{Module, Function as IrFunction, ValueId};
use crate::ir::instructions::Instruction;
use crate::error::Result;
use super::pass_manager::OptimizationPass;
use std::collections::HashMap;

/// Common subexpression elimination pass
pub struct CommonSubexprElimination {
    changed: bool,
}

impl CommonSubexprElimination {
    pub fn new() -> Self {
        CommonSubexprElimination { changed: false }
    }
    
    /// Get a hash key for an instruction
    fn instruction_key(&self, inst: &Instruction) -> Option<String> {
        match inst {
            Instruction::Binary { op, lhs, rhs, ty } => {
                Some(format!("bin_{:?}_{:?}_{:?}_{:?}", op, lhs, rhs, ty))
            }
            Instruction::Unary { op, operand, ty } => {
                Some(format!("un_{:?}_{:?}_{:?}", op, operand, ty))
            }
            Instruction::Load { ptr, ty } => {
                Some(format!("load_{:?}_{:?}", ptr, ty))
            }
            Instruction::GetElementPtr { ptr, indices, ty } => {
                Some(format!("gep_{:?}_{:?}_{:?}", ptr, indices, ty))
            }
            // Don't CSE instructions with side effects
            _ => None,
        }
    }
}

impl Default for CommonSubexprElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for CommonSubexprElimination {
    fn name(&self) -> &str {
        "Common Subexpression Elimination"
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
        
        for block in &mut function.blocks {
            let mut expr_map: HashMap<String, ValueId> = HashMap::new();
            let mut replacements: HashMap<ValueId, ValueId> = HashMap::new();
            
            // Find common subexpressions
            for (value_id, inst) in &block.instructions {
                if let Some(key) = self.instruction_key(inst) {
                    if let Some(&existing_id) = expr_map.get(&key) {
                        // Found a common subexpression
                        replacements.insert(*value_id, existing_id);
                        self.changed = true;
                    } else {
                        expr_map.insert(key, *value_id);
                    }
                }
            }
            
            // Apply replacements
            if !replacements.is_empty() {
                for (_value_id, inst) in &mut block.instructions {
                    // Replace used values with their equivalents
                    match inst {
                        Instruction::Binary { lhs, rhs, .. } => {
                            if let Some(&new_lhs) = replacements.get(lhs) {
                                *lhs = new_lhs;
                            }
                            if let Some(&new_rhs) = replacements.get(rhs) {
                                *rhs = new_rhs;
                            }
                        }
                        Instruction::Unary { operand, .. } => {
                            if let Some(&new_operand) = replacements.get(operand) {
                                *operand = new_operand;
                            }
                        }
                        Instruction::Load { ptr, .. } => {
                            if let Some(&new_ptr) = replacements.get(ptr) {
                                *ptr = new_ptr;
                            }
                        }
                        Instruction::Store { ptr, value } => {
                            if let Some(&new_ptr) = replacements.get(ptr) {
                                *ptr = new_ptr;
                            }
                            if let Some(&new_value) = replacements.get(value) {
                                *value = new_value;
                            }
                        }
                        Instruction::Call { func, args, .. } => {
                            if let Some(&new_func) = replacements.get(func) {
                                *func = new_func;
                            }
                            for arg in args {
                                if let Some(&new_arg) = replacements.get(arg) {
                                    *arg = new_arg;
                                }
                            }
                        }
                        Instruction::Return { value } => {
                            if let Some(val) = value {
                                if let Some(&new_val) = replacements.get(val) {
                                    *val = new_val;
                                }
                            }
                        }
                        Instruction::Branch { cond, .. } => {
                            if let Some(&new_cond) = replacements.get(cond) {
                                *cond = new_cond;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        
        Ok(self.changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instructions::BinaryOp;
    use crate::ir::types::IrType;

    #[test]
    fn test_cse_creation() {
        let cse = CommonSubexprElimination::new();
        assert!(!cse.changed);
    }

    #[test]
    fn test_instruction_key() {
        let cse = CommonSubexprElimination::new();
        
        let inst1 = Instruction::Binary {
            op: BinaryOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: IrType::I32,
        };
        
        let inst2 = Instruction::Binary {
            op: BinaryOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: IrType::I32,
        };
        
        let key1 = cse.instruction_key(&inst1);
        let key2 = cse.instruction_key(&inst2);
        
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_no_key_for_side_effects() {
        let cse = CommonSubexprElimination::new();
        
        let store = Instruction::Store {
            ptr: ValueId(0),
            value: ValueId(1),
        };
        
        assert!(cse.instruction_key(&store).is_none());
    }
}
