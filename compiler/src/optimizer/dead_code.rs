//! Dead code elimination optimization pass

use crate::ir::ssa::{Module, Function as IrFunction, ValueId};
use crate::ir::instructions::Instruction;
use crate::error::Result;
use super::pass_manager::OptimizationPass;
use std::collections::HashSet;

/// Dead code elimination pass
pub struct DeadCodeElimination {
    changed: bool,
}

impl DeadCodeElimination {
    pub fn new() -> Self {
        DeadCodeElimination { changed: false }
    }
    
    /// Mark all live values starting from roots
    fn mark_live(&self, function: &IrFunction) -> HashSet<ValueId> {
        let mut live = HashSet::new();
        let mut worklist = Vec::new();
        
        // Start with all instructions that have side effects
        for block in &function.blocks {
            for (value_id, inst) in &block.instructions {
                if self.has_side_effects(inst) {
                    live.insert(*value_id);
                    worklist.push(*value_id);
                }
            }
        }
        
        // Mark all values used by live instructions
        while let Some(value_id) = worklist.pop() {
            // Find the instruction for this value
            for block in &function.blocks {
                for (vid, inst) in &block.instructions {
                    if *vid == value_id {
                        // Mark all used values as live
                        for used in inst.used_values() {
                            if !live.contains(&used) {
                                live.insert(used);
                                worklist.push(used);
                            }
                        }
                        break;
                    }
                }
            }
        }
        
        live
    }
    
    /// Check if an instruction has side effects
    fn has_side_effects(&self, inst: &Instruction) -> bool {
        matches!(
            inst,
            Instruction::Store { .. } |
            Instruction::Call { .. } |
            Instruction::Return { .. } |
            Instruction::Branch { .. } |
            Instruction::Jump { .. }
        )
    }
}

impl Default for DeadCodeElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "Dead Code Elimination"
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
        
        // Mark all live values
        let live = self.mark_live(function);
        
        // Remove dead instructions
        for block in &mut function.blocks {
            let original_len = block.instructions.len();
            
            block.instructions.retain(|(value_id, inst)| {
                // Keep instructions with side effects or live values
                self.has_side_effects(inst) || live.contains(value_id)
            });
            
            if block.instructions.len() < original_len {
                self.changed = true;
            }
        }
        
        Ok(self.changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ssa::{BasicBlock, ValueId};
    use crate::ir::instructions::BinaryOp;
    use crate::ir::types::IrType;

    #[test]
    fn test_dce_creation() {
        let dce = DeadCodeElimination::new();
        assert!(!dce.changed);
    }

    #[test]
    fn test_has_side_effects() {
        let dce = DeadCodeElimination::new();
        
        let store = Instruction::Store {
            ptr: ValueId(0),
            value: ValueId(1),
        };
        assert!(dce.has_side_effects(&store));
        
        let add = Instruction::Binary {
            op: BinaryOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: IrType::I32,
        };
        assert!(!dce.has_side_effects(&add));
    }

    #[test]
    fn test_mark_live_with_return() {
        let mut func = IrFunction::new("test".to_string(), vec![], IrType::I32);
        let block_id = func.add_block();
        
        // Add a return instruction (has side effects)
        let ret_inst = Instruction::Return {
            value: Some(ValueId(0)),
        };
        let ret_val = func.add_value(crate::ir::ssa::Value::Instruction(ret_inst.clone()));
        
        if let Some(block) = func.get_block_mut(block_id) {
            block.add_instruction(ret_val, ret_inst);
        }
        
        let dce = DeadCodeElimination::new();
        let live = dce.mark_live(&func);
        
        // Return instruction should be marked as live
        assert!(live.contains(&ret_val));
    }
}
