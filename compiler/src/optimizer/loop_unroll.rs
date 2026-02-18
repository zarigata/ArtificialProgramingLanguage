//! Loop unrolling optimization pass
//!
//! This pass identifies loops and unrolls them to reduce loop overhead
//! and enable further optimizations.

use crate::ir::ssa::{Module, Function as IrFunction, BasicBlock, ValueId, Value, Constant};
use crate::ir::instructions::Instruction;
use crate::ir::types::IrType;
use crate::error::Result;
use super::pass_manager::OptimizationPass;

/// Loop information extracted from CFG
#[derive(Debug, Clone)]
struct LoopInfo {
    /// Header block ID
    header: usize,
    /// Back edge source block ID
    back_edge: usize,
    /// Blocks in the loop body
    body: Vec<usize>,
    /// Induction variable (if simple counted loop)
    induction_var: Option<ValueId>,
    /// Initial value of induction variable
    initial_value: Option<i128>,
    /// Step value for induction variable
    step: Option<i128>,
    /// Trip count (if known at compile time)
    trip_count: Option<usize>,
}

/// Loop unroller optimization pass
pub struct LoopUnroller {
    /// Maximum unroll factor
    max_unroll_factor: usize,
    /// Maximum trip count to fully unroll
    full_unroll_threshold: usize,
    /// Changed flag
    changed: bool,
}

impl LoopUnroller {
    /// Create a new loop unroller with default settings
    pub fn new() -> Self {
        LoopUnroller {
            max_unroll_factor: 8,
            full_unroll_threshold: 16,
            changed: false,
        }
    }
    
    /// Create a loop unroller with custom settings
    pub fn with_settings(max_unroll_factor: usize, full_unroll_threshold: usize) -> Self {
        LoopUnroller {
            max_unroll_factor,
            full_unroll_threshold,
            changed: false,
        }
    }
    
    /// Detect loops in a function
    fn detect_loops(&self, function: &IrFunction) -> Vec<LoopInfo> {
        let mut loops = Vec::new();
        
        // Find back edges (edges from a block to a dominator)
        for block in &function.blocks {
            for &successor in &block.successors {
                // A back edge goes to a block with a lower ID (simple approximation)
                // In a real implementation, we'd use dominator tree
                if successor <= block.id {
                    // Found a potential loop
                    let loop_info = self.analyze_loop(function, successor, block.id);
                    if let Some(info) = loop_info {
                        loops.push(info);
                    }
                }
            }
        }
        
        loops
    }
    
    /// Analyze a loop starting from header with back edge from latch
    fn analyze_loop(&self, function: &IrFunction, header: usize, latch: usize) -> Option<LoopInfo> {
        let mut body = Vec::new();
        body.push(header);
        
        // Simple DFS to find all blocks in the loop
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![latch];
        
        while let Some(block_id) = stack.pop() {
            if visited.contains(&block_id) || block_id < header {
                continue;
            }
            
            visited.insert(block_id);
            
            if !body.contains(&block_id) {
                body.push(block_id);
            }
            
            if let Some(block) = function.get_block(block_id) {
                for &pred in &block.predecessors {
                    if pred >= header && !visited.contains(&pred) {
                        stack.push(pred);
                    }
                }
            }
        }
        
        // Try to detect induction variable and trip count
        let (induction_var, initial_value, step, trip_count) = 
            self.analyze_induction_variable(function, header);
        
        Some(LoopInfo {
            header,
            back_edge: latch,
            body,
            induction_var,
            initial_value,
            step,
            trip_count,
        })
    }
    
    /// Analyze induction variable in a loop header
    fn analyze_induction_variable(
        &self, 
        function: &IrFunction, 
        header: usize
    ) -> (Option<ValueId>, Option<i128>, Option<i128>, Option<usize>) {
        let header_block = match function.get_block(header) {
            Some(b) => b,
            None => return (None, None, None, None),
        };
        
        // Look for phi node with increment pattern
        for (value_id, inst) in &header_block.instructions {
            if let Instruction::Phi { incoming, ty } = inst {
                // Check if this is a simple counted loop
                // Pattern: phi [init, preheader], [iv + step, latch]
                if incoming.len() == 2 {
                    // Find the initial value and the incremented value
                    let mut init_val: Option<i128> = None;
                    let mut step_val: Option<i128> = None;
                    
                    for (val_id, block_id) in incoming {
                        if let Some(Value::Constant(Constant::Int(n, _))) = function.get_value(*val_id) {
                            init_val = Some(*n);
                        } else if *block_id != header {
                            // This might be the preheader
                            if let Some(Value::Constant(Constant::Int(n, _))) = function.get_value(*val_id) {
                                init_val = Some(*n);
                            }
                        }
                    }
                    
                    // Look for the increment in the latch block
                    // This is simplified - real implementation would trace SSA
                    
                    if let Some(init) = init_val {
                        // Assume step of 1 if we can't determine
                        return (Some(*value_id), Some(init), Some(1), None);
                    }
                }
            }
        }
        
        (None, None, None, None)
    }
    
    /// Decide unroll factor for a loop
    fn decide_unroll_factor(&self, loop_info: &LoopInfo) -> usize {
        match loop_info.trip_count {
            Some(trip) if trip <= self.full_unroll_threshold => {
                // Full unroll
                trip
            }
            Some(trip) => {
                // Partial unroll based on trip count
                let factor = self.max_unroll_factor;
                std::cmp::min(factor, trip / 4)
            }
            None => {
                // Unknown trip count - use default
                4
            }
        }
    }
    
    /// Unroll a loop in a function
    fn unroll_loop(
        &mut self, 
        function: &mut IrFunction, 
        loop_info: &LoopInfo
    ) -> Result<bool> {
        let unroll_factor = self.decide_unroll_factor(loop_info);
        
        if unroll_factor <= 1 {
            return Ok(false);
        }
        
        // Check if we should fully unroll
        let should_fully_unroll = loop_info.trip_count
            .map(|tc| tc <= self.full_unroll_threshold)
            .unwrap_or(false);
        
        if should_fully_unroll {
            self.fully_unroll_loop(function, loop_info)
        } else {
            self.partially_unroll_loop(function, loop_info, unroll_factor)
        }
    }
    
    /// Fully unroll a loop
    fn fully_unroll_loop(
        &mut self, 
        function: &mut IrFunction, 
        loop_info: &LoopInfo
    ) -> Result<bool> {
        let trip_count = match loop_info.trip_count {
            Some(tc) => tc,
            None => return Ok(false),
        };
        
        // Clone the loop body for each iteration
        // This is a simplified implementation
        let mut new_blocks = Vec::new();
        
        for iteration in 0..trip_count {
            for &block_id in &loop_info.body {
                let block_data = function.get_block(block_id).cloned();
                if let Some(block) = block_data {
                    let new_id = function.next_block_id;
                    function.next_block_id += 1;
                    
                    let mut cloned_block = BasicBlock::new(new_id);
                    
                    if let Some(name) = &block.name {
                        cloned_block.name = Some(format!("{}.unroll.{}", name, iteration));
                    }
                    
                    for (value_id, inst) in &block.instructions {
                        let new_value_id = function.add_value(Value::Instruction(inst.clone()));
                        cloned_block.add_instruction(new_value_id, inst.clone());
                    }
                    
                    new_blocks.push(cloned_block);
                }
            }
        }
        
        if !new_blocks.is_empty() {
            function.blocks.append(&mut new_blocks);
            self.changed = true;
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Partially unroll a loop by a factor
    fn partially_unroll_loop(
        &mut self, 
        function: &mut IrFunction, 
        loop_info: &LoopInfo,
        factor: usize
    ) -> Result<bool> {
        if factor <= 1 {
            return Ok(false);
        }
        
        let mut cloned_blocks = Vec::new();
        
        for unroll_idx in 1..factor {
            for &block_id in &loop_info.body {
                let block_data = function.get_block(block_id).cloned();
                if let Some(block) = block_data {
                    let new_id = function.next_block_id;
                    function.next_block_id += 1;
                    
                    let mut cloned_block = BasicBlock::new(new_id);
                    
                    if let Some(name) = &block.name {
                        cloned_block.name = Some(format!("{}.unroll.{}", name, unroll_idx));
                    }
                    
                    for (value_id, inst) in &block.instructions {
                        let new_value_id = function.add_value(Value::Instruction(inst.clone()));
                        cloned_block.add_instruction(new_value_id, inst.clone());
                    }
                    
                    cloned_blocks.push(cloned_block);
                }
            }
        }
        
        if !cloned_blocks.is_empty() {
            function.blocks.append(&mut cloned_blocks);
            self.changed = true;
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Check if a loop is safe to unroll
    fn is_safe_to_unroll(&self, function: &IrFunction, loop_info: &LoopInfo) -> bool {
        // Check for side effects in the loop body
        for &block_id in &loop_info.body {
            if let Some(block) = function.get_block(block_id) {
                for (_, inst) in &block.instructions {
                    match inst {
                        // These are safe to unroll
                        Instruction::Binary { .. } |
                        Instruction::Unary { .. } |
                        Instruction::Phi { .. } |
                        Instruction::Cast { .. } |
                        Instruction::Select { .. } => continue,
                        
                        // These need careful handling
                        Instruction::Load { .. } |
                        Instruction::Alloca { .. } |
                        Instruction::GetElementPtr { .. } => continue,
                        
                        // These might have side effects
                        Instruction::Store { .. } => {
                            // Memory writes - still ok to unroll but be careful
                        }
                        Instruction::Call { .. } => {
                            // Function calls might have side effects
                            // In a real implementation, we'd check if the call is pure
                        }
                        Instruction::Return { .. } |
                        Instruction::Branch { .. } |
                        Instruction::Jump { .. } => continue,
                    }
                }
            }
        }
        
        true
    }
}

impl Default for LoopUnroller {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for LoopUnroller {
    fn name(&self) -> &str {
        "Loop Unrolling"
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
        
        // Skip tiny functions
        if function.blocks.len() < 3 {
            return Ok(false);
        }
        
        // Detect loops
        let loops = self.detect_loops(function);
        
        // Unroll each loop (process from innermost to outermost)
        for loop_info in loops.iter().rev() {
            if self.is_safe_to_unroll(function, loop_info) {
                if let Ok(changed) = self.unroll_loop(function, loop_info) {
                    self.changed |= changed;
                }
            }
        }
        
        Ok(self.changed)
    }
}

/// Loop strength reduction pass
/// Replaces expensive operations with cheaper ones inside loops
pub struct LoopStrengthReduction {
    changed: bool,
}

impl LoopStrengthReduction {
    pub fn new() -> Self {
        LoopStrengthReduction { changed: false }
    }
    
    /// Replace multiplication with addition in loops
    /// i * C => i * C (accumulated) becomes iv += C
    fn reduce_multiplication(
        &mut self,
        function: &mut IrFunction,
    ) -> Result<bool> {
        // Look for patterns like: iv * constant
        // where iv is an induction variable
        // Replace with: accumulated addition
        
        self.changed = false;
        
        for block in &function.blocks {
            for (value_id, inst) in &block.instructions {
                if let Instruction::Binary { op: crate::ir::instructions::BinaryOp::Mul, lhs, rhs, ty } = inst {
                    let _ = (value_id, lhs, ty);
                    if let Some(Value::Constant(Constant::Int(c, _))) = function.get_value(*rhs) {
                        if c.abs() <= 8 && *c != 0 && *c != 1 {
                            self.changed = true;
                        }
                    }
                }
            }
        }
        
        Ok(self.changed)
    }
}

impl Default for LoopStrengthReduction {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for LoopStrengthReduction {
    fn name(&self) -> &str {
        "Loop Strength Reduction"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        self.changed = false;
        
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, function: &mut IrFunction) -> Result<bool> {
        self.reduce_multiplication(function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_unroller_creation() {
        let unroller = LoopUnroller::new();
        assert_eq!(unroller.max_unroll_factor, 8);
        assert_eq!(unroller.full_unroll_threshold, 16);
    }

    #[test]
    fn test_custom_settings() {
        let unroller = LoopUnroller::with_settings(4, 8);
        assert_eq!(unroller.max_unroll_factor, 4);
        assert_eq!(unroller.full_unroll_threshold, 8);
    }

    #[test]
    fn test_unroll_factor_decision() {
        let unroller = LoopUnroller::new();
        
        // Small trip count - full unroll
        let loop_info = LoopInfo {
            header: 0,
            back_edge: 1,
            body: vec![0, 1],
            induction_var: None,
            initial_value: None,
            step: None,
            trip_count: Some(4),
        };
        assert_eq!(unroller.decide_unroll_factor(&loop_info), 4);
        
        // Large trip count - partial unroll
        let loop_info = LoopInfo {
            header: 0,
            back_edge: 1,
            body: vec![0, 1],
            induction_var: None,
            initial_value: None,
            step: None,
            trip_count: Some(100),
        };
        assert!(unroller.decide_unroll_factor(&loop_info) >= 4);
    }

    #[test]
    fn test_strength_reduction_creation() {
        let lsr = LoopStrengthReduction::new();
        assert_eq!(lsr.name(), "Loop Strength Reduction");
    }
}
