//! Escape Analysis optimization pass
//!
//! Determines if objects allocated in a function can escape the function's scope.
//! Objects that don't escape can be stack-allocated instead of heap-allocated,
//! or even eliminated entirely through scalar replacement.

use std::collections::{HashMap, HashSet};
use crate::ir::ssa::{Module, Function as IrFunction, BasicBlock, ValueId, Value};
use crate::ir::instructions::Instruction;
use crate::ir::types::IrType;
use crate::error::Result;
use super::pass_manager::OptimizationPass;

/// Represents an allocation site
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AllocationSite {
    /// Value ID of the allocation
    value_id: ValueId,
    /// Block where allocation occurs
    block_id: usize,
}

/// Escape information for an allocation
#[derive(Debug, Clone)]
struct EscapeInfo {
    /// Whether the allocation escapes the function
    escapes: bool,
    /// How it escapes (if it does)
    escape_path: Option<EscapePath>,
    /// Fields that escape individually (for partial escape)
    escaping_fields: HashSet<usize>,
}

/// How an allocation escapes
#[derive(Debug, Clone)]
enum EscapePath {
    /// Returned from the function
    Returned,
    /// Stored to a global or escaping pointer
    StoredGlobal,
    /// Passed to another function that may capture it
    PassedToFunction(String),
    /// Stored to a field of an escaping object
    StoredToEscapingObject(ValueId),
}

/// Points-to graph node
#[derive(Debug, Clone)]
struct PointsToNode {
    /// Node ID
    id: usize,
    /// Points-to set
    points_to: HashSet<usize>,
    /// Whether this node represents an escape point
    is_escape_point: bool,
}

/// Escape analysis result
#[derive(Debug, Clone, Default)]
struct AnalysisResult {
    /// Allocations that don't escape
    non_escaping: HashSet<ValueId>,
    /// Allocations that escape
    escaping: HashSet<ValueId>,
    /// Escape paths for escaping allocations
    escape_paths: HashMap<ValueId, EscapePath>,
}

/// Escape analysis pass
pub struct EscapeAnalyzer {
    /// Analysis result
    result: AnalysisResult,
    /// Changed flag
    changed: bool,
    /// Points-to graph
    points_to_graph: HashMap<usize, PointsToNode>,
    /// Next node ID
    next_node_id: usize,
}

impl EscapeAnalyzer {
    /// Create a new escape analyzer
    pub fn new() -> Self {
        EscapeAnalyzer {
            result: AnalysisResult::default(),
            changed: false,
            points_to_graph: HashMap::new(),
            next_node_id: 0,
        }
    }
    
    /// Analyze a function for escaping allocations
    fn analyze_function(&mut self, function: &IrFunction) {
        self.result = AnalysisResult::default();
        self.points_to_graph.clear();
        self.next_node_id = 0;
        
        // Step 1: Find all allocations
        let allocations = self.find_allocations(function);
        
        // Step 2: Build points-to graph
        self.build_points_to_graph(function);
        
        // Step 3: Compute escape information
        for alloc in &allocations {
            let escapes = self.check_escape(function, alloc.value_id);
            
            if escapes {
                self.result.escaping.insert(alloc.value_id);
                self.result.escape_paths.insert(
                    alloc.value_id,
                    self.find_escape_path(function, alloc.value_id)
                        .unwrap_or(EscapePath::Returned)
                );
            } else {
                self.result.non_escaping.insert(alloc.value_id);
            }
        }
    }
    
    /// Find all allocation sites in a function
    fn find_allocations(&self, function: &IrFunction) -> Vec<AllocationSite> {
        let mut allocations = Vec::new();
        
        for (block_id, block) in function.blocks.iter().enumerate() {
            for (value_id, inst) in &block.instructions {
                if let Instruction::Alloca { .. } = inst {
                    allocations.push(AllocationSite {
                        value_id: *value_id,
                        block_id,
                    });
                }
                
                // Also track heap allocations (in real impl)
                // new, malloc, Box::new, etc.
            }
        }
        
        allocations
    }
    
    /// Build points-to graph for the function
    fn build_points_to_graph(&mut self, function: &IrFunction) {
        // Create a node for each allocation
        for block in &function.blocks {
            for (value_id, inst) in &block.instructions {
                if let Instruction::Alloca { .. } = inst {
                    let node = PointsToNode {
                        id: self.next_node_id,
                        points_to: HashSet::new(),
                        is_escape_point: false,
                    };
                    self.points_to_graph.insert(value_id.0, node);
                    self.next_node_id += 1;
                }
            }
        }
        
        // Add edges based on stores and loads
        for block in &function.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::Store { ptr, value } => {
                        // ptr -> value edge
                        if let Some(ptr_node) = self.points_to_graph.get_mut(&ptr.0) {
                            ptr_node.points_to.insert(value.0);
                        }
                    }
                    Instruction::Load { ptr, .. } => {
                        // Load creates an alias
                    }
                    _ => {}
                }
            }
        }
    }
    
    /// Check if an allocation escapes
    fn check_escape(&self, function: &IrFunction, alloc: ValueId) -> bool {
        // Track all uses of the allocation
        let mut visited = HashSet::new();
        let mut worklist = vec![alloc];
        
        while let Some(value) = worklist.pop() {
            if visited.contains(&value) {
                continue;
            }
            visited.insert(value);
            
            // Check all uses of this value
            for block in &function.blocks {
                for (use_value_id, inst) in &block.instructions {
                    match inst {
                        Instruction::Return { value: Some(ret_val) } => {
                            // Returned values escape
                            if *ret_val == value || self.may_alias(function, *ret_val, value) {
                                return true;
                            }
                        }
                        Instruction::Store { ptr, value: stored } => {
                            // Storing to a global or escaping pointer
                            if *stored == value {
                                // Check if ptr escapes
                                if self.is_escaping_pointer(function, *ptr) {
                                    return true;
                                }
                                // Follow the pointer
                                worklist.push(*ptr);
                            }
                        }
                        Instruction::Call { args, .. } => {
                            // Passing to a function may cause escape
                            for arg in args {
                                if *arg == value || self.may_alias(function, *arg, value) {
                                    // Conservative: assume it escapes
                                    // In real impl, analyze callee
                                    return true;
                                }
                            }
                        }
                        Instruction::Load { ptr, .. } => {
                            // Loading doesn't cause escape by itself
                            if *ptr == value {
                                // The loaded value might escape
                                // Add to worklist for transitive analysis
                            }
                        }
                        Instruction::GetElementPtr { ptr, .. } => {
                            // GEP creates a derived pointer
                            if *ptr == value {
                                // Derived pointer, track it
                                worklist.push(*use_value_id);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        
        false
    }
    
    /// Find the escape path for an allocation
    fn find_escape_path(&self, function: &IrFunction, alloc: ValueId) -> Option<EscapePath> {
        for block in &function.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::Return { value: Some(ret_val) } => {
                        if *ret_val == alloc {
                            return Some(EscapePath::Returned);
                        }
                    }
                    Instruction::Store { ptr, value } => {
                        if *value == alloc {
                            // Check what ptr is
                            if self.is_global_pointer(function, *ptr) {
                                return Some(EscapePath::StoredGlobal);
                            }
                        }
                    }
                    Instruction::Call { func, args, .. } => {
                        for arg in args {
                            if *arg == alloc {
                                // Get function name if possible
                                if let Some(Value::Global(name, _)) = function.get_value(*func) {
                                    return Some(EscapePath::PassedToFunction(name.clone()));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        
        None
    }
    
    /// Check if two values may alias
    fn may_alias(&self, function: &IrFunction, a: ValueId, b: ValueId) -> bool {
        // Simple check - in real impl use alias analysis
        a == b
    }
    
    /// Check if a pointer is escaping (global, parameter, etc.)
    fn is_escaping_pointer(&self, function: &IrFunction, ptr: ValueId) -> bool {
        match function.get_value(ptr) {
            Some(Value::Global(_, _)) => true,
            Some(Value::Parameter(_, _)) => true,
            Some(Value::Instruction(Instruction::Load { ptr: inner, .. })) => {
                self.is_escaping_pointer(function, *inner)
            }
            _ => false,
        }
    }
    
    /// Check if a pointer points to global memory
    fn is_global_pointer(&self, function: &IrFunction, ptr: ValueId) -> bool {
        matches!(function.get_value(ptr), Some(Value::Global(_, _)))
    }
    
    /// Perform scalar replacement on non-escaping allocations
    fn scalar_replace(&mut self, function: &mut IrFunction) -> Result<bool> {
        let mut replacements = Vec::new();
        
        for alloc_id in &self.result.non_escaping {
            // For each non-escaping allocation, check if we can scalarize
            if let Some(Value::Instruction(Instruction::Alloca { ty })) = function.get_value(*alloc_id) {
                if self.can_scalarize(ty) {
                    // Create scalar variables for each field
                    replacements.push((*alloc_id, ty.clone()));
                }
            }
        }
        
        // Apply replacements
        if !replacements.is_empty() {
            self.changed = true;
            // In real impl, replace loads/stores with direct variable access
        }
        
        Ok(!replacements.is_empty())
    }
    
    /// Check if a type can be scalarized
    fn can_scalarize(&self, ty: &IrType) -> bool {
        match ty {
            IrType::Struct(fields) => fields.len() <= 16, // Don't scalarize huge structs
            IrType::Array(elem, size) => *size <= 8 && self.can_scalarize(elem),
            IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 |
            IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64 |
            IrType::F32 | IrType::F64 | IrType::Bool => true,
            _ => false,
        }
    }
    
    /// Convert heap allocations to stack allocations where possible
    fn heap_to_stack(&mut self, function: &mut IrFunction) -> Result<bool> {
        // Find heap allocations that don't escape
        // Replace with stack allocations
        
        Ok(false)
    }
    
    /// Eliminate dead allocations
    fn eliminate_dead_allocations(&mut self, function: &mut IrFunction) -> Result<bool> {
        let mut dead = Vec::new();
        
        for alloc_id in &self.result.non_escaping {
            // Check if the allocation is actually used
            if !self.is_used(function, *alloc_id) {
                dead.push(*alloc_id);
            }
        }
        
        // Remove dead allocations
        if !dead.is_empty() {
            self.changed = true;
            // In real impl, remove the alloca and its stores
        }
        
        Ok(!dead.is_empty())
    }
    
    /// Check if an allocation is used
    fn is_used(&self, function: &IrFunction, alloc: ValueId) -> bool {
        for block in &function.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::Load { ptr, .. } if *ptr == alloc => return true,
                    Instruction::Store { ptr, .. } if *ptr == alloc => return true,
                    Instruction::GetElementPtr { ptr, .. } if *ptr == alloc => return true,
                    _ => {}
                }
            }
        }
        false
    }
}

impl Default for EscapeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for EscapeAnalyzer {
    fn name(&self) -> &str {
        "Escape Analysis"
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
        
        // Perform analysis
        self.analyze_function(function);
        
        // Apply optimizations
        self.scalar_replace(function)?;
        self.heap_to_stack(function)?;
        self.eliminate_dead_allocations(function)?;
        
        Ok(self.changed)
    }
}

/// Stack Allocation promotion
/// Promotes heap allocations to stack when they don't escape
pub struct StackPromotion {
    changed: bool,
}

impl StackPromotion {
    pub fn new() -> Self {
        StackPromotion { changed: false }
    }
}

impl Default for StackPromotion {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for StackPromotion {
    fn name(&self) -> &str {
        "Stack Promotion"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        self.changed = false;
        
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, function: &mut IrFunction) -> Result<bool> {
        // Find heap allocations (new, malloc, etc.)
        // Check if they escape using escape analysis
        // Replace with alloca if they don't escape
        
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_analyzer_creation() {
        let analyzer = EscapeAnalyzer::new();
        assert_eq!(analyzer.name(), "Escape Analysis");
        assert!(analyzer.result.non_escaping.is_empty());
        assert!(analyzer.result.escaping.is_empty());
    }

    #[test]
    fn test_can_scalarize() {
        let analyzer = EscapeAnalyzer::new();
        
        assert!(analyzer.can_scalarize(&IrType::I32));
        assert!(analyzer.can_scalarize(&IrType::F64));
        assert!(analyzer.can_scalarize(&IrType::Bool));
    }

    #[test]
    fn test_stack_promotion() {
        let promo = StackPromotion::new();
        assert_eq!(promo.name(), "Stack Promotion");
    }
}
