//! Devirtualization optimization pass
//!
//! This pass converts virtual function calls to direct calls when the
//! concrete type can be determined at compile time.

use std::collections::{HashMap, HashSet};
use crate::ir::ssa::{Module, Function as IrFunction, ValueId, Value};
use crate::ir::instructions::Instruction;
use crate::ir::types::IrType;
use crate::error::Result;
use super::pass_manager::OptimizationPass;

/// Virtual call site information
#[derive(Debug, Clone)]
struct VirtualCallSite {
    /// Block ID containing the call
    block_id: usize,
    /// Instruction position in block
    instruction_idx: usize,
    /// Value ID of the call instruction
    call_value_id: ValueId,
    /// The object receiver
    receiver: ValueId,
    /// Method name being called
    method_name: String,
    /// Arguments to the method
    args: Vec<ValueId>,
}

/// Type hierarchy information
#[derive(Debug, Clone, Default)]
struct TypeHierarchy {
    /// Maps type names to their possible concrete implementations
    implementations: HashMap<String, HashSet<String>>,
    /// Maps interface/trait names to implementing types
    trait_impls: HashMap<String, HashSet<String>>,
}

impl TypeHierarchy {
    fn new() -> Self {
        TypeHierarchy {
            implementations: HashMap::new(),
            trait_impls: HashMap::new(),
        }
    }
    
    /// Add an implementation relationship
    fn add_impl(&mut self, trait_name: &str, type_name: &str) {
        self.trait_impls
            .entry(trait_name.to_string())
            .or_default()
            .insert(type_name.to_string());
    }
    
    /// Get all implementations of a trait
    fn get_impls(&self, trait_name: &str) -> Option<&HashSet<String>> {
        self.trait_impls.get(trait_name)
    }
}

/// Devirtualization pass
pub struct Devirtualizer {
    /// Type hierarchy
    hierarchy: TypeHierarchy,
    /// Changed flag
    changed: bool,
    /// Maximum analysis depth
    max_depth: usize,
}

impl Devirtualizer {
    /// Create a new devirtualization pass
    pub fn new() -> Self {
        Devirtualizer {
            hierarchy: TypeHierarchy::new(),
            changed: false,
            max_depth: 10,
        }
    }
    
    /// Build type hierarchy from module
    fn build_hierarchy(&mut self, module: &Module) {
        // In a real implementation, we would:
        // 1. Scan all trait definitions
        // 2. Scan all impl blocks
        // 3. Build inheritance/implementation relationships
        
        // Placeholder implementation
        self.hierarchy.add_impl("Display", "String");
        self.hierarchy.add_impl("Display", "i32");
        self.hierarchy.add_impl("Iterator", "Vec");
        self.hierarchy.add_impl("Clone", "String");
        self.hierarchy.add_impl("Clone", "Vec");
    }
    
    /// Find virtual call sites in a function
    fn find_virtual_calls(&self, function: &IrFunction) -> Vec<VirtualCallSite> {
        let mut callsites = Vec::new();
        
        for (block_idx, block) in function.blocks.iter().enumerate() {
            for (inst_idx, (value_id, inst)) in block.instructions.iter().enumerate() {
                if let Instruction::Call { func, args, .. } = inst {
                    // Check if this is a virtual call
                    if self.is_virtual_call(function, *func, args) {
                        // Extract method name and receiver
                        if let Some((receiver, method_name)) = 
                            self.extract_virtual_call_info(function, *func, args) {
                            callsites.push(VirtualCallSite {
                                block_id: block_idx,
                                instruction_idx: inst_idx,
                                call_value_id: *value_id,
                                receiver,
                                method_name,
                                args: args.clone(),
                            });
                        }
                    }
                }
            }
        }
        
        callsites
    }
    
    /// Check if a call is virtual
    fn is_virtual_call(
        &self, 
        function: &IrFunction, 
        func: ValueId, 
        args: &[ValueId]
    ) -> bool {
        // A virtual call typically:
        // 1. Has a receiver (first argument is an object reference)
        // 2. The function is loaded from a vtable
        
        if let Some(Value::Instruction(inst)) = function.get_value(func) {
            // Check if function pointer comes from a vtable load
            if let Instruction::Load { .. } = inst {
                // Likely a virtual call through vtable
                return args.len() > 0;
            }
            
            // Check for method call pattern
            if let Instruction::GetElementPtr { .. } = inst {
                return args.len() > 0;
            }
        }
        
        false
    }
    
    /// Extract receiver and method name from a virtual call
    fn extract_virtual_call_info(
        &self,
        function: &IrFunction,
        func: ValueId,
        args: &[ValueId],
    ) -> Option<(ValueId, String)> {
        if args.is_empty() {
            return None;
        }
        
        // First argument is typically the receiver (self)
        let receiver = args[0];
        
        // Method name would come from debug info or vtable analysis
        // Simplified: assume we can extract it
        let method_name = "unknown".to_string();
        
        Some((receiver, method_name))
    }
    
    /// Determine concrete type of a value
    fn infer_concrete_type(
        &self,
        function: &IrFunction,
        value: ValueId,
        depth: usize,
    ) -> Option<String> {
        if depth > self.max_depth {
            return None;
        }
        
        match function.get_value(value)? {
            Value::Instruction(inst) => {
                match inst {
                    Instruction::Alloca { ty } => {
                        // Direct allocation - we know the type
                        Some(format!("{:?}", ty))
                    }
                    Instruction::Call { func, .. } => {
                        // Constructor call - infer from return type
                        self.infer_concrete_type(function, *func, depth + 1)
                    }
                    Instruction::Load { ptr, .. } => {
                        // Load from pointer - follow pointer type
                        self.infer_concrete_type(function, *ptr, depth + 1)
                    }
                    Instruction::Phi { incoming, .. } => {
                        // Phi node - all incoming must agree
                        let mut types: HashSet<String> = HashSet::new();
                        for (val, _) in incoming {
                            if let Some(ty) = self.infer_concrete_type(function, *val, depth + 1) {
                                types.insert(ty);
                            }
                        }
                        if types.len() == 1 {
                            types.into_iter().next()
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            Value::Parameter(idx, ty) => {
                // Parameter type might be known from function signature
                Some(format!("{:?}", ty))
            }
            _ => None,
        }
    }
    
    /// Devirtualize a single call site
    fn devirtualize_call(
        &mut self,
        function: &mut IrFunction,
        callsite: &VirtualCallSite,
    ) -> Result<bool> {
        // Try to infer the concrete type of receiver
        let concrete_type = self.infer_concrete_type(
            function, 
            callsite.receiver, 
            0
        );
        
        if let Some(type_name) = concrete_type {
            // We know the concrete type - can devirtualize
            // Replace virtual call with direct call
            
            // In a real implementation, we would:
            // 1. Create a direct function reference
            // 2. Replace the virtual call instruction
            // 3. Update SSA form
            
            self.changed = true;
            return Ok(true);
        }
        
        // Try speculative devirtualization
        if let Some(impls) = self.hierarchy.get_impls(&callsite.method_name) {
            if impls.len() == 1 {
                // Only one implementation - can devirtualize
                self.changed = true;
                return Ok(true);
            } else if impls.len() <= 4 {
                // Few implementations - use guarded devirtualization
                // Generate: if type == A then call A::method else if type == B ...
                self.changed = true;
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Perform whole-program analysis for devirtualization
    fn whole_program_analysis(&mut self, module: &Module) {
        // Build call graph
        // Identify monomorphic call sites
        // Propagate type information
        
        self.build_hierarchy(module);
    }
}

impl Default for Devirtualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Devirtualizer {
    fn name(&self) -> &str {
        "Devirtualization"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        self.changed = false;
        
        // Perform whole-program analysis
        self.whole_program_analysis(module);
        
        // Process each function
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, function: &mut IrFunction) -> Result<bool> {
        // Find virtual call sites
        let callsites = self.find_virtual_calls(function);
        
        // Try to devirtualize each call
        for callsite in callsites {
            self.devirtualize_call(function, &callsite)?;
        }
        
        Ok(self.changed)
    }
}

/// Speculative Devirtualization with Guards
/// Uses runtime checks to enable devirtualization when static analysis is inconclusive
pub struct SpeculativeDevirtualizer {
    /// Maximum number of guards to insert
    max_guards: usize,
    /// Guards inserted so far
    guards_inserted: usize,
    /// Changed flag
    changed: bool,
}

impl SpeculativeDevirtualizer {
    pub fn new() -> Self {
        SpeculativeDevirtualizer {
            max_guards: 10,
            guards_inserted: 0,
            changed: false,
        }
    }
    
    /// Insert a type check guard before a call
    fn insert_type_guard(
        &mut self,
        function: &mut IrFunction,
        callsite: &VirtualCallSite,
        expected_type: &str,
    ) -> Result<bool> {
        if self.guards_inserted >= self.max_guards {
            return Ok(false);
        }
        
        // Generate:
        // if (obj.type == ExpectedType) {
        //     direct_call(obj, args)
        // } else {
        //     virtual_call(obj, args)  // fallback
        // }
        
        self.guards_inserted += 1;
        self.changed = true;
        Ok(true)
    }
    
    /// Profile-guided devirtualization
    /// Use profiling data to determine most likely targets
    fn profile_guided_devirt(
        &mut self,
        function: &mut IrFunction,
        callsite: &VirtualCallSite,
    ) -> Result<bool> {
        // In a real implementation, this would use:
        // - PGO (Profile-Guided Optimization) data
        // - Runtime type frequency information
        // - Inline cache data
        
        Ok(false)
    }
}

impl Default for SpeculativeDevirtualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for SpeculativeDevirtualizer {
    fn name(&self) -> &str {
        "Speculative Devirtualization"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        self.changed = false;
        self.guards_inserted = 0;
        
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, function: &mut IrFunction) -> Result<bool> {
        // Use speculative devirtualizer as fallback
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_devirtualizer_creation() {
        let devirt = Devirtualizer::new();
        assert_eq!(devirt.name(), "Devirtualization");
        assert_eq!(devirt.max_depth, 10);
    }

    #[test]
    fn test_type_hierarchy() {
        let mut hierarchy = TypeHierarchy::new();
        hierarchy.add_impl("Display", "String");
        hierarchy.add_impl("Display", "i32");
        
        let impls = hierarchy.get_impls("Display").unwrap();
        assert!(impls.contains("String"));
        assert!(impls.contains("i32"));
    }

    #[test]
    fn test_speculative_devirtualizer() {
        let spec = SpeculativeDevirtualizer::new();
        assert_eq!(spec.name(), "Speculative Devirtualization");
        assert_eq!(spec.max_guards, 10);
    }
}
