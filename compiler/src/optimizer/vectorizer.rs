//! SIMD Vectorization optimization pass
//!
//! This pass identifies patterns that can be vectorized to use SIMD instructions.

use crate::ir::ssa::{Module, Function as IrFunction, BasicBlock, ValueId, Value, Constant};
use crate::ir::instructions::{Instruction, BinaryOp};
use crate::ir::types::IrType;
use crate::error::Result;
use super::pass_manager::OptimizationPass;

/// Vector width for SIMD operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorWidth {
    /// 128-bit SIMD (SSE, NEON)
    Width128,
    /// 256-bit SIMD (AVX)
    Width256,
    /// 512-bit SIMD (AVX-512)
    Width512,
}

impl VectorWidth {
    /// Get the number of elements for a given element type
    pub fn element_count(&self, element_ty: &IrType) -> usize {
        let element_bits = element_ty.size_in_bits();
        let total_bits = match self {
            VectorWidth::Width128 => 128,
            VectorWidth::Width256 => 256,
            VectorWidth::Width512 => 512,
        };
        total_bits / element_bits
    }
}

/// SIMD target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTarget {
    /// Intel SSE/AVX/AVX-512
    X86,
    /// ARM NEON/SVE
    Arm,
    /// WebAssembly SIMD
    Wasm,
    /// Auto-detect from target
    Auto,
}

/// Vectorizable loop pattern
#[derive(Debug, Clone)]
struct VectorizablePattern {
    /// Loop header block
    loop_header: usize,
    /// Array base pointer
    array_base: ValueId,
    /// Element type
    element_ty: IrType,
    /// Vector width
    vector_width: VectorWidth,
    /// Operations that can be vectorized
    operations: Vec<VectorizableOp>,
}

/// A vectorizable operation
#[derive(Debug, Clone)]
struct VectorizableOp {
    /// The original instruction value ID
    value_id: ValueId,
    /// Operation type
    op: VectorOp,
    /// Source operands
    sources: Vec<ValueId>,
    /// Destination (if store)
    dest: Option<ValueId>,
}

/// Vector operation kind
#[derive(Debug, Clone)]
enum VectorOp {
    /// Load vector from memory
    Load,
    /// Store vector to memory
    Store,
    /// Binary operation on vectors
    Binary(BinaryOp),
    /// Unary operation on vectors
    Unary(crate::ir::instructions::UnaryOp),
    /// Reduction (sum, min, max, etc.)
    Reduce(ReduceOp),
    /// Gather (indexed load)
    Gather,
    /// Scatter (indexed store)
    Scatter,
    /// Blend/select
    Blend,
}

/// Reduction operation kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceOp {
    Sum,
    Min,
    Max,
    Product,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

/// Vectorizer optimization pass
pub struct Vectorizer {
    /// Target SIMD width
    target_width: VectorWidth,
    /// Target architecture
    target: SimdTarget,
    /// Maximum vectorization factor
    max_factor: usize,
    /// Changed flag
    changed: bool,
}

impl Vectorizer {
    /// Create a new vectorizer with default settings
    pub fn new() -> Self {
        Vectorizer {
            target_width: VectorWidth::Width256,
            target: SimdTarget::Auto,
            max_factor: 8,
            changed: false,
        }
    }
    
    /// Create a vectorizer for a specific target
    pub fn for_target(target: SimdTarget) -> Self {
        let width = match target {
            SimdTarget::X86 => VectorWidth::Width256,
            SimdTarget::Arm => VectorWidth::Width128,
            SimdTarget::Wasm => VectorWidth::Width128,
            SimdTarget::Auto => VectorWidth::Width256,
        };
        
        Vectorizer {
            target_width: width,
            target,
            max_factor: 8,
            changed: false,
        }
    }
    
    /// Analyze a function for vectorizable patterns
    fn analyze_function(&self, function: &IrFunction) -> Vec<VectorizablePattern> {
        let mut patterns = Vec::new();
        
        // Look for simple loop patterns:
        // for i in 0..n:
        //     a[i] = b[i] op c[i]
        
        for block in &function.blocks {
            // Check for consecutive load/store patterns
            let pattern = self.analyze_block(function, block);
            patterns.extend(pattern);
        }
        
        patterns
    }
    
    /// Analyze a single block for vectorizable patterns
    fn analyze_block(&self, function: &IrFunction, block: &BasicBlock) -> Vec<VectorizablePattern> {
        let mut patterns = Vec::new();
        let mut loads = Vec::new();
        let mut stores = Vec::new();
        let mut ops = Vec::new();
        
        for (value_id, inst) in &block.instructions {
            match inst {
                Instruction::Load { ptr, ty } => {
                    // Check if this is loading from an array with index
                    loads.push((*value_id, *ptr, ty.clone()));
                }
                Instruction::Store { ptr, value } => {
                    stores.push((*ptr, *value));
                }
                Instruction::Binary { op, lhs, rhs, ty } => {
                    // Check if operating on loaded values
                    let is_vectorizable = self.is_type_vectorizable(ty);
                    if is_vectorizable {
                        ops.push((*value_id, *op, *lhs, *rhs, ty.clone()));
                    }
                }
                _ => {}
            }
        }
        
        // Check if we have a load-op-store pattern
        if !loads.is_empty() && !stores.is_empty() && !ops.is_empty() {
            // Found potential vectorizable pattern
            for (_, ptr, ty) in &loads {
                let element_count = self.target_width.element_count(ty);
                
                if element_count >= 2 {
                    patterns.push(VectorizablePattern {
                        loop_header: block.id,
                        array_base: *ptr,
                        element_ty: ty.clone(),
                        vector_width: self.target_width,
                        operations: ops.iter().map(|(vid, op, lhs, rhs, ty)| {
                            VectorizableOp {
                                value_id: *vid,
                                op: VectorOp::Binary(*op),
                                sources: vec![*lhs, *rhs],
                                dest: None,
                            }
                        }).collect(),
                    });
                }
            }
        }
        
        patterns
    }
    
    /// Check if a type can be vectorized
    fn is_type_vectorizable(&self, ty: &IrType) -> bool {
        matches!(ty, 
            IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 |
            IrType::F32 | IrType::F64 |
            IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64
        )
    }
    
    /// Vectorize a pattern
    fn vectorize_pattern(
        &mut self,
        function: &mut IrFunction,
        pattern: &VectorizablePattern,
    ) -> Result<bool> {
        let element_count = pattern.vector_width.element_count(&pattern.element_ty);
        
        // Create vector type
        let vector_ty = IrType::Vector(Box::new(pattern.element_ty.clone()), element_count);
        
        // Generate vectorized instructions
        // This is a simplified version - real implementation would:
        // 1. Check for dependencies
        // 2. Generate peel/scalar loops for remainder
        // 3. Insert mask operations for conditional execution
        
        // Mark as changed for now
        self.changed = true;
        
        Ok(true)
    }
    
    /// Check if a loop has dependencies that prevent vectorization
    fn check_dependencies(&self, function: &IrFunction, pattern: &VectorizablePattern) -> bool {
        // Simple check: no backward dependencies
        // Real implementation would use dependence analysis
        
        // For now, assume independent
        true
    }
}

impl Default for Vectorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Vectorizer {
    fn name(&self) -> &str {
        "SIMD Vectorization"
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
        
        // Analyze for vectorizable patterns
        let patterns = self.analyze_function(function);
        
        // Vectorize each pattern
        for pattern in patterns {
            if self.check_dependencies(function, &pattern) {
                if let Ok(changed) = self.vectorize_pattern(function, &pattern) {
                    self.changed |= changed;
                }
            }
        }
        
        Ok(self.changed)
    }
}

/// SLP (Superword Level Parallelism) Vectorizer
/// Vectorizes straight-line code by finding isomorphic operations
pub struct SlpVectorizer {
    max_pack_size: usize,
    changed: bool,
}

impl SlpVectorizer {
    pub fn new() -> Self {
        SlpVectorizer {
            max_pack_size: 4,
            changed: false,
        }
    }
    
    /// Find isomorphic instruction trees
    fn find_packable_trees(&self, function: &IrFunction, block: &BasicBlock) -> Vec<Vec<ValueId>> {
        let mut packs = Vec::new();
        
        // Group instructions by operation type
        let mut by_opcode: std::collections::HashMap<String, Vec<ValueId>> = 
            std::collections::HashMap::new();
        
        for (value_id, inst) in &block.instructions {
            let opcode = self.get_opcode_key(inst);
            by_opcode.entry(opcode).or_default().push(*value_id);
        }
        
        // Find groups of same operations
        for (_, values) in by_opcode {
            if values.len() >= 2 {
                // Pack consecutive operations
                for chunk in values.chunks(self.max_pack_size) {
                    if chunk.len() >= 2 {
                        packs.push(chunk.to_vec());
                    }
                }
            }
        }
        
        packs
    }
    
    /// Get a key for grouping isomorphic instructions
    fn get_opcode_key(&self, inst: &Instruction) -> String {
        match inst {
            Instruction::Binary { op, .. } => format!("Binary_{:?}", op),
            Instruction::Unary { op, .. } => format!("Unary_{:?}", op),
            Instruction::Load { .. } => "Load".to_string(),
            Instruction::Store { .. } => "Store".to_string(),
            _ => format!("{:?}", std::mem::discriminant(inst)),
        }
    }
    
    /// Pack operations into vector instructions
    fn pack_operations(
        &mut self,
        function: &mut IrFunction,
        pack: &[ValueId],
    ) -> Result<bool> {
        if pack.len() < 2 {
            return Ok(false);
        }
        
        // Create vector operation
        self.changed = true;
        Ok(true)
    }
}

impl Default for SlpVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for SlpVectorizer {
    fn name(&self) -> &str {
        "SLP Vectorization"
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
        
        for block in function.blocks.clone() {
            let packs = self.find_packable_trees(function, &block);
            
            for pack in packs {
                self.pack_operations(function, &pack)?;
            }
        }
        
        Ok(self.changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_width_element_count() {
        let width128 = VectorWidth::Width128;
        let width256 = VectorWidth::Width256;
        let width512 = VectorWidth::Width512;
        
        assert_eq!(width128.element_count(&IrType::F32), 4);
        assert_eq!(width256.element_count(&IrType::F32), 8);
        assert_eq!(width512.element_count(&IrType::F32), 16);
        
        assert_eq!(width128.element_count(&IrType::F64), 2);
        assert_eq!(width256.element_count(&IrType::F64), 4);
        assert_eq!(width512.element_count(&IrType::F64), 8);
    }

    #[test]
    fn test_vectorizer_creation() {
        let vec = Vectorizer::new();
        assert_eq!(vec.name(), "SIMD Vectorization");
    }

    #[test]
    fn test_slp_vectorizer_creation() {
        let slp = SlpVectorizer::new();
        assert_eq!(slp.name(), "SLP Vectorization");
        assert_eq!(slp.max_pack_size, 4);
    }

    #[test]
    fn test_type_vectorizable() {
        let vec = Vectorizer::new();
        
        assert!(vec.is_type_vectorizable(&IrType::F32));
        assert!(vec.is_type_vectorizable(&IrType::F64));
        assert!(vec.is_type_vectorizable(&IrType::I32));
        assert!(!vec.is_type_vectorizable(&IrType::Void));
    }
}
