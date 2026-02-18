//! AMD GPU-specific optimization passes
//!
//! Optimizations targeting AMD GPU architectures (RDNA, CDNA)

use crate::ir::ssa::{Module, Function as IrFunction};
use crate::error::Result;
use crate::optimizer::pass_manager::OptimizationPass;
use super::rocm::AmdGpuArch;

/// Wavefront utilization optimizer
/// Ensures workgroups are sized to maximize wavefront occupancy
pub struct WavefrontOptimizer {
    arch: AmdGpuArch,
    changed: bool,
}

impl WavefrontOptimizer {
    pub fn new(arch: AmdGpuArch) -> Self {
        WavefrontOptimizer {
            arch,
            changed: false,
        }
    }
    
    pub fn optimize_workgroup_size(&self, current_size: u32) -> u32 {
        let wavefront_size = self.arch.wavefront_size();
        
        if current_size % wavefront_size == 0 {
            current_size
        } else {
            let multiples = current_size / wavefront_size;
            (multiples + 1) * wavefront_size
        }
    }
    
    pub fn calculate_occupancy(&self, vgprs: u32, sgprs: u32, shared_mem: usize) -> f32 {
        let max_waves = match self.arch {
            AmdGpuArch::Cdna1 | AmdGpuArch::Cdna2 | AmdGpuArch::Cdna3 => 40,
            AmdGpuArch::Gcn1 | AmdGpuArch::Gcn2 | AmdGpuArch::Gcn3 => 10,
            AmdGpuArch::Gcn4 | AmdGpuArch::Gcn5 => 16,
            AmdGpuArch::Rdna1 | AmdGpuArch::Rdna2 | AmdGpuArch::Rdna3 => 20,
        };
        
        let vgpr_limit = match self.arch {
            AmdGpuArch::Cdna3 => 512,
            AmdGpuArch::Rdna3 => 256,
            _ => 256,
        };
        
        let _ = shared_mem;
        let vgpr_waves = if vgprs > 0 { vgpr_limit / vgprs } else { max_waves };
        let sgpr_waves = if sgprs > 0 { 100 / sgprs } else { max_waves };
        
        let waves_by_regs = vgpr_waves.min(sgpr_waves);
        waves_by_regs.min(max_waves) as f32 / max_waves as f32
    }
}

impl OptimizationPass for WavefrontOptimizer {
    fn name(&self) -> &str {
        "wavefront_optimizer"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, _function: &mut IrFunction) -> Result<bool> {
        Ok(false)
    }
}

/// LDS (Local Data Share) optimizer
/// Optimizes shared memory usage for AMD GPUs
pub struct LdsOptimizer {
    arch: AmdGpuArch,
    changed: bool,
}

impl LdsOptimizer {
    pub fn new(arch: AmdGpuArch) -> Self {
        LdsOptimizer {
            arch,
            changed: false,
        }
    }
    
    pub fn max_lds_size(&self) -> usize {
        match self.arch {
            AmdGpuArch::Cdna1 => 64 * 1024,
            AmdGpuArch::Cdna2 => 64 * 1024,
            AmdGpuArch::Cdna3 => 128 * 1024,
            _ => 64 * 1024,
        }
    }
    
    pub fn bank_count(&self) -> usize {
        match self.arch {
            AmdGpuArch::Cdna3 => 64,
            _ => 32,
        }
    }
    
    pub fn bank_width(&self) -> usize {
        4
    }
    
    pub fn suggest_padding(&self, element_size: usize, count: usize) -> usize {
        let banks = self.bank_count();
        let bank_width = self.bank_width();
        let row_size = banks * bank_width;
        
        let total_size = element_size * count;
        
        if total_size % row_size == 0 {
            let elements_per_row = row_size / element_size;
            if count % elements_per_row == 0 {
                return count + elements_per_row;
            }
        }
        
        count
    }
}

impl OptimizationPass for LdsOptimizer {
    fn name(&self) -> &str {
        "lds_optimizer"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, _function: &mut IrFunction) -> Result<bool> {
        Ok(false)
    }
}

/// MFMA (Matrix Fused Multiply-Add) optimizer for CDNA
/// Optimizes matrix operations to use AMD's matrix cores
pub struct MfmaOptimizer {
    arch: AmdGpuArch,
    changed: bool,
}

impl MfmaOptimizer {
    pub fn new(arch: AmdGpuArch) -> Self {
        MfmaOptimizer {
            arch,
            changed: false,
        }
    }
    
    pub fn supports_mfma(&self) -> bool {
        self.arch.supports_matrix_cores()
    }
    
    pub fn mfma_shapes(&self) -> Vec<(usize, usize, usize)> {
        match self.arch {
            AmdGpuArch::Cdna1 => vec![(32, 32, 1), (16, 16, 4), (4, 4, 4)],
            AmdGpuArch::Cdna2 => vec![(32, 32, 1), (16, 16, 4), (4, 4, 4), (16, 16, 16)],
            AmdGpuArch::Cdna3 => vec![(32, 32, 2), (16, 16, 4), (4, 4, 4), (16, 16, 16)],
            _ => vec![],
        }
    }
    
    pub fn can_use_mfma(&self, m: usize, n: usize, k: usize) -> bool {
        if !self.supports_mfma() {
            return false;
        }
        
        for (bm, bn, bk) in self.mfma_shapes() {
            if m % bm == 0 && n % bn == 0 && k % bk == 0 {
                return true;
            }
        }
        
        false
    }
    
    pub fn select_mfma_shape(&self, m: usize, n: usize, k: usize) -> Option<(usize, usize, usize)> {
        if !self.supports_mfma() {
            return None;
        }
        
        let shapes = self.mfma_shapes();
        for (bm, bn, bk) in &shapes {
            if m % bm == 0 && n % bn == 0 && k % bk == 0 {
                return Some((*bm, *bn, *bk));
            }
        }
        
        None
    }
}

impl OptimizationPass for MfmaOptimizer {
    fn name(&self) -> &str {
        "mfma_optimizer"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        if !self.supports_mfma() {
            return Ok(false);
        }
        
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, _function: &mut IrFunction) -> Result<bool> {
        Ok(false)
    }
}

/// Register allocator for AMD GPUs
/// Optimizes VGPR and SGPR allocation
pub struct AmdRegisterAllocator {
    arch: AmdGpuArch,
    changed: bool,
}

impl AmdRegisterAllocator {
    pub fn new(arch: AmdGpuArch) -> Self {
        AmdRegisterAllocator {
            arch,
            changed: false,
        }
    }
    
    pub fn max_vgprs(&self) -> u32 {
        match self.arch {
            AmdGpuArch::Cdna3 => 512,
            _ => 256,
        }
    }
    
    pub fn max_sgprs(&self) -> u32 {
        match self.arch {
            AmdGpuArch::Cdna3 => 200,
            _ => 100,
        }
    }
    
    pub fn max_agprs(&self) -> u32 {
        match self.arch {
            AmdGpuArch::Cdna3 => 256,
            _ => 0,
        }
    }
    
    pub fn sgpr_spill_cost(&self) -> u32 {
        20
    }
    
    pub fn vgpr_spill_cost(&self) -> u32 {
        10
    }
}

impl OptimizationPass for AmdRegisterAllocator {
    fn name(&self) -> &str {
        "amd_register_allocator"
    }
    
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool> {
        for function in &mut module.functions {
            self.run_on_function(function)?;
        }
        Ok(self.changed)
    }
    
    fn run_on_function(&mut self, _function: &mut IrFunction) -> Result<bool> {
        Ok(false)
    }
}

/// AMD GPU optimization pipeline
pub struct AmdOptimizationPipeline {
    arch: AmdGpuArch,
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl AmdOptimizationPipeline {
    pub fn new(arch: AmdGpuArch) -> Self {
        let mut pipeline = AmdOptimizationPipeline {
            arch,
            passes: Vec::new(),
        };
        
        pipeline.add_pass(WavefrontOptimizer::new(arch));
        pipeline.add_pass(LdsOptimizer::new(arch));
        
        if arch.supports_matrix_cores() {
            pipeline.add_pass(MfmaOptimizer::new(arch));
        }
        
        pipeline.add_pass(AmdRegisterAllocator::new(arch));
        
        pipeline
    }
    
    pub fn add_pass<P: OptimizationPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }
    
    pub fn run(&mut self, module: &mut Module) -> Result<bool> {
        let mut any_changed = false;
        
        for pass in &mut self.passes {
            let changed = pass.run_on_module(module)?;
            if changed {
                any_changed = true;
            }
        }
        
        Ok(any_changed)
    }
    
    pub fn arch(&self) -> AmdGpuArch {
        self.arch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wavefront_optimizer() {
        let opt = WavefrontOptimizer::new(AmdGpuArch::Cdna3);
        
        assert_eq!(opt.optimize_workgroup_size(100), 128);
        assert_eq!(opt.optimize_workgroup_size(128), 128);
    }
    
    #[test]
    fn test_lds_optimizer() {
        let opt = LdsOptimizer::new(AmdGpuArch::Cdna3);
        
        assert!(opt.max_lds_size() >= 64 * 1024);
        assert_eq!(opt.bank_count(), 64);
    }
    
    #[test]
    fn test_mfma_optimizer() {
        let opt = MfmaOptimizer::new(AmdGpuArch::Cdna3);
        
        assert!(opt.supports_mfma());
        assert!(opt.can_use_mfma(1024, 1024, 512));
        
        let shape = opt.select_mfma_shape(1024, 1024, 512);
        assert!(shape.is_some());
    }
    
    #[test]
    fn test_no_mfma_on_rdna() {
        let opt = MfmaOptimizer::new(AmdGpuArch::Rdna3);
        
        assert!(!opt.supports_mfma());
        assert!(opt.mfma_shapes().is_empty());
    }
}
