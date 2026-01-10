//! Optimization pass manager

use crate::ir::ssa::{Module, Function as IrFunction};
use crate::error::Result;

/// Trait for optimization passes
pub trait OptimizationPass {
    /// Get the name of this pass
    fn name(&self) -> &str;
    
    /// Run the pass on a module
    fn run_on_module(&mut self, module: &mut Module) -> Result<bool>;
    
    /// Run the pass on a function
    fn run_on_function(&mut self, function: &mut IrFunction) -> Result<bool>;
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Moderate optimization
    O2,
    /// Aggressive optimization
    O3,
}

/// Pass manager for running optimization passes
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
    opt_level: OptLevel,
    iterations: usize,
}

impl PassManager {
    /// Create a new pass manager
    pub fn new(opt_level: OptLevel) -> Self {
        PassManager {
            passes: Vec::new(),
            opt_level,
            iterations: match opt_level {
                OptLevel::O0 => 0,
                OptLevel::O1 => 1,
                OptLevel::O2 => 3,
                OptLevel::O3 => 5,
            },
        }
    }
    
    /// Add a pass to the manager
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }
    
    /// Run all passes on a module
    pub fn run(&mut self, module: &mut Module) -> Result<()> {
        if self.opt_level == OptLevel::O0 {
            return Ok(());
        }
        
        // Run passes until convergence or max iterations
        for iteration in 0..self.iterations {
            let mut changed = false;
            
            for pass in &mut self.passes {
                let pass_changed = pass.run_on_module(module)?;
                changed |= pass_changed;
                
                if pass_changed {
                    println!("  [Iteration {}] {} made changes", iteration + 1, pass.name());
                }
            }
            
            // If no pass made changes, we've converged
            if !changed {
                println!("  Converged after {} iterations", iteration + 1);
                break;
            }
        }
        
        Ok(())
    }
    
    /// Get the optimization level
    pub fn opt_level(&self) -> OptLevel {
        self.opt_level
    }
    
    /// Get the number of passes
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new(OptLevel::O2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPass {
        name: String,
        should_change: bool,
    }

    impl DummyPass {
        fn new(name: &str, should_change: bool) -> Self {
            DummyPass {
                name: name.to_string(),
                should_change,
            }
        }
    }

    impl OptimizationPass for DummyPass {
        fn name(&self) -> &str {
            &self.name
        }

        fn run_on_module(&mut self, _module: &mut Module) -> Result<bool> {
            Ok(self.should_change)
        }

        fn run_on_function(&mut self, _function: &mut IrFunction) -> Result<bool> {
            Ok(self.should_change)
        }
    }

    #[test]
    fn test_pass_manager_creation() {
        let pm = PassManager::new(OptLevel::O2);
        assert_eq!(pm.opt_level(), OptLevel::O2);
        assert_eq!(pm.pass_count(), 0);
    }

    #[test]
    fn test_add_pass() {
        let mut pm = PassManager::new(OptLevel::O1);
        pm.add_pass(Box::new(DummyPass::new("test", false)));
        assert_eq!(pm.pass_count(), 1);
    }

    #[test]
    fn test_opt_levels() {
        let pm0 = PassManager::new(OptLevel::O0);
        let pm1 = PassManager::new(OptLevel::O1);
        let pm2 = PassManager::new(OptLevel::O2);
        let pm3 = PassManager::new(OptLevel::O3);
        
        assert_eq!(pm0.iterations, 0);
        assert_eq!(pm1.iterations, 1);
        assert_eq!(pm2.iterations, 3);
        assert_eq!(pm3.iterations, 5);
    }

    #[test]
    fn test_run_no_optimization() {
        let mut pm = PassManager::new(OptLevel::O0);
        let mut module = Module::new("test".to_string());
        
        assert!(pm.run(&mut module).is_ok());
    }
}
