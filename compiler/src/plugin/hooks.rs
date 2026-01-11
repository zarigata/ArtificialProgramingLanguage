// Plugin Hooks
// Hook system for compiler phases

use super::*;

pub struct HookRegistry {
    hooks: Vec<Box<dyn Fn() -> Result<()>>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        HookRegistry {
            hooks: Vec::new(),
        }
    }
    
    pub fn register<F>(&mut self, hook: F)
    where
        F: Fn() -> Result<()> + 'static,
    {
        self.hooks.push(Box::new(hook));
    }
    
    pub fn execute_all(&self) -> Result<()> {
        for hook in &self.hooks {
            hook()?;
        }
        Ok(())
    }
}
