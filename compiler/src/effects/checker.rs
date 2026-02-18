//! Effect checking and validation

use std::collections::HashMap;
use crate::parser::ast::{Function as AstFunction, Item};
use crate::error::Result;
use super::effect::{EffectSet, EffectKind};
use super::inference::EffectInference;

#[derive(Debug, Clone)]
pub struct EffectError {
    pub message: String,
    pub expected: Option<EffectSet>,
    pub actual: EffectSet,
}

impl EffectError {
    pub fn new(message: &str, actual: EffectSet) -> Self {
        EffectError {
            message: message.to_string(),
            expected: None,
            actual,
        }
    }
    
    pub fn with_expected(message: &str, expected: EffectSet, actual: EffectSet) -> Self {
        EffectError {
            message: message.to_string(),
            expected: Some(expected),
            actual,
        }
    }
}

pub struct EffectChecker {
    inference: EffectInference,
    allowed_effects: HashMap<String, EffectSet>,
    strict_mode: bool,
}

impl EffectChecker {
    pub fn new() -> Self {
        EffectChecker {
            inference: EffectInference::new(),
            allowed_effects: HashMap::new(),
            strict_mode: false,
        }
    }
    
    pub fn strict() -> Self {
        EffectChecker {
            inference: EffectInference::new(),
            allowed_effects: HashMap::new(),
            strict_mode: true,
        }
    }
    
    pub fn allow_effects(&mut self, function: &str, effects: EffectSet) {
        self.allowed_effects.insert(function.to_string(), effects);
    }
    
    pub fn check_module(&mut self, items: &[Item]) -> Result<Vec<EffectError>> {
        let mut errors = Vec::new();
        
        for item in items {
            if let Item::Function(func) = item {
                if let Err(e) = self.check_function(func) {
                    errors.push(e);
                }
            }
        }
        
        Ok(errors)
    }
    
    pub fn check_function(&mut self, func: &AstFunction) -> std::result::Result<(), EffectError> {
        let inferred = self.inference.infer_function(func)
            .map_err(|e| EffectError::new(&e.to_string(), EffectSet::unknown()))?;
        
        if let Some(annotation) = self.extract_effect_annotation(func) {
            if !self.effects_subset(&inferred, &annotation) {
                return Err(EffectError::with_expected(
                    &format!("Function '{}' has more effects than annotated", func.name),
                    annotation,
                    inferred,
                ));
            }
        }
        
        if self.strict_mode {
            let has_unknown = inferred.iter().any(|e| matches!(e, EffectKind::Unknown));
            if has_unknown {
                return Err(EffectError::new(
                    &format!("Function '{}' has unknown effects in strict mode", func.name),
                    inferred,
                ));
            }
        }
        
        Ok(())
    }
    
    fn extract_effect_annotation(&self, func: &AstFunction) -> Option<EffectSet> {
        for attr in &func.attributes {
            if attr.name == "effects" {
                let mut set = EffectSet::new();
                if let Some(crate::parser::ast::Expr::Literal(crate::parser::ast::Literal::String(s))) = &attr.value {
                    for arg in s.split(',') {
                        let arg = arg.trim();
                        match arg {
                            "IO" => {
                                set.insert(EffectKind::IoRead);
                                set.insert(EffectKind::IoWrite);
                            }
                            "IO.read" => set.insert(EffectKind::IoRead),
                            "IO.write" => set.insert(EffectKind::IoWrite),
                            "State" => {
                                set.insert(EffectKind::StateRead);
                                set.insert(EffectKind::StateWrite);
                            }
                            "State.read" => set.insert(EffectKind::StateRead),
                            "State.write" => set.insert(EffectKind::StateWrite),
                            "Async" => {
                                set.insert(EffectKind::AsyncYield);
                                set.insert(EffectKind::AsyncSpawn);
                            }
                            "Pure" => set.insert(EffectKind::Pure),
                            "Throw" => set.insert(EffectKind::Throw),
                            _ => set.insert(EffectKind::Custom(arg.to_string())),
                        }
                    }
                }
                return Some(set);
            }
        }
        None
    }
    
    fn effects_subset(&self, subset: &EffectSet, superset: &EffectSet) -> bool {
        for effect in subset.iter() {
            if !superset.contains(effect) && !matches!(effect, EffectKind::Pure) {
                return false;
            }
        }
        true
    }
    
    pub fn get_inferred_effects(&self, function: &str) -> Option<&EffectSet> {
        self.inference.get_function_effects(function)
    }
    
    pub fn is_pure_function(&self, function: &str) -> bool {
        self.inference.get_function_effects(function)
            .map(|e| e.is_pure())
            .unwrap_or(false)
    }
    
    pub fn check_purity(&mut self, func: &AstFunction) -> Result<bool> {
        let effects = self.inference.infer_function(func)?;
        Ok(effects.is_pure())
    }
    
    pub fn analyze_call_graph(&mut self, functions: &[AstFunction]) -> HashMap<String, EffectSet> {
        let mut results = HashMap::new();
        
        for func in functions {
            if let Ok(effects) = self.inference.infer_function(func) {
                results.insert(func.name.clone(), effects);
            }
        }
        
        results
    }
}

impl Default for EffectChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checker_creation() {
        let checker = EffectChecker::new();
        assert!(!checker.strict_mode);
    }

    #[test]
    fn test_strict_mode() {
        let checker = EffectChecker::strict();
        assert!(checker.strict_mode);
    }

    #[test]
    fn test_allowed_effects() {
        let mut checker = EffectChecker::new();
        let mut effects = EffectSet::new();
        effects.insert(EffectKind::IoRead);
        
        checker.allow_effects("read_input", effects);
        assert!(checker.allowed_effects.contains_key("read_input"));
    }

    #[test]
    fn test_effects_subset() {
        let checker = EffectChecker::new();
        
        let mut subset = EffectSet::new();
        subset.insert(EffectKind::Pure);
        
        let mut superset = EffectSet::new();
        superset.insert(EffectKind::IoRead);
        superset.insert(EffectKind::Pure);
        
        assert!(checker.effects_subset(&subset, &superset));
    }
}
