//! Lifetime inference and tracking

use std::collections::HashMap;
use crate::error::{Error, ErrorKind, Result};

/// Lifetime identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LifetimeId(usize);

impl LifetimeId {
    pub fn new(id: usize) -> Self {
        LifetimeId(id)
    }
    
    pub fn static_lifetime() -> Self {
        LifetimeId(0)
    }
}

/// Lifetime representation
#[derive(Debug, Clone, PartialEq)]
pub enum Lifetime {
    /// Static lifetime ('static)
    Static,
    /// Named lifetime ('a, 'b, etc.)
    Named(String),
    /// Inferred lifetime
    Inferred(LifetimeId),
    /// Anonymous lifetime
    Anonymous,
}

impl Lifetime {
    pub fn is_static(&self) -> bool {
        matches!(self, Lifetime::Static)
    }
    
    pub fn is_anonymous(&self) -> bool {
        matches!(self, Lifetime::Anonymous)
    }
}

/// Lifetime constraint
#[derive(Debug, Clone, PartialEq)]
pub enum LifetimeConstraint {
    /// 'a: 'b (a outlives b)
    Outlives(Lifetime, Lifetime),
    /// 'a == 'b (lifetimes are equal)
    Equal(Lifetime, Lifetime),
}

/// Lifetime environment for inference
#[derive(Debug, Clone)]
pub struct LifetimeEnv {
    /// Next lifetime ID
    next_id: usize,
    /// Named lifetime bindings
    bindings: HashMap<String, LifetimeId>,
    /// Lifetime constraints
    constraints: Vec<LifetimeConstraint>,
    /// Lifetime relationships (outlives graph)
    outlives: HashMap<LifetimeId, Vec<LifetimeId>>,
}

impl LifetimeEnv {
    pub fn new() -> Self {
        LifetimeEnv {
            next_id: 1, // 0 is reserved for 'static
            bindings: HashMap::new(),
            constraints: Vec::new(),
            outlives: HashMap::new(),
        }
    }
    
    /// Generate a fresh lifetime
    pub fn fresh_lifetime(&mut self) -> LifetimeId {
        let id = LifetimeId::new(self.next_id);
        self.next_id += 1;
        id
    }
    
    /// Bind a named lifetime
    pub fn bind_named(&mut self, name: String) -> LifetimeId {
        if let Some(&id) = self.bindings.get(&name) {
            id
        } else {
            let id = self.fresh_lifetime();
            self.bindings.insert(name, id);
            id
        }
    }
    
    /// Lookup a named lifetime
    pub fn lookup(&self, name: &str) -> Option<LifetimeId> {
        self.bindings.get(name).copied()
    }
    
    /// Add an outlives constraint: 'a: 'b
    pub fn add_outlives(&mut self, longer: LifetimeId, shorter: LifetimeId) {
        self.outlives.entry(longer)
            .or_insert_with(Vec::new)
            .push(shorter);
        
        self.constraints.push(LifetimeConstraint::Outlives(
            Lifetime::Inferred(longer),
            Lifetime::Inferred(shorter)
        ));
    }
    
    /// Add an equality constraint: 'a == 'b
    pub fn add_equal(&mut self, lt1: LifetimeId, lt2: LifetimeId) {
        self.constraints.push(LifetimeConstraint::Equal(
            Lifetime::Inferred(lt1),
            Lifetime::Inferred(lt2)
        ));
    }
    
    /// Check if 'a outlives 'b
    pub fn outlives(&self, longer: LifetimeId, shorter: LifetimeId) -> bool {
        if longer == shorter {
            return true;
        }
        
        if longer == LifetimeId::static_lifetime() {
            return true;
        }
        
        // Check direct relationship
        if let Some(outlived) = self.outlives.get(&longer) {
            if outlived.contains(&shorter) {
                return true;
            }
            
            // Check transitive relationships
            for &intermediate in outlived {
                if self.outlives(intermediate, shorter) {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Solve lifetime constraints
    pub fn solve_constraints(&mut self) -> Result<()> {
        // Simple constraint solving for now
        for constraint in &self.constraints.clone() {
            match constraint {
                LifetimeConstraint::Outlives(Lifetime::Inferred(longer), Lifetime::Inferred(shorter)) => {
                    if !self.outlives(*longer, *shorter) {
                        return Err(Error::new(
                            ErrorKind::BorrowError,
                            format!("Lifetime constraint violation: {:?} does not outlive {:?}", longer, shorter)
                        ));
                    }
                }
                LifetimeConstraint::Equal(Lifetime::Inferred(lt1), Lifetime::Inferred(lt2)) => {
                    // Merge lifetimes by adding bidirectional outlives
                    self.add_outlives(*lt1, *lt2);
                    self.add_outlives(*lt2, *lt1);
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Get all constraints
    pub fn constraints(&self) -> &[LifetimeConstraint] {
        &self.constraints
    }
}

impl Default for LifetimeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifetime annotation on types
#[derive(Debug, Clone, PartialEq)]
pub struct LifetimeAnnotation {
    pub lifetime: Lifetime,
}

impl LifetimeAnnotation {
    pub fn new(lifetime: Lifetime) -> Self {
        LifetimeAnnotation { lifetime }
    }
    
    pub fn static_annotation() -> Self {
        LifetimeAnnotation {
            lifetime: Lifetime::Static,
        }
    }
    
    pub fn anonymous() -> Self {
        LifetimeAnnotation {
            lifetime: Lifetime::Anonymous,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifetime_creation() {
        let mut env = LifetimeEnv::new();
        let lt1 = env.fresh_lifetime();
        let lt2 = env.fresh_lifetime();
        assert_ne!(lt1, lt2);
    }

    #[test]
    fn test_named_lifetime() {
        let mut env = LifetimeEnv::new();
        let lt = env.bind_named("a".to_string());
        assert_eq!(env.lookup("a"), Some(lt));
    }

    #[test]
    fn test_outlives_constraint() {
        let mut env = LifetimeEnv::new();
        let lt1 = env.fresh_lifetime();
        let lt2 = env.fresh_lifetime();
        
        env.add_outlives(lt1, lt2);
        assert!(env.outlives(lt1, lt2));
    }

    #[test]
    fn test_static_outlives_all() {
        let env = LifetimeEnv::new();
        let lt = LifetimeId::new(1);
        assert!(env.outlives(LifetimeId::static_lifetime(), lt));
    }

    #[test]
    fn test_transitive_outlives() {
        let mut env = LifetimeEnv::new();
        let lt1 = env.fresh_lifetime();
        let lt2 = env.fresh_lifetime();
        let lt3 = env.fresh_lifetime();
        
        env.add_outlives(lt1, lt2);
        env.add_outlives(lt2, lt3);
        
        assert!(env.outlives(lt1, lt3));
    }

    #[test]
    fn test_lifetime_equality() {
        let mut env = LifetimeEnv::new();
        let lt1 = env.fresh_lifetime();
        let lt2 = env.fresh_lifetime();
        
        env.add_equal(lt1, lt2);
        
        // After equality, both should outlive each other
        assert!(env.outlives(lt1, lt2));
        assert!(env.outlives(lt2, lt1));
    }
}
