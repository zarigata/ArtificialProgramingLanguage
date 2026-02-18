//! Dependent type checking

use std::collections::HashMap;
use super::types::{DependentType, Nat, Constraint, BaseType};

pub struct DependentTypeChecker {
    constraints: Vec<Constraint>,
    substitutions: HashMap<String, Nat>,
}

impl DependentTypeChecker {
    pub fn new() -> Self {
        DependentTypeChecker {
            constraints: Vec::new(),
            substitutions: HashMap::new(),
        }
    }
    
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }
    
    pub fn check_all(&self) -> bool {
        self.constraints.iter().all(|c| self.check_constraint(c))
    }
    
    fn check_constraint(&self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::True => true,
            Constraint::False => false,
            Constraint::LessThan(a, b) => {
                let a_val = self.eval_nat(a);
                let b_val = self.eval_nat(b);
                match (a_val.to_usize(), b_val.to_usize()) {
                    (Some(x), Some(y)) => x < y,
                    _ => true,
                }
            }
            Constraint::LessThanOrEqual(a, b) => {
                let a_val = self.eval_nat(a);
                let b_val = self.eval_nat(b);
                match (a_val.to_usize(), b_val.to_usize()) {
                    (Some(x), Some(y)) => x <= y,
                    _ => true,
                }
            }
            Constraint::Equal(a, b) => {
                self.eval_nat(a) == self.eval_nat(b)
            }
            Constraint::NotEqual(a, b) => {
                self.eval_nat(a) != self.eval_nat(b)
            }
            Constraint::And(l, r) => {
                self.check_constraint(l) && self.check_constraint(r)
            }
            Constraint::Or(l, r) => {
                self.check_constraint(l) || self.check_constraint(r)
            }
        }
    }
    
    fn eval_nat(&self, n: &Nat) -> Nat {
        match n {
            Nat::Zero => Nat::Zero,
            Nat::Succ(inner) => Nat::Succ(Box::new(self.eval_nat(inner))),
            Nat::Var(name) => {
                self.substitutions.get(name).cloned().unwrap_or_else(|| n.clone())
            }
        }
    }
    
    pub fn unify(&mut self, expected: &DependentType, actual: &DependentType) -> bool {
        match (expected, actual) {
            (DependentType::Base(a), DependentType::Base(b)) => a == b,
            
            (DependentType::Vec(e1, l1), DependentType::Vec(e2, l2)) => {
                self.unify(e1, e2) && self.unify_nat(l1, l2)
            }
            
            (DependentType::NonZero(inner), DependentType::Base(BaseType::Int)) => {
                matches!(inner.as_ref(), DependentType::Base(BaseType::Int))
            }
            
            (DependentType::Base(BaseType::Int), DependentType::NonZero(_)) => {
                self.add_constraint(Constraint::False);
                false
            }
            
            (DependentType::Ranged(t, lo, hi), DependentType::Base(BaseType::Int)) => {
                matches!(t.as_ref(), DependentType::Base(BaseType::Int))
            }
            
            (DependentType::Dependent(_, t1, c1), DependentType::Dependent(_, t2, c2)) => {
                self.add_constraint(c1.clone());
                self.add_constraint(c2.clone());
                self.unify(t1, t2)
            }
            
            (DependentType::Dependent(_, ty, c), other) => {
                self.add_constraint(c.clone());
                self.unify(ty, other)
            }
            
            (other, DependentType::Dependent(_, ty, c)) => {
                self.add_constraint(c.clone());
                self.unify(other, ty)
            }
            
            _ => expected == actual,
        }
    }
    
    fn unify_nat(&mut self, a: &Nat, b: &Nat) -> bool {
        match (a, b) {
            (Nat::Zero, Nat::Zero) => true,
            (Nat::Succ(a1), Nat::Succ(b1)) => self.unify_nat(a1, b1),
            (Nat::Var(name), other) | (other, Nat::Var(name)) => {
                if let Some(existing) = self.substitutions.get(name) {
                    existing == other
                } else {
                    self.substitutions.insert(name.clone(), other.clone());
                    true
                }
            }
            _ => false,
        }
    }
    
    pub fn check_bounds(&mut self, index: &Nat, length: &Nat) -> bool {
        let constraint = Constraint::LessThan(index.clone(), length.clone());
        self.add_constraint(constraint.clone());
        constraint.is_satisfiable()
    }
    
    pub fn check_nonzero(&self, value: i64) -> bool {
        value != 0
    }
    
    pub fn check_positive(&self, value: i64) -> bool {
        value > 0
    }
    
    pub fn check_ranged(&self, value: i64, lo: i64, hi: i64) -> bool {
        value >= lo && value <= hi
    }
    
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.substitutions.clear();
    }
}

impl Default for DependentTypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checker_creation() {
        let checker = DependentTypeChecker::new();
        assert!(checker.constraints.is_empty());
    }

    #[test]
    fn test_base_type_unification() {
        let mut checker = DependentTypeChecker::new();
        
        assert!(checker.unify(
            &DependentType::Base(BaseType::Int),
            &DependentType::Base(BaseType::Int)
        ));
        
        assert!(!checker.unify(
            &DependentType::Base(BaseType::Int),
            &DependentType::Base(BaseType::Bool)
        ));
    }

    #[test]
    fn test_bounds_checking() {
        let mut checker = DependentTypeChecker::new();
        
        assert!(checker.check_bounds(&Nat::from_usize(2), &Nat::from_usize(5)));
    }
}
