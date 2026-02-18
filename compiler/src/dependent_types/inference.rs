//! Dependent type inference

use std::collections::HashMap;
use crate::ir::types::IrType;
use super::types::{DependentType, Nat, Constraint, BaseType};

pub struct DependentTypeInference {
    type_vars: HashMap<String, DependentType>,
    nat_vars: HashMap<String, Nat>,
    next_var: usize,
}

impl DependentTypeInference {
    pub fn new() -> Self {
        DependentTypeInference {
            type_vars: HashMap::new(),
            nat_vars: HashMap::new(),
            next_var: 0,
        }
    }
    
    pub fn fresh_type_var(&mut self) -> String {
        let name = format!("T{}", self.next_var);
        self.next_var += 1;
        name
    }
    
    pub fn fresh_nat_var(&mut self) -> String {
        let name = format!("n{}", self.next_var);
        self.next_var += 1;
        name
    }
    
    pub fn from_ir_type(ty: &IrType) -> DependentType {
        match ty {
            IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128 => {
                DependentType::Base(BaseType::Int)
            }
            IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64 | IrType::U128 => {
                DependentType::Base(BaseType::Int)
            }
            IrType::F16 | IrType::F32 | IrType::F64 | IrType::BF16 => {
                DependentType::Base(BaseType::Float)
            }
            IrType::Bool => DependentType::Base(BaseType::Bool),
            IrType::Void => DependentType::Base(BaseType::Unit),
            IrType::Array(elem, size) => {
                DependentType::Vec(
                    Box::new(Self::from_ir_type(elem)),
                    Nat::from_usize(*size)
                )
            }
            IrType::Pointer(inner) => {
                Self::from_ir_type(inner)
            }
            _ => DependentType::Base(BaseType::Unit),
        }
    }
    
    pub fn to_ir_type(ty: &DependentType) -> Option<IrType> {
        match ty {
            DependentType::Base(BaseType::Int) => Some(IrType::I64),
            DependentType::Base(BaseType::Float) => Some(IrType::F64),
            DependentType::Base(BaseType::Bool) => Some(IrType::Bool),
            DependentType::Base(BaseType::Unit) => Some(IrType::Void),
            DependentType::Vec(elem, len) => {
                let elem_ty = Self::to_ir_type(elem)?;
                len.to_usize().map(|n| IrType::Array(Box::new(elem_ty), n))
            }
            DependentType::NonZero(inner) => Self::to_ir_type(inner),
            DependentType::Positive(inner) => Self::to_ir_type(inner),
            DependentType::Ranged(inner, _, _) => Self::to_ir_type(inner),
            _ => None,
        }
    }
    
    pub fn infer_nonzero(&self, value: i64) -> Option<DependentType> {
        if value != 0 {
            Some(DependentType::NonZero(Box::new(DependentType::Base(BaseType::Int))))
        } else {
            None
        }
    }
    
    pub fn infer_positive(&self, value: i64) -> Option<DependentType> {
        if value > 0 {
            Some(DependentType::Positive(Box::new(DependentType::Base(BaseType::Int))))
        } else {
            None
        }
    }
    
    pub fn infer_ranged(&self, value: i64, lo: i64, hi: i64) -> Option<DependentType> {
        if value >= lo && value <= hi {
            Some(DependentType::Ranged(
                Box::new(DependentType::Base(BaseType::Int)),
                lo, hi
            ))
        } else {
            None
        }
    }
    
    pub fn create_vec_type(&mut self, elem_ty: DependentType, len: usize) -> DependentType {
        DependentType::Vec(Box::new(elem_ty), Nat::from_usize(len))
    }
    
    pub fn create_dependent_vec(&mut self) -> (DependentType, String) {
        let len_var = self.fresh_nat_var();
        let len = Nat::Var(len_var.clone());
        let elem_ty = DependentType::Base(BaseType::Int);
        (DependentType::Vec(Box::new(elem_ty), len), len_var)
    }
    
    pub fn add_bound_constraint(&mut self, index_var: &str, len_var: &str) -> Constraint {
        Constraint::LessThan(
            Nat::Var(index_var.to_string()),
            Nat::Var(len_var.to_string())
        )
    }
}

impl Default for DependentTypeInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_creation() {
        let inf = DependentTypeInference::new();
        assert!(inf.type_vars.is_empty());
    }

    #[test]
    fn test_ir_type_conversion() {
        let ir = IrType::I32;
        let dep = DependentTypeInference::from_ir_type(&ir);
        assert_eq!(dep, DependentType::Base(BaseType::Int));
    }

    #[test]
    fn test_nonzero_inference() {
        let inf = DependentTypeInference::new();
        
        assert!(inf.infer_nonzero(5).is_some());
        assert!(inf.infer_nonzero(0).is_none());
        assert!(inf.infer_nonzero(-5).is_some());
    }

    #[test]
    fn test_positive_inference() {
        let inf = DependentTypeInference::new();
        
        assert!(inf.infer_positive(5).is_some());
        assert!(inf.infer_positive(0).is_none());
        assert!(inf.infer_positive(-5).is_none());
    }
}
