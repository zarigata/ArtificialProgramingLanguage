//! Dependent type definitions

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Nat {
    Zero,
    Succ(Box<Nat>),
    Var(String),
}

impl Nat {
    pub fn zero() -> Self {
        Nat::Zero
    }
    
    pub fn one() -> Self {
        Nat::Succ(Box::new(Nat::Zero))
    }
    
    pub fn from_usize(n: usize) -> Self {
        let mut result = Nat::Zero;
        for _ in 0..n {
            result = Nat::Succ(Box::new(result));
        }
        result
    }
    
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            Nat::Zero => Some(0),
            Nat::Succ(inner) => inner.to_usize().map(|n| n + 1),
            Nat::Var(_) => None,
        }
    }
    
    pub fn add(&self, other: &Nat) -> Nat {
        match (self, other) {
            (Nat::Zero, _) => other.clone(),
            (Nat::Succ(n), _) => Nat::Succ(Box::new(n.add(other))),
            (Nat::Var(_), Nat::Zero) => self.clone(),
            (Nat::Var(_), _) => Nat::Var(format!("{:?}+{:?}", self, other)),
        }
    }
    
    pub fn sub(&self, other: &Nat) -> Option<Nat> {
        match (self, other) {
            (_, Nat::Zero) => Some(self.clone()),
            (Nat::Succ(a), Nat::Succ(b)) => a.sub(b),
            _ => None,
        }
    }
    
    pub fn mul(&self, other: &Nat) -> Nat {
        match (self, other) {
            (Nat::Zero, _) | (_, Nat::Zero) => Nat::Zero,
            (Nat::Succ(n), m) => m.add(&n.mul(m)),
            (Nat::Var(_), _) | (_, Nat::Var(_)) => Nat::Var(format!("{:?}*{:?}", self, other)),
        }
    }
    
    pub fn is_less_than(&self, other: &Nat) -> bool {
        match (self.to_usize(), other.to_usize()) {
            (Some(a), Some(b)) => a < b,
            _ => false,
        }
    }
}

impl fmt::Display for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nat::Zero => write!(f, "0"),
            Nat::Succ(inner) => write!(f, "S({})", inner),
            Nat::Var(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vec<T> {
    pub elem_type: Box<DependentType>,
    pub length: Nat,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Vec<T> {
    pub fn new(elem: DependentType, len: Nat) -> Self {
        Vec {
            elem_type: Box::new(elem),
            length: len,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SafeRef<T> {
    pub base_type: Box<DependentType>,
    pub index: Nat,
    pub bound: Nat,
    _marker: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    LessThan(Nat, Nat),
    LessThanOrEqual(Nat, Nat),
    Equal(Nat, Nat),
    NotEqual(Nat, Nat),
    And(Box<Constraint>, Box<Constraint>),
    Or(Box<Constraint>, Box<Constraint>),
    True,
    False,
}

impl Constraint {
    pub fn is_satisfiable(&self) -> bool {
        match self {
            Constraint::True => true,
            Constraint::False => false,
            Constraint::LessThan(a, b) => a.is_less_than(b),
            Constraint::LessThanOrEqual(a, b) => {
                match (a.to_usize(), b.to_usize()) {
                    (Some(x), Some(y)) => x <= y,
                    _ => true,
                }
            }
            Constraint::Equal(a, b) => a == b,
            Constraint::NotEqual(a, b) => a != b,
            Constraint::And(l, r) => l.is_satisfiable() && r.is_satisfiable(),
            Constraint::Or(l, r) => l.is_satisfiable() || r.is_satisfiable(),
        }
    }
    
    pub fn simplify(&self) -> Constraint {
        match self {
            Constraint::And(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Constraint::True, _) => rs,
                    (_, Constraint::True) => ls,
                    (Constraint::False, _) | (_, Constraint::False) => Constraint::False,
                    _ => Constraint::And(Box::new(ls), Box::new(rs)),
                }
            }
            Constraint::Or(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Constraint::False, _) => rs,
                    (_, Constraint::False) => ls,
                    (Constraint::True, _) | (_, Constraint::True) => Constraint::True,
                    _ => Constraint::Or(Box::new(ls), Box::new(rs)),
                }
            }
            other => other.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DependentType {
    Base(BaseType),
    Vec(Box<DependentType>, Nat),
    SafeRef(Box<DependentType>, Nat, Nat),
    Dependent(String, Box<DependentType>, Constraint),
    NonZero(Box<DependentType>),
    Positive(Box<DependentType>),
    Sorted(Box<DependentType>),
    Ranged(Box<DependentType>, i64, i64),
    Predicate(Box<DependentType>, String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BaseType {
    Int,
    Float,
    Bool,
    String,
    Char,
    Unit,
}

impl fmt::Display for DependentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependentType::Base(b) => match b {
                BaseType::Int => write!(f, "int"),
                BaseType::Float => write!(f, "float"),
                BaseType::Bool => write!(f, "bool"),
                BaseType::String => write!(f, "string"),
                BaseType::Char => write!(f, "char"),
                BaseType::Unit => write!(f, "()"),
            },
            DependentType::Vec(elem, len) => write!(f, "Vec<{}, {}>", elem, len),
            DependentType::SafeRef(ty, idx, bound) => write!(f, "Ref<{}, {}<{}>", ty, idx, bound),
            DependentType::Dependent(name, ty, constraint) => {
                write!(f, "{}: {} where {}", name, ty, constraint)
            }
            DependentType::NonZero(ty) => write!(f, "NonZero<{}>", ty),
            DependentType::Positive(ty) => write!(f, "Positive<{}>", ty),
            DependentType::Sorted(ty) => write!(f, "Sorted<{}>", ty),
            DependentType::Ranged(ty, lo, hi) => write!(f, "{}<{}..{}>", ty, lo, hi),
            DependentType::Predicate(ty, pred) => write!(f, "{} where {}", ty, pred),
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::LessThan(a, b) => write!(f, "{} < {}", a, b),
            Constraint::LessThanOrEqual(a, b) => write!(f, "{} <= {}", a, b),
            Constraint::Equal(a, b) => write!(f, "{} = {}", a, b),
            Constraint::NotEqual(a, b) => write!(f, "{} != {}", a, b),
            Constraint::And(l, r) => write!(f, "({} && {})", l, r),
            Constraint::Or(l, r) => write!(f, "({} || {})", l, r),
            Constraint::True => write!(f, "true"),
            Constraint::False => write!(f, "false"),
        }
    }
}

pub type NonZero = std::marker::PhantomData<()>;
pub type Positive = std::marker::PhantomData<()>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_arithmetic() {
        let zero = Nat::zero();
        let one = Nat::one();
        let two = Nat::from_usize(2);
        
        assert_eq!(zero.to_usize(), Some(0));
        assert_eq!(one.to_usize(), Some(1));
        assert_eq!(two.to_usize(), Some(2));
        
        let three = one.add(&two);
        assert_eq!(three.to_usize(), Some(3));
        
        let four = two.mul(&two);
        assert_eq!(four.to_usize(), Some(4));
    }

    #[test]
    fn test_constraints() {
        let c = Constraint::LessThan(Nat::one(), Nat::from_usize(2));
        assert!(c.is_satisfiable());
        
        let c2 = Constraint::LessThan(Nat::from_usize(3), Nat::one());
        assert!(!c2.is_satisfiable());
    }
}
