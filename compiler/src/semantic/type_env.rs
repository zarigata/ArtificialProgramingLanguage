//! Type environment and type inference

use std::collections::HashMap;
use crate::parser::Type;
use crate::error::{Error, ErrorKind, Result};

/// Type variable for inference
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVar(usize);

impl TypeVar {
    pub fn new(id: usize) -> Self {
        TypeVar(id)
    }
}

/// Inferred type representation
#[derive(Debug, Clone, PartialEq)]
pub enum InferredType {
    /// Concrete type from AST
    Concrete(Type),
    /// Type variable (to be inferred)
    Var(TypeVar),
    /// Function type
    Function(Vec<InferredType>, Box<InferredType>),
    /// Generic type with arguments
    Generic(String, Vec<InferredType>),
    /// Tuple type
    Tuple(Vec<InferredType>),
    /// Array type
    Array(Box<InferredType>, usize),
    /// Reference type
    Reference(Box<InferredType>),
    /// Mutable reference type
    MutableReference(Box<InferredType>),
}

impl InferredType {
    /// Create a type from AST Type
    pub fn from_ast(ty: &Type) -> Self {
        match ty {
            Type::Named(name) => InferredType::Concrete(Type::Named(name.clone())),
            Type::Generic(name, args) => {
                let inferred_args = args.iter().map(InferredType::from_ast).collect();
                InferredType::Generic(name.clone(), inferred_args)
            }
            Type::Reference(inner) => {
                InferredType::Reference(Box::new(InferredType::from_ast(inner)))
            }
            Type::MutableReference(inner) => {
                InferredType::MutableReference(Box::new(InferredType::from_ast(inner)))
            }
            Type::Array(inner, size) => {
                InferredType::Array(Box::new(InferredType::from_ast(inner)), *size)
            }
            Type::Tuple(types) => {
                let inferred = types.iter().map(InferredType::from_ast).collect();
                InferredType::Tuple(inferred)
            }
            Type::Function(params, ret) => {
                let param_types = params.iter().map(InferredType::from_ast).collect();
                let ret_type = Box::new(InferredType::from_ast(ret));
                InferredType::Function(param_types, ret_type)
            }
            Type::TraitObject(_) => {
                // Trait objects handled separately
                InferredType::Concrete(ty.clone())
            }
        }
    }
    
    /// Check if type contains a type variable
    pub fn contains_var(&self, var: &TypeVar) -> bool {
        match self {
            InferredType::Var(v) => v == var,
            InferredType::Function(params, ret) => {
                params.iter().any(|p| p.contains_var(var)) || ret.contains_var(var)
            }
            InferredType::Generic(_, args) => {
                args.iter().any(|a| a.contains_var(var))
            }
            InferredType::Tuple(types) => {
                types.iter().any(|t| t.contains_var(var))
            }
            InferredType::Array(inner, _) => inner.contains_var(var),
            InferredType::Reference(inner) | InferredType::MutableReference(inner) => {
                inner.contains_var(var)
            }
            InferredType::Concrete(_) => false,
        }
    }
}

/// Substitution mapping type variables to types
#[derive(Debug, Clone)]
pub struct Substitution {
    map: HashMap<TypeVar, InferredType>,
}

impl Substitution {
    pub fn new() -> Self {
        Substitution {
            map: HashMap::new(),
        }
    }
    
    /// Add a binding
    pub fn bind(&mut self, var: TypeVar, ty: InferredType) {
        self.map.insert(var, ty);
    }
    
    /// Apply substitution to a type
    pub fn apply(&self, ty: &InferredType) -> InferredType {
        match ty {
            InferredType::Var(var) => {
                if let Some(substituted) = self.map.get(var) {
                    self.apply(substituted)
                } else {
                    ty.clone()
                }
            }
            InferredType::Function(params, ret) => {
                let new_params = params.iter().map(|p| self.apply(p)).collect();
                let new_ret = Box::new(self.apply(ret));
                InferredType::Function(new_params, new_ret)
            }
            InferredType::Generic(name, args) => {
                let new_args = args.iter().map(|a| self.apply(a)).collect();
                InferredType::Generic(name.clone(), new_args)
            }
            InferredType::Tuple(types) => {
                let new_types = types.iter().map(|t| self.apply(t)).collect();
                InferredType::Tuple(new_types)
            }
            InferredType::Array(inner, size) => {
                InferredType::Array(Box::new(self.apply(inner)), *size)
            }
            InferredType::Reference(inner) => {
                InferredType::Reference(Box::new(self.apply(inner)))
            }
            InferredType::MutableReference(inner) => {
                InferredType::MutableReference(Box::new(self.apply(inner)))
            }
            InferredType::Concrete(_) => ty.clone(),
        }
    }
    
    /// Compose two substitutions
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();
        
        // Apply other to all bindings in self
        for (var, ty) in &self.map {
            result.bind(var.clone(), other.apply(ty));
        }
        
        // Add bindings from other that aren't in self
        for (var, ty) in &other.map {
            if !self.map.contains_key(var) {
                result.bind(var.clone(), ty.clone());
            }
        }
        
        result
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

/// Type environment for inference
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Variable to type mapping
    bindings: HashMap<String, InferredType>,
    /// Next type variable ID
    next_var: usize,
}

impl TypeEnv {
    pub fn new() -> Self {
        TypeEnv {
            bindings: HashMap::new(),
            next_var: 0,
        }
    }
    
    /// Generate a fresh type variable
    pub fn fresh_var(&mut self) -> TypeVar {
        let var = TypeVar::new(self.next_var);
        self.next_var += 1;
        var
    }
    
    /// Bind a variable to a type
    pub fn bind(&mut self, name: String, ty: InferredType) {
        self.bindings.insert(name, ty);
    }
    
    /// Lookup a variable's type
    pub fn lookup(&self, name: &str) -> Option<&InferredType> {
        self.bindings.get(name)
    }
    
    /// Create a child environment
    pub fn child(&self) -> TypeEnv {
        TypeEnv {
            bindings: self.bindings.clone(),
            next_var: self.next_var,
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Unification algorithm
pub struct Unifier;

impl Unifier {
    /// Unify two types
    pub fn unify(t1: &InferredType, t2: &InferredType) -> Result<Substitution> {
        match (t1, t2) {
            // Same type variable
            (InferredType::Var(v1), InferredType::Var(v2)) if v1 == v2 => {
                Ok(Substitution::new())
            }
            
            // Bind type variable
            (InferredType::Var(var), ty) | (ty, InferredType::Var(var)) => {
                if ty.contains_var(var) {
                    Err(Error::new(
                        ErrorKind::TypeError,
                        "Occurs check failed: infinite type"
                    ))
                } else {
                    let mut subst = Substitution::new();
                    subst.bind(var.clone(), ty.clone());
                    Ok(subst)
                }
            }
            
            // Function types
            (InferredType::Function(params1, ret1), InferredType::Function(params2, ret2)) => {
                if params1.len() != params2.len() {
                    return Err(Error::new(
                        ErrorKind::TypeError,
                        format!("Function arity mismatch: {} vs {}", params1.len(), params2.len())
                    ));
                }
                
                let mut subst = Substitution::new();
                
                // Unify parameters
                for (p1, p2) in params1.iter().zip(params2.iter()) {
                    let s = Self::unify(&subst.apply(p1), &subst.apply(p2))?;
                    subst = subst.compose(&s);
                }
                
                // Unify return types
                let s = Self::unify(&subst.apply(ret1), &subst.apply(ret2))?;
                Ok(subst.compose(&s))
            }
            
            // Generic types
            (InferredType::Generic(name1, args1), InferredType::Generic(name2, args2)) => {
                if name1 != name2 {
                    return Err(Error::new(
                        ErrorKind::TypeError,
                        format!("Type mismatch: {} vs {}", name1, name2)
                    ));
                }
                
                if args1.len() != args2.len() {
                    return Err(Error::new(
                        ErrorKind::TypeError,
                        "Generic argument count mismatch"
                    ));
                }
                
                let mut subst = Substitution::new();
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    let s = Self::unify(&subst.apply(a1), &subst.apply(a2))?;
                    subst = subst.compose(&s);
                }
                
                Ok(subst)
            }
            
            // Tuple types
            (InferredType::Tuple(types1), InferredType::Tuple(types2)) => {
                if types1.len() != types2.len() {
                    return Err(Error::new(
                        ErrorKind::TypeError,
                        "Tuple size mismatch"
                    ));
                }
                
                let mut subst = Substitution::new();
                for (t1, t2) in types1.iter().zip(types2.iter()) {
                    let s = Self::unify(&subst.apply(t1), &subst.apply(t2))?;
                    subst = subst.compose(&s);
                }
                
                Ok(subst)
            }
            
            // Array types
            (InferredType::Array(inner1, size1), InferredType::Array(inner2, size2)) => {
                if size1 != size2 {
                    return Err(Error::new(
                        ErrorKind::TypeError,
                        "Array size mismatch"
                    ));
                }
                
                Self::unify(inner1, inner2)
            }
            
            // Reference types
            (InferredType::Reference(inner1), InferredType::Reference(inner2)) => {
                Self::unify(inner1, inner2)
            }
            
            (InferredType::MutableReference(inner1), InferredType::MutableReference(inner2)) => {
                Self::unify(inner1, inner2)
            }
            
            // Concrete types
            (InferredType::Concrete(Type::Named(n1)), InferredType::Concrete(Type::Named(n2))) => {
                if n1 == n2 {
                    Ok(Substitution::new())
                } else {
                    Err(Error::new(
                        ErrorKind::TypeError,
                        format!("Type mismatch: {} vs {}", n1, n2)
                    ))
                }
            }
            
            // Mismatch
            _ => Err(Error::new(
                ErrorKind::TypeError,
                format!("Cannot unify types: {:?} and {:?}", t1, t2)
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_var_creation() {
        let var1 = TypeVar::new(0);
        let var2 = TypeVar::new(1);
        assert_ne!(var1, var2);
    }

    #[test]
    fn test_substitution() {
        let mut subst = Substitution::new();
        let var = TypeVar::new(0);
        let ty = InferredType::Concrete(Type::Named("i32".to_string()));
        
        subst.bind(var.clone(), ty.clone());
        
        let result = subst.apply(&InferredType::Var(var));
        assert_eq!(result, ty);
    }

    #[test]
    fn test_unify_same_var() {
        let var = TypeVar::new(0);
        let t1 = InferredType::Var(var.clone());
        let t2 = InferredType::Var(var);
        
        let result = Unifier::unify(&t1, &t2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unify_var_with_concrete() {
        let var = TypeVar::new(0);
        let t1 = InferredType::Var(var);
        let t2 = InferredType::Concrete(Type::Named("i32".to_string()));
        
        let subst = Unifier::unify(&t1, &t2).unwrap();
        let result = subst.apply(&t1);
        assert_eq!(result, t2);
    }

    #[test]
    fn test_unify_concrete_types() {
        let t1 = InferredType::Concrete(Type::Named("i32".to_string()));
        let t2 = InferredType::Concrete(Type::Named("i32".to_string()));
        
        let result = Unifier::unify(&t1, &t2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unify_mismatch() {
        let t1 = InferredType::Concrete(Type::Named("i32".to_string()));
        let t2 = InferredType::Concrete(Type::Named("f64".to_string()));
        
        let result = Unifier::unify(&t1, &t2);
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_function_types() {
        let params1 = vec![InferredType::Concrete(Type::Named("i32".to_string()))];
        let ret1 = Box::new(InferredType::Concrete(Type::Named("i32".to_string())));
        let t1 = InferredType::Function(params1, ret1);
        
        let params2 = vec![InferredType::Concrete(Type::Named("i32".to_string()))];
        let ret2 = Box::new(InferredType::Concrete(Type::Named("i32".to_string())));
        let t2 = InferredType::Function(params2, ret2);
        
        let result = Unifier::unify(&t1, &t2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_occurs_check() {
        let var = TypeVar::new(0);
        let t1 = InferredType::Var(var.clone());
        let t2 = InferredType::Array(Box::new(InferredType::Var(var)), 10);
        
        let result = Unifier::unify(&t1, &t2);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_env() {
        let mut env = TypeEnv::new();
        let ty = InferredType::Concrete(Type::Named("i32".to_string()));
        
        env.bind("x".to_string(), ty.clone());
        assert_eq!(env.lookup("x"), Some(&ty));
        assert_eq!(env.lookup("y"), None);
    }

    #[test]
    fn test_fresh_var() {
        let mut env = TypeEnv::new();
        let var1 = env.fresh_var();
        let var2 = env.fresh_var();
        assert_ne!(var1, var2);
    }
}
