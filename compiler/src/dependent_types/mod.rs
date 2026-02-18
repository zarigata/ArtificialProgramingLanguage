//! Dependent Types Support (Experimental)
//!
//! This module provides support for type-level computation and dependent types.
//! Features include:
//! - Length-parameterized vectors
//! - Value-dependent types
//! - Compile-time bounds checking
//! - Type-level naturals

pub mod types;
pub mod checker;
pub mod inference;

pub use types::{DependentType, Nat, Vec};
pub use checker::DependentTypeChecker;
pub use inference::DependentTypeInference;
