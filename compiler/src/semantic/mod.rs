//! Semantic analysis

pub mod symbol_table;
pub mod resolver;
pub mod type_env;
pub mod type_checker;
pub mod types;

pub use symbol_table::{SymbolTable, Symbol, SymbolKind, ScopeId};
pub use resolver::Resolver;
pub use type_env::{TypeEnv, InferredType, Substitution, Unifier, TypeVar};
pub use type_checker::TypeChecker;