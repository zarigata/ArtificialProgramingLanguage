//! Optimization passes

pub mod constant_folding;
pub mod dead_code;
pub mod common_subexpr;
pub mod inline;
pub mod pass_manager;

pub use constant_folding::ConstantFolding;
pub use dead_code::DeadCodeElimination;
pub use common_subexpr::CommonSubexprElimination;
pub use inline::InlineExpansion;
pub use pass_manager::{OptimizationPass, PassManager};
