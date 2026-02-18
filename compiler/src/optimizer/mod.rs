//! Optimization passes

pub mod constant_folding;
pub mod dead_code;
pub mod common_subexpr;
pub mod inline;
pub mod pass_manager;
pub mod loop_unroll;
pub mod vectorizer;
pub mod devirtualizer;
pub mod escape_analysis;

pub use constant_folding::ConstantFolding;
pub use dead_code::DeadCodeElimination;
pub use common_subexpr::CommonSubexprElimination;
pub use inline::InlineExpansion;
pub use pass_manager::{OptimizationPass, PassManager, OptLevel};
pub use loop_unroll::{LoopUnroller, LoopStrengthReduction};
pub use vectorizer::{Vectorizer, SlpVectorizer, VectorWidth, SimdTarget};
pub use devirtualizer::{Devirtualizer, SpeculativeDevirtualizer};
pub use escape_analysis::{EscapeAnalyzer, StackPromotion};
