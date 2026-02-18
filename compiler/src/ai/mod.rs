//! AI Integration Module
//!
//! Provides tools for extracting semantic context to enable AI code generation.

pub mod context;
pub mod prompts;
pub mod patterns;

pub use context::{AiContext, extract_context, FunctionSignature, TypeDefinition};
pub use prompts::{PromptTemplate, PromptLibrary};
pub use patterns::{CodePattern, PatternMatcher};
