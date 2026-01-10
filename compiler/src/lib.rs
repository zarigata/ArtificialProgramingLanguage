//! VeZ Compiler Library
//!
//! This is the main library for the VeZ programming language compiler.
//! It provides all the necessary components for compiling VeZ source code
//! to executable binaries.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod borrow;
pub mod ir;
pub mod optimizer;
pub mod codegen;
pub mod driver;
pub mod error;
pub mod span;
pub mod symbol;

/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Language file extension
pub const FILE_EXTENSION: &str = "zari";

/// Prelude module for common imports
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::span::{Span, Position};
    pub use crate::driver::Compiler;
}
