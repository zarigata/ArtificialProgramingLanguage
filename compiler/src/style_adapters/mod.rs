//! Style Adapters - Support for multiple programming language syntaxes
//! 
//! This module provides adapters that allow writing VeZ code using familiar
//! syntax from other programming languages (Python, JavaScript, Go, C++, Ruby).
//! All styles transpile to the unified VeZ AST.

pub mod python;
pub mod javascript;
pub mod go_style;
pub mod cpp_style;
pub mod converter;

use crate::error::Result;
use crate::parser::ast::Program;

/// Supported syntax styles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntaxStyle {
    /// Native VeZ syntax (Rust-like)
    Native,
    /// Python-style syntax
    Python,
    /// JavaScript/TypeScript-style syntax
    JavaScript,
    /// Go-style syntax
    Go,
    /// C++-style syntax
    Cpp,
    /// Ruby-style syntax
    Ruby,
}

impl SyntaxStyle {
    /// Parse from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "vez" | "zari" => Some(SyntaxStyle::Native),
            "pyvez" => Some(SyntaxStyle::Python),
            "jsvez" | "tsvez" => Some(SyntaxStyle::JavaScript),
            "gvez" => Some(SyntaxStyle::Go),
            "cppvez" => Some(SyntaxStyle::Cpp),
            "rbvez" => Some(SyntaxStyle::Ruby),
            _ => None,
        }
    }

    /// Get file extension for this style
    pub fn extension(&self) -> &'static str {
        match self {
            SyntaxStyle::Native => "vez",
            SyntaxStyle::Python => "pyvez",
            SyntaxStyle::JavaScript => "jsvez",
            SyntaxStyle::Go => "gvez",
            SyntaxStyle::Cpp => "cppvez",
            SyntaxStyle::Ruby => "rbvez",
        }
    }
}

/// Parse source code in a specific style and convert to VeZ AST
pub fn parse_with_style(source: &str, style: SyntaxStyle) -> Result<Program> {
    match style {
        SyntaxStyle::Native => {
            // Use standard VeZ parser
            let mut lexer = crate::lexer::Lexer::new(source);
            let tokens = lexer.tokenize()?;
            let mut parser = crate::parser::Parser::new(tokens);
            parser.parse()
        }
        SyntaxStyle::Python => python::parse(source),
        SyntaxStyle::JavaScript => javascript::parse(source),
        SyntaxStyle::Go => go_style::parse(source),
        SyntaxStyle::Cpp => cpp_style::parse(source),
        SyntaxStyle::Ruby => {
            // Ruby style not yet implemented
            Err(crate::error::Error::new(
                crate::error::ErrorKind::InvalidSyntax,
                "Ruby style not yet implemented"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_detection() {
        assert_eq!(SyntaxStyle::from_extension("vez"), Some(SyntaxStyle::Native));
        assert_eq!(SyntaxStyle::from_extension("pyvez"), Some(SyntaxStyle::Python));
        assert_eq!(SyntaxStyle::from_extension("jsvez"), Some(SyntaxStyle::JavaScript));
        assert_eq!(SyntaxStyle::from_extension("gvez"), Some(SyntaxStyle::Go));
        assert_eq!(SyntaxStyle::from_extension("unknown"), None);
    }

    #[test]
    fn test_style_extensions() {
        assert_eq!(SyntaxStyle::Native.extension(), "vez");
        assert_eq!(SyntaxStyle::Python.extension(), "pyvez");
        assert_eq!(SyntaxStyle::JavaScript.extension(), "jsvez");
    }
}
