//! Error handling for the VeZ compiler

use std::fmt;
use crate::span::Span;

/// Result type alias for compiler operations
pub type Result<T> = std::result::Result<T, Error>;

/// Compiler error types
#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub span: Option<Span>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    // Lexer errors
    InvalidCharacter,
    UnterminatedString,
    InvalidEscape,
    InvalidNumber,
    
    // Parser errors
    UnexpectedToken,
    ExpectedToken,
    InvalidSyntax,
    
    // Semantic errors
    UndefinedSymbol,
    TypeMismatch,
    TypeError,
    DuplicateSymbol,
    InvalidType,
    DuplicateDefinition,
    
    // Borrow checker errors
    BorrowError,
    MoveError,
    LifetimeError,
    
    // Other errors
    IoError,
    InternalError,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            span: None,
            message: message.into(),
        }
    }
    
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}
