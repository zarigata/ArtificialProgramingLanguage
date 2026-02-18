//! Error handling for the VeZ compiler

use std::fmt;
use crate::span::Span;

/// Result type alias for compiler operations
pub type Result<T> = std::result::Result<T, Error>;

/// Compiler error with rich context and suggestions
#[derive(Debug, Clone)]
pub struct Error {
    pub code: ErrorCode,
    pub kind: ErrorKind,
    pub span: Option<Span>,
    pub message: String,
    pub suggestion: Option<Suggestion>,
    pub notes: Vec<String>,
    pub help: Option<String>,
}

/// Error codes for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorCode(pub u32);

impl ErrorCode {
    pub const E0001: ErrorCode = ErrorCode(1);   // Undefined variable
    pub const E0002: ErrorCode = ErrorCode(2);   // Type mismatch
    pub const E0003: ErrorCode = ErrorCode(3);   // Undefined function
    pub const E0004: ErrorCode = ErrorCode(4);   // Invalid syntax
    pub const E0005: ErrorCode = ErrorCode(5);   // Borrow error
    pub const E0006: ErrorCode = ErrorCode(6);   // Move error
    pub const E0007: ErrorCode = ErrorCode(7);   // Lifetime error
    pub const E0008: ErrorCode = ErrorCode(8);   // Duplicate definition
    pub const E0009: ErrorCode = ErrorCode(9);   // Invalid character
    pub const E0010: ErrorCode = ErrorCode(10);  // Unterminated string
    pub const E0011: ErrorCode = ErrorCode(11);  // Invalid escape
    pub const E0012: ErrorCode = ErrorCode(12);  // Invalid number
    pub const E0013: ErrorCode = ErrorCode(13);  // Unexpected token
    pub const E0014: ErrorCode = ErrorCode(14);  // Expected token
    pub const E0015: ErrorCode = ErrorCode(15);  // Invalid type
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{:04}", self.0)
    }
}

/// Suggestion for fixing an error
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub message: String,
    pub replacement: Option<String>,
    pub span: Option<Span>,
}

impl Suggestion {
    pub fn new(message: impl Into<String>) -> Self {
        Suggestion {
            message: message.into(),
            replacement: None,
            span: None,
        }
    }
    
    pub fn with_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.replacement = Some(replacement.into());
        self
    }
    
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
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
    UndefinedVariable,
    UndefinedFunction,
    TypeMismatch,
    TypeError,
    DuplicateSymbol,
    DuplicateDefinition,
    InvalidType,
    
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
            code: Error::code_for_kind(&kind),
            kind,
            span: None,
            message: message.into(),
            suggestion: None,
            notes: Vec::new(),
            help: None,
        }
    }
    
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
    
    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestion = Some(suggestion);
        self
    }
    
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
    
    pub fn add_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
    
    fn code_for_kind(kind: &ErrorKind) -> ErrorCode {
        match kind {
            ErrorKind::UndefinedSymbol | ErrorKind::UndefinedVariable => ErrorCode::E0001,
            ErrorKind::TypeMismatch => ErrorCode::E0002,
            ErrorKind::UndefinedFunction => ErrorCode::E0003,
            ErrorKind::InvalidSyntax => ErrorCode::E0004,
            ErrorKind::BorrowError => ErrorCode::E0005,
            ErrorKind::MoveError => ErrorCode::E0006,
            ErrorKind::LifetimeError => ErrorCode::E0007,
            ErrorKind::DuplicateSymbol | ErrorKind::DuplicateDefinition => ErrorCode::E0008,
            ErrorKind::InvalidCharacter => ErrorCode::E0009,
            ErrorKind::UnterminatedString => ErrorCode::E0010,
            ErrorKind::InvalidEscape => ErrorCode::E0011,
            ErrorKind::InvalidNumber => ErrorCode::E0012,
            ErrorKind::UnexpectedToken => ErrorCode::E0013,
            ErrorKind::ExpectedToken => ErrorCode::E0014,
            ErrorKind::InvalidType => ErrorCode::E0015,
            _ => ErrorCode::E0001,
        }
    }
    
    pub fn undefined_variable(name: &str, similar: Option<&str>) -> Self {
        let mut err = Error::new(
            ErrorKind::UndefinedVariable,
            format!("cannot find value `{}` in this scope", name)
        );
        
        if let Some(similar) = similar {
            err = err.with_suggestion(
                Suggestion::new(format!("did you mean `{}`?", similar))
                    .with_replacement(similar)
            );
        } else {
            err = err.add_note("variables must be declared before use");
        }
        
        err
    }
    
    pub fn undefined_function(name: &str, similar: Option<&str>) -> Self {
        let mut err = Error::new(
            ErrorKind::UndefinedFunction,
            format!("cannot find function `{}` in this scope", name)
        );
        
        if let Some(similar) = similar {
            err = err.with_suggestion(
                Suggestion::new(format!("did you mean `{}`?", similar))
            );
        }
        
        err = err.add_note("functions must be defined or imported before use");
        err
    }
    
    pub fn type_mismatch(expected: &str, found: &str) -> Self {
        Error::new(
            ErrorKind::TypeMismatch,
            format!("expected `{}`, found `{}`", expected, found)
        )
        .with_help(format!("try converting the value to `{}`", expected))
    }
    
    pub fn borrow_error(borrow_kind: &str, reason: &str) -> Self {
        Error::new(
            ErrorKind::BorrowError,
            format!("cannot borrow as {}", borrow_kind)
        )
        .add_note(reason)
    }
    
    pub fn move_error(name: &str, reason: &str) -> Self {
        Error::new(
            ErrorKind::MoveError,
            format!("use of moved value: `{}`", name)
        )
        .add_note(reason)
        .with_help("consider cloning the value instead")
    }
    
    pub fn duplicate_definition(name: &str, kind: &str) -> Self {
        Error::new(
            ErrorKind::DuplicateDefinition,
            format!("the {} `{}` is defined multiple times", kind, name)
        )
        .add_note("must be defined only once per scope")
    }
    
    pub fn unexpected_token(found: &str, expected: &[&str]) -> Self {
        let expected_str = if expected.len() == 1 {
            expected[0].to_string()
        } else {
            format!("one of: {}", expected.join(", "))
        };
        
        Error::new(
            ErrorKind::UnexpectedToken,
            format!("unexpected token `{}`, expected {}", found, expected_str)
        )
    }
    
    pub fn invalid_escape(seq: &str) -> Self {
        Error::new(
            ErrorKind::InvalidEscape,
            format!("invalid escape sequence: `{}`", seq)
        )
        .with_help("valid escape sequences are: \\n, \\t, \\r, \\\\, \\\", \\0")
    }
    
    pub fn unterminated_string() -> Self {
        Error::new(
            ErrorKind::UnterminatedString,
            "unterminated string literal"
        )
        .with_help("add a closing \" to terminate the string")
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]: {}", self.code, self.message)?;
        
        if let Some(span) = &self.span {
            write!(f, "\n  --> {}", span)?;
        }
        
        if let Some(suggestion) = &self.suggestion {
            write!(f, "\n  = help: {}", suggestion.message)?;
            if let Some(repl) = &suggestion.replacement {
                write!(f, "\n         replace with: `{}`", repl)?;
            }
        }
        
        for note in &self.notes {
            write!(f, "\n  = note: {}", note)?;
        }
        
        if let Some(help) = &self.help {
            write!(f, "\n  = help: {}", help)?;
        }
        
        Ok(())
    }
}

impl std::error::Error for Error {}

/// Find similar identifiers for suggestions
pub fn find_similar(name: &str, candidates: &[&str]) -> Option<String> {
    let mut best_match: Option<(String, usize)> = None;
    let name_lower = name.to_lowercase();
    
    for candidate in candidates {
        let candidate_lower = candidate.to_lowercase();
        
        if candidate_lower == name_lower {
            return Some(candidate.to_string());
        }
        
        let dist = levenshtein_distance(&name_lower, &candidate_lower);
        let threshold = (name.len() / 3).max(1);
        
        if dist <= threshold {
            if let Some((_, best_dist)) = &best_match {
                if dist < *best_dist {
                    best_match = Some((candidate.to_string(), dist));
                }
            } else {
                best_match = Some((candidate.to_string(), dist));
            }
        }
    }
    
    best_match.map(|(s, _)| s)
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }
    
    let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];
    
    for i in 0..=a_len {
        matrix[i][0] = i;
    }
    for j in 0..=b_len {
        matrix[0][j] = j;
    }
    
    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }
    
    matrix[a_len][b_len]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = Error::undefined_variable("x", Some("y"));
        let output = format!("{}", err);
        assert!(output.contains("cannot find value `x`"));
        assert!(output.contains("did you mean `y`?"));
    }
    
    #[test]
    fn test_find_similar() {
        let candidates = vec!["hello", "world", "help", "held"];
        assert_eq!(find_similar("helo", &candidates), Some("hello".to_string()));
        assert_eq!(find_similar("worl", &candidates), Some("world".to_string()));
    }
    
    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
    }
}
