//! Diagnostic messages for the linter

use std::path::PathBuf;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Suggestion,
    Warning,
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Suggestion => write!(f, "suggestion"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
        }
    }
}

impl Severity {
    pub fn color(&self) -> &'static str {
        match self {
            Severity::Info => "\x1b[34m",
            Severity::Suggestion => "\x1b[36m",
            Severity::Warning => "\x1b[33m",
            Severity::Error => "\x1b[31m",
        }
    }

    pub fn reset(&self) -> &'static str {
        "\x1b[0m"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

impl Position {
    pub fn new(line: usize, column: usize) -> Self {
        Position { line, column }
    }

    pub fn start() -> Self {
        Position { line: 1, column: 1 }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::start()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

impl Span {
    pub fn new(start: Position, end: Position) -> Self {
        Span { start, end }
    }

    pub fn at(line: usize, column: usize) -> Self {
        Span {
            start: Position::new(line, column),
            end: Position::new(line, column + 1),
        }
    }

    pub fn range(start_line: usize, start_col: usize, end_line: usize, end_col: usize) -> Self {
        Span {
            start: Position::new(start_line, start_col),
            end: Position::new(end_line, end_col),
        }
    }

    pub fn unknown() -> Self {
        Span {
            start: Position::start(),
            end: Position::start(),
        }
    }
}

impl Default for Span {
    fn default() -> Self {
        Self::unknown()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticKind {
    UnusedVariable,
    UnusedImport,
    UnusedParameter,
    UnreachableCode,
    DeadCode,
    DeprecatedSyntax,
    NamingConvention,
    MissingDoc,
    ComplexFunction,
    MagicNumber,
    TodoComment,
    UnnecessaryMut,
    InefficientLoop,
    PotentialPanic,
    TypeComplexity,
    CognitiveComplexity,
    StyleViolation,
    Security,
    Performance,
    Custom(String),
}

impl fmt::Display for DiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticKind::UnusedVariable => write!(f, "unused-variable"),
            DiagnosticKind::UnusedImport => write!(f, "unused-import"),
            DiagnosticKind::UnusedParameter => write!(f, "unused-parameter"),
            DiagnosticKind::UnreachableCode => write!(f, "unreachable-code"),
            DiagnosticKind::DeadCode => write!(f, "dead-code"),
            DiagnosticKind::DeprecatedSyntax => write!(f, "deprecated-syntax"),
            DiagnosticKind::NamingConvention => write!(f, "naming-convention"),
            DiagnosticKind::MissingDoc => write!(f, "missing-doc"),
            DiagnosticKind::ComplexFunction => write!(f, "complex-function"),
            DiagnosticKind::MagicNumber => write!(f, "magic-number"),
            DiagnosticKind::TodoComment => write!(f, "todo-comment"),
            DiagnosticKind::UnnecessaryMut => write!(f, "unnecessary-mut"),
            DiagnosticKind::InefficientLoop => write!(f, "inefficient-loop"),
            DiagnosticKind::PotentialPanic => write!(f, "potential-panic"),
            DiagnosticKind::TypeComplexity => write!(f, "type-complexity"),
            DiagnosticKind::CognitiveComplexity => write!(f, "cognitive-complexity"),
            DiagnosticKind::StyleViolation => write!(f, "style-violation"),
            DiagnosticKind::Security => write!(f, "security"),
            DiagnosticKind::Performance => write!(f, "performance"),
            DiagnosticKind::Custom(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub kind: DiagnosticKind,
    pub severity: Severity,
    pub message: String,
    pub span: Span,
    pub file: Option<PathBuf>,
    pub fix: Option<Fix>,
    pub hints: Vec<String>,
    pub related: Vec<RelatedInfo>,
}

impl Diagnostic {
    pub fn new(kind: DiagnosticKind, severity: Severity, message: &str, span: Span) -> Self {
        Diagnostic {
            kind,
            severity,
            message: message.to_string(),
            span,
            file: None,
            fix: None,
            hints: Vec::new(),
            related: Vec::new(),
        }
    }

    pub fn error(kind: DiagnosticKind, message: &str, span: Span) -> Self {
        Self::new(kind, Severity::Error, message, span)
    }

    pub fn warning(kind: DiagnosticKind, message: &str, span: Span) -> Self {
        Self::new(kind, Severity::Warning, message, span)
    }

    pub fn suggestion(kind: DiagnosticKind, message: &str, span: Span) -> Self {
        Self::new(kind, Severity::Suggestion, message, span)
    }

    pub fn info(kind: DiagnosticKind, message: &str, span: Span) -> Self {
        Self::new(kind, Severity::Info, message, span)
    }

    pub fn with_file(mut self, file: PathBuf) -> Self {
        self.file = Some(file);
        self
    }

    pub fn with_fix(mut self, fix: Fix) -> Self {
        self.fix = Some(fix);
        self
    }

    pub fn with_hint(mut self, hint: &str) -> Self {
        self.hints.push(hint.to_string());
        self
    }

    pub fn with_related(mut self, span: Span, message: &str) -> Self {
        self.related.push(RelatedInfo {
            span,
            message: message.to_string(),
        });
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = self.file.as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<stdin>".to_string());
        
        write!(
            f,
            "{}{}{}: {} [{}]\n  --> {}:{}:{}",
            self.severity.color(),
            self.severity,
            self.severity.reset(),
            self.message,
            self.kind,
            file,
            self.span.start.line,
            self.span.start.column
        )?;

        for hint in &self.hints {
            write!(f, "\n  = hint: {}", hint)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RelatedInfo {
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Fix {
    pub span: Span,
    pub replacement: String,
    pub message: String,
}

impl Fix {
    pub fn new(span: Span, replacement: &str, message: &str) -> Self {
        Fix {
            span,
            replacement: replacement.to_string(),
            message: message.to_string(),
        }
    }

    pub fn replace(span: Span, replacement: &str) -> Self {
        Self::new(span, replacement, "Replace with")
    }

    pub fn remove(span: Span) -> Self {
        Self::new(span, "", "Remove")
    }

    pub fn insert_before(span: Span, text: &str) -> Self {
        Self::new(span, text, "Insert before")
    }

    pub fn insert_after(span: Span, text: &str) -> Self {
        Self::new(span, text, "Insert after")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Suggestion);
        assert!(Severity::Suggestion > Severity::Info);
    }

    #[test]
    fn test_position() {
        let pos = Position::new(10, 5);
        assert_eq!(pos.line, 10);
        assert_eq!(pos.column, 5);
        assert_eq!(format!("{}", pos), "10:5");
    }

    #[test]
    fn test_diagnostic() {
        let diag = Diagnostic::warning(
            DiagnosticKind::UnusedVariable,
            "variable `x` is never used",
            Span::at(5, 10),
        );

        assert_eq!(diag.severity, Severity::Warning);
        assert!(diag.message.contains("never used"));
    }

    #[test]
    fn test_fix() {
        let fix = Fix::replace(Span::at(1, 1), "new_value");
        assert_eq!(fix.replacement, "new_value");
    }
}
