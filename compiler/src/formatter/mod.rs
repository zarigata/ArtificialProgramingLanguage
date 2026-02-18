//! Code Formatter for VeZ (vezfmt)
//!
//! Provides automatic code formatting with configurable options.
//! Inspired by rustfmt and prettier.

pub mod config;
pub mod format;
pub mod indent;
pub mod comment;

pub use config::{FormatterConfig, IndentStyle, LineWidth};
pub use format::Formatter;

use std::path::{Path, PathBuf};
use std::io;

pub type FormatResult<T> = std::result::Result<T, FormatError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    IoError(String),
    ParseError(String),
    InvalidConfig(String),
    MaxWidthExceeded(String),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::IoError(msg) => write!(f, "IO error: {}", msg),
            FormatError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            FormatError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            FormatError::MaxWidthExceeded(msg) => write!(f, "Max width exceeded: {}", msg),
        }
    }
}

impl std::error::Error for FormatError {}

#[derive(Debug, Clone, Default)]
pub struct FormatStats {
    pub files_formatted: usize,
    pub files_unchanged: usize,
    pub lines_added: usize,
    pub lines_removed: usize,
}

impl FormatStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_files(&self) -> usize {
        self.files_formatted + self.files_unchanged
    }

    pub fn changed(&self) -> bool {
        self.files_formatted > 0
    }
}

pub fn format_source(source: &str, config: &FormatterConfig) -> FormatResult<String> {
    let mut formatter = Formatter::new(config);
    formatter.format(source)
}

pub fn format_file(path: &Path, config: &FormatterConfig) -> FormatResult<bool> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| FormatError::IoError(e.to_string()))?;
    
    let formatted = format_source(&source, config)?;
    
    if source == formatted {
        Ok(false)
    } else {
        std::fs::write(path, formatted)
            .map_err(|e| FormatError::IoError(e.to_string()))?;
        Ok(true)
    }
}

pub fn format_files(files: &[PathBuf], config: &FormatterConfig) -> FormatResult<FormatStats> {
    let mut stats = FormatStats::new();

    for path in files {
        match format_file(path, config) {
            Ok(true) => stats.files_formatted += 1,
            Ok(false) => stats.files_unchanged += 1,
            Err(e) => return Err(e),
        }
    }

    Ok(stats)
}

pub fn check_source(source: &str, config: &FormatterConfig) -> FormatResult<Vec<DiffLine>> {
    let formatted = format_source(source, config)?;
    
    let original_lines: Vec<&str> = source.lines().collect();
    let formatted_lines: Vec<&str> = formatted.lines().collect();
    
    let mut diff = Vec::new();
    let max_len = original_lines.len().max(formatted_lines.len());
    
    for i in 0..max_len {
        let orig = original_lines.get(i).copied();
        let fmt = formatted_lines.get(i).copied();
        
        if orig != fmt {
            diff.push(DiffLine {
                line_number: i + 1,
                original: orig.map(|s| s.to_string()),
                formatted: fmt.map(|s| s.to_string()),
            });
        }
    }
    
    Ok(diff)
}

#[derive(Debug, Clone)]
pub struct DiffLine {
    pub line_number: usize,
    pub original: Option<String>,
    pub formatted: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_stats() {
        let mut stats = FormatStats::new();
        stats.files_formatted = 5;
        stats.files_unchanged = 3;
        
        assert_eq!(stats.total_files(), 8);
        assert!(stats.changed());
    }

    #[test]
    fn test_format_source_simple() {
        let config = FormatterConfig::default();
        let source = "def add(a:int,b:int)->int:return a+b";
        let result = format_source(source, &config);
        
        assert!(result.is_ok());
        let formatted = result.unwrap();
        assert!(formatted.contains("def add(a: int, b: int) -> int:"));
    }
}
