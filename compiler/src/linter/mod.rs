//! Linter for VeZ (vezlint)
//!
//! Provides static analysis and code quality checks.

pub mod rules;
pub mod config;
pub mod diagnostic;
pub mod fixer;

pub use rules::{LintRule, LintRegistry};
pub use config::{LinterConfig, LintLevel};
pub use diagnostic::{Diagnostic, DiagnosticKind, Severity, Fix, Span, Position};
pub use fixer::Fixer;

use std::path::{Path, PathBuf};
use std::collections::HashMap;

pub type LintResult<T> = std::result::Result<T, LintError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LintError {
    IoError(String),
    ParseError(String),
    InvalidConfig(String),
    RuleNotFound(String),
}

impl std::fmt::Display for LintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LintError::IoError(msg) => write!(f, "IO error: {}", msg),
            LintError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LintError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            LintError::RuleNotFound(msg) => write!(f, "Rule not found: {}", msg),
        }
    }
}

impl std::error::Error for LintError {}

#[derive(Debug, Clone, Default)]
pub struct LintStats {
    pub files_checked: usize,
    pub errors: usize,
    pub warnings: usize,
    pub suggestions: usize,
    pub fixes_applied: usize,
}

impl LintStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_issues(&self) -> usize {
        self.errors + self.warnings + self.suggestions
    }

    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }

    pub fn clean(&self) -> bool {
        self.total_issues() == 0
    }
}

pub struct Linter {
    config: LinterConfig,
    registry: LintRegistry,
}

impl Linter {
    pub fn new() -> Self {
        Linter {
            config: LinterConfig::default(),
            registry: LintRegistry::with_default_rules(),
        }
    }

    pub fn with_config(config: LinterConfig) -> Self {
        Linter {
            config,
            registry: LintRegistry::with_default_rules(),
        }
    }

    pub fn add_rule(&mut self, rule: Box<dyn LintRule>) {
        self.registry.register(rule);
    }

    pub fn check_source(&self, source: &str, path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for rule in self.registry.rules() {
            if self.config.is_enabled(rule.name()) {
                let mut rule_diagnostics = rule.check(source, path);
                
                for diag in &mut rule_diagnostics {
                    diag.severity = self.config.get_severity(rule.name()).unwrap_or(diag.severity);
                }
                
                diagnostics.extend(rule_diagnostics);
            }
        }

        diagnostics.sort_by(|a, b| {
            a.span.start.line.cmp(&b.span.start.line)
                .then_with(|| a.span.start.column.cmp(&b.span.start.column))
        });

        diagnostics
    }

    pub fn check_file(&self, path: &Path) -> LintResult<Vec<Diagnostic>> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| LintError::IoError(e.to_string()))?;
        
        Ok(self.check_source(&source, path))
    }

    pub fn check_files(&self, files: &[PathBuf]) -> LintResult<(LintStats, HashMap<PathBuf, Vec<Diagnostic>>)> {
        let mut stats = LintStats::new();
        let mut all_diagnostics = HashMap::new();

        for path in files {
            let diagnostics = self.check_file(path)?;
            
            for diag in &diagnostics {
                match diag.severity {
                    Severity::Error => stats.errors += 1,
                    Severity::Warning => stats.warnings += 1,
                    Severity::Suggestion => stats.suggestions += 1,
                    Severity::Info => {}
                }
            }
            
            stats.files_checked += 1;
            all_diagnostics.insert(path.clone(), diagnostics);
        }

        Ok((stats, all_diagnostics))
    }

    pub fn fix_source(&self, source: &str, path: &Path) -> (String, Vec<Fix>) {
        let diagnostics = self.check_source(source, path);
        let mut fixer = Fixer::new(source);
        let mut applied_fixes = Vec::new();

        for diag in &diagnostics {
            if let Some(fix) = &diag.fix {
                if fixer.apply(fix) {
                    applied_fixes.push(fix.clone());
                }
            }
        }

        (fixer.finish(), applied_fixes)
    }

    pub fn fix_file(&self, path: &Path) -> LintResult<Vec<Fix>> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| LintError::IoError(e.to_string()))?;
        
        let (fixed, fixes) = self.fix_source(&source, path);
        
        if !fixes.is_empty() {
            std::fs::write(path, fixed)
                .map_err(|e| LintError::IoError(e.to_string()))?;
        }
        
        Ok(fixes)
    }
}

impl Default for Linter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_stats() {
        let mut stats = LintStats::new();
        stats.errors = 2;
        stats.warnings = 5;
        
        assert_eq!(stats.total_issues(), 7);
        assert!(stats.has_errors());
        assert!(!stats.clean());
    }

    #[test]
    fn test_linter_creation() {
        let linter = Linter::new();
        assert!(!linter.registry.rules().is_empty());
    }
}
