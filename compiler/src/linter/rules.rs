//! Lint rules for the linter

use std::path::Path;
use std::collections::HashMap;
use super::diagnostic::{Diagnostic, DiagnosticKind, Severity, Span, Position, Fix};
use super::config::LintLevel;

pub trait LintRule: std::fmt::Debug {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn check(&self, source: &str, path: &Path) -> Vec<Diagnostic>;
}

#[derive(Debug, Default)]
pub struct LintRegistry {
    rules: HashMap<String, Box<dyn LintRule>>,
}

impl LintRegistry {
    pub fn new() -> Self {
        LintRegistry {
            rules: HashMap::new(),
        }
    }

    pub fn with_default_rules() -> Self {
        let mut registry = Self::new();
        
        registry.register(Box::new(UnusedVariableRule));
        registry.register(Box::new(UnusedImportRule));
        registry.register(Box::new(UnusedParameterRule));
        registry.register(Box::new(UnreachableCodeRule));
        registry.register(Box::new(DeadCodeRule));
        registry.register(Box::new(NamingConventionRule));
        registry.register(Box::new(MagicNumberRule));
        registry.register(Box::new(TodoCommentRule));
        registry.register(Box::new(ComplexFunctionRule));
        registry.register(Box::new(LongLineRule));
        
        registry
    }

    pub fn register(&mut self, rule: Box<dyn LintRule>) {
        self.rules.insert(rule.name().to_string(), rule);
    }

    pub fn get(&self, name: &str) -> Option<&dyn LintRule> {
        self.rules.get(name).map(|r| r.as_ref())
    }

    pub fn rules(&self) -> impl Iterator<Item = &dyn LintRule> {
        self.rules.values().map(|r| r.as_ref())
    }

    pub fn rule_names(&self) -> impl Iterator<Item = &str> {
        self.rules.keys().map(|s| s.as_str())
    }
}

#[derive(Debug)]
pub struct UnusedVariableRule;

impl LintRule for UnusedVariableRule {
    fn name(&self) -> &str {
        "unused-variable"
    }

    fn description(&self) -> &str {
        "Detects variables that are declared but never used"
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut declared_vars: HashMap<String, (usize, usize)> = HashMap::new();
        let mut used_vars: HashMap<String, bool> = HashMap::new();

        for (line_num, line) in source.lines().enumerate() {
            let line = line.trim();
            
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if line.starts_with("let ") || line.starts_with("var ") || line.starts_with("const ") {
                let rest = line.split_whitespace().skip(1).collect::<Vec<_>>();
                if let Some(var_part) = rest.first() {
                    let var_name = var_part.split(':').next().unwrap_or("").split('=').next().unwrap_or("");
                    let var_name = var_name.trim().trim_end_matches(':');
                    if !var_name.starts_with('_') && !var_name.is_empty() {
                        let col = line.find(var_part).unwrap_or(0);
                        declared_vars.insert(var_name.to_string(), (line_num + 1, col + 1));
                    }
                }
            }

            for (var_name, _) in &declared_vars {
                if line.contains(var_name) && !line.starts_with("let ") && !line.starts_with("var ") && !line.starts_with("const ") {
                    *used_vars.entry(var_name.clone()).or_insert(false) = true;
                }
            }
        }

        for (var_name, (line, col)) in &declared_vars {
            if !used_vars.contains_key(var_name) {
                let mut diag = Diagnostic::warning(
                    DiagnosticKind::UnusedVariable,
                    &format!("variable `{}` is declared but never used", var_name),
                    Span::at(*line, *col),
                );
                
                diag = diag.with_fix(Fix::new(
                    Span::at(*line, *col),
                    &format!("_{}", var_name),
                    "Prefix with underscore to mark as intentionally unused",
                ));
                
                diagnostics.push(diag);
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct UnusedImportRule;

impl LintRule for UnusedImportRule {
    fn name(&self) -> &str {
        "unused-import"
    }

    fn description(&self) -> &str {
        "Detects imports that are never used"
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut imports: HashMap<String, (usize, usize)> = HashMap::new();
        let mut used_items: HashMap<String, bool> = HashMap::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                let import_name = if trimmed.starts_with("import ") {
                    trimmed.strip_prefix("import ").unwrap_or("")
                } else {
                    trimmed.split("import").nth(1).unwrap_or("")
                };
                
                let import_name = import_name.split_whitespace().next().unwrap_or("");
                if !import_name.is_empty() {
                    let col = line.find(import_name).unwrap_or(0);
                    imports.insert(import_name.to_string(), (line_num + 1, col + 1));
                }
            }

            if !trimmed.starts_with("import ") && !trimmed.starts_with("from ") {
                for (import_name, _) in &imports {
                    if line.contains(import_name) {
                        *used_items.entry(import_name.clone()).or_insert(false) = true;
                    }
                }
            }
        }

        for (import_name, (line, col)) in &imports {
            if !used_items.contains_key(import_name) {
                diagnostics.push(Diagnostic::warning(
                    DiagnosticKind::UnusedImport,
                    &format!("import `{}` is never used", import_name),
                    Span::at(*line, *col),
                ));
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct UnusedParameterRule;

impl LintRule for UnusedParameterRule {
    fn name(&self) -> &str {
        "unused-parameter"
    }

    fn description(&self) -> &str {
        "Detects function parameters that are never used"
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Allow
    }

    fn check(&self, _source: &str, _path: &Path) -> Vec<Diagnostic> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct UnreachableCodeRule;

impl LintRule for UnreachableCodeRule {
    fn name(&self) -> &str {
        "unreachable-code"
    }

    fn description(&self) -> &str {
        "Detects code that can never be executed"
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut after_return = false;

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            
            if trimmed.starts_with('#') || trimmed.is_empty() {
                continue;
            }

            if after_return && !trimmed.starts_with('}') && !trimmed.is_empty() {
                diagnostics.push(Diagnostic::warning(
                    DiagnosticKind::UnreachableCode,
                    "unreachable code detected",
                    Span::at(line_num + 1, 1),
                ));
            }

            if trimmed.starts_with("return ") || trimmed == "return" {
                after_return = true;
            }

            if trimmed.contains("}") && after_return {
                after_return = false;
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct DeadCodeRule;

impl LintRule for DeadCodeRule {
    fn name(&self) -> &str {
        "dead-code"
    }

    fn description(&self) -> &str {
        "Detects functions that are never called"
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut defined_fns: HashMap<String, (usize, usize)> = HashMap::new();
        let mut called_fns: HashMap<String, bool> = HashMap::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("def ") || trimmed.starts_with("fn ") {
                let rest = if trimmed.starts_with("def ") {
                    trimmed.strip_prefix("def ").unwrap_or("")
                } else {
                    trimmed.strip_prefix("fn ").unwrap_or("")
                };
                
                let fn_name = rest.split('(').next().unwrap_or("");
                let fn_name = fn_name.trim();
                if !fn_name.is_empty() && !fn_name.starts_with('_') {
                    let col = line.find(fn_name).unwrap_or(0);
                    defined_fns.insert(fn_name.to_string(), (line_num + 1, col + 1));
                }
            }

            if !trimmed.starts_with("def ") && !trimmed.starts_with("fn ") {
                for (fn_name, _) in &defined_fns {
                    if line.contains(&format!("{}(", fn_name)) || line.contains(&format!("{} (", fn_name)) {
                        *called_fns.entry(fn_name.clone()).or_insert(false) = true;
                    }
                }
            }
        }

        for (fn_name, (line, col)) in &defined_fns {
            if !called_fns.contains_key(fn_name) {
                diagnostics.push(Diagnostic::warning(
                    DiagnosticKind::DeadCode,
                    &format!("function `{}` is never called", fn_name),
                    Span::at(*line, *col),
                ).with_hint("Consider making the function public or removing it"));
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct NamingConventionRule;

impl LintRule for NamingConventionRule {
    fn name(&self) -> &str {
        "naming-convention"
    }

    fn description(&self) -> &str {
        "Enforces naming conventions"
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Allow
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("def ") || trimmed.starts_with("fn ") {
                let rest = if trimmed.starts_with("def ") {
                    trimmed.strip_prefix("def ").unwrap_or("")
                } else {
                    trimmed.strip_prefix("fn ").unwrap_or("")
                };
                
                let fn_name = rest.split('(').next().unwrap_or("");
                let fn_name = fn_name.trim();
                
                if !fn_name.is_empty() && !fn_name.contains('_') && fn_name.chars().any(|c| c.is_uppercase()) {
                    let col = line.find(fn_name).unwrap_or(0);
                    diagnostics.push(Diagnostic::suggestion(
                        DiagnosticKind::NamingConvention,
                        &format!("function `{}` should use snake_case", fn_name),
                        Span::at(line_num + 1, col + 1),
                    ));
                }
            }

            if trimmed.starts_with("struct ") || trimmed.starts_with("class ") {
                let rest = if trimmed.starts_with("struct ") {
                    trimmed.strip_prefix("struct ").unwrap_or("")
                } else {
                    trimmed.strip_prefix("class ").unwrap_or("")
                };
                
                let type_name = rest.split('{').next().unwrap_or("").split(':').next().unwrap_or("");
                let type_name = type_name.trim();
                
                if !type_name.is_empty() && type_name.chars().next().map_or(true, |c| c.is_lowercase()) {
                    let col = line.find(type_name).unwrap_or(0);
                    diagnostics.push(Diagnostic::suggestion(
                        DiagnosticKind::NamingConvention,
                        &format!("type `{}` should use PascalCase", type_name),
                        Span::at(line_num + 1, col + 1),
                    ));
                }
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct MagicNumberRule;

impl LintRule for MagicNumberRule {
    fn name(&self) -> &str {
        "magic-number"
    }

    fn description(&self) -> &str {
        "Detects magic numbers that should be constants"
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Allow
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            
            if trimmed.starts_with('#') || trimmed.starts_with("const ") || trimmed.starts_with("def ") || trimmed.starts_with("fn ") {
                continue;
            }

            let numbers: Vec<_> = trimmed.match_indices(|c: char| c.is_ascii_digit())
                .collect();
            
            for (idx, _) in numbers {
                if idx > 0 {
                    let before = trimmed.chars().nth(idx - 1);
                    if before == Some('_') || before.map_or(false, |c| c.is_alphanumeric()) {
                        continue;
                    }
                }

                let num_str: String = trimmed[idx..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                
                if let Ok(n) = num_str.parse::<f64>() {
                    if n != 0.0 && n != 1.0 && n != -1.0 && n.abs() > 1e-10 {
                        diagnostics.push(Diagnostic::suggestion(
                            DiagnosticKind::MagicNumber,
                            &format!("magic number `{}` should be a named constant", num_str),
                            Span::at(line_num + 1, idx + 1),
                        ).with_hint("Extract to a named constant for better readability"));
                    }
                }
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct TodoCommentRule;

impl LintRule for TodoCommentRule {
    fn name(&self) -> &str {
        "todo-comment"
    }

    fn description(&self) -> &str {
        "Detects TODO comments in the code"
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Allow
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let lower = line.to_lowercase();
            
            if lower.contains("todo") || lower.contains("fixme") || lower.contains("hack") || lower.contains("xxx") {
                let col = lower.find("todo")
                    .or_else(|| lower.find("fixme"))
                    .or_else(|| lower.find("hack"))
                    .or_else(|| lower.find("xxx"))
                    .unwrap_or(0);
                
                let keyword = if lower.contains("todo") { "TODO" }
                    else if lower.contains("fixme") { "FIXME" }
                    else if lower.contains("hack") { "HACK" }
                    else { "XXX" };
                
                diagnostics.push(Diagnostic::info(
                    DiagnosticKind::TodoComment,
                    &format!("{} comment found", keyword),
                    Span::at(line_num + 1, col + 1),
                ).with_hint("Consider creating an issue to track this"));
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct ComplexFunctionRule;

impl LintRule for ComplexFunctionRule {
    fn name(&self) -> &str {
        "complex-function"
    }

    fn description(&self) -> &str {
        "Warns about functions that are too complex"
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut current_fn: Option<(String, usize, usize)> = None;
        let mut complexity = 0;

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with("def ") || trimmed.starts_with("fn ") {
                if let Some((fn_name, start_line, _)) = current_fn.take() {
                    if complexity > 15 {
                        diagnostics.push(Diagnostic::warning(
                            DiagnosticKind::ComplexFunction,
                            &format!("function `{}` is too complex (complexity: {})", fn_name, complexity),
                            Span::at(start_line, 1),
                        ).with_hint("Consider breaking this function into smaller functions"));
                    }
                }

                let rest = if trimmed.starts_with("def ") {
                    trimmed.strip_prefix("def ").unwrap_or("")
                } else {
                    trimmed.strip_prefix("fn ").unwrap_or("")
                };
                let fn_name = rest.split('(').next().unwrap_or("").trim();
                current_fn = Some((fn_name.to_string(), line_num + 1, 0));
                complexity = 1;
            }

            if current_fn.is_some() {
                if trimmed.contains("if ") || trimmed.contains("elif ") || trimmed.starts_with("if ") || trimmed.starts_with("elif ") {
                    complexity += 1;
                }
                if trimmed.contains("for ") || trimmed.starts_with("for ") {
                    complexity += 1;
                }
                if trimmed.contains("while ") || trimmed.starts_with("while ") {
                    complexity += 1;
                }
                if trimmed.contains(" and ") || trimmed.contains(" or ") {
                    complexity += 1;
                }
                if trimmed.contains("match ") || trimmed.starts_with("match ") {
                    complexity += 2;
                }
            }

            if trimmed.starts_with("}") || (current_fn.is_some() && line_num > 0 && !source.lines().nth(line_num + 1).map(|l| l.starts_with(&" ".repeat(4))).unwrap_or(false)) {
            }
        }

        if let Some((fn_name, start_line, _)) = current_fn {
            if complexity > 15 {
                diagnostics.push(Diagnostic::warning(
                    DiagnosticKind::ComplexFunction,
                    &format!("function `{}` is too complex (complexity: {})", fn_name, complexity),
                    Span::at(start_line, 1),
                ).with_hint("Consider breaking this function into smaller functions"));
            }
        }

        diagnostics
    }
}

#[derive(Debug)]
pub struct LongLineRule;

impl LintRule for LongLineRule {
    fn name(&self) -> &str {
        "long-line"
    }

    fn description(&self) -> &str {
        "Detects lines that exceed the maximum length"
    }

    fn check(&self, source: &str, _path: &Path) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let max_length = 100;

        for (line_num, line) in source.lines().enumerate() {
            if line.len() > max_length {
                diagnostics.push(Diagnostic::suggestion(
                    DiagnosticKind::StyleViolation,
                    &format!("line exceeds {} characters ({} chars)", max_length, line.len()),
                    Span::at(line_num + 1, max_length),
                ).with_hint("Consider breaking this line into multiple lines"));
            }
        }

        diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unused_variable_rule() {
        let rule = UnusedVariableRule;
        let source = "let x = 5\nlet y = 10\nprint(y)";
        let diags = rule.check(source, Path::new("test.zari"));
        
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("x"));
    }

    #[test]
    fn test_unused_import_rule() {
        let rule = UnusedImportRule;
        let source = "import os\nimport sys\nprint(sys.version)";
        let diags = rule.check(source, Path::new("test.zari"));
        
        assert!(!diags.is_empty());
        assert!(diags.iter().any(|d| d.message.contains("os")));
    }

    #[test]
    fn test_long_line_rule() {
        let rule = LongLineRule;
        let long_line = "x".repeat(150);
        let source = &long_line;
        let diags = rule.check(source, Path::new("test.zari"));
        
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("exceeds"));
    }

    #[test]
    fn test_todo_comment_rule() {
        let rule = TodoCommentRule;
        let source = "# TODO: implement this\n# FIXME: broken";
        let diags = rule.check(source, Path::new("test.zari"));
        
        assert_eq!(diags.len(), 2);
    }

    #[test]
    fn test_registry() {
        let registry = LintRegistry::with_default_rules();
        
        assert!(registry.get("unused-variable").is_some());
        assert!(registry.get("nonexistent").is_none());
    }
}
