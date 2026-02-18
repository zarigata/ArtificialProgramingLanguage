//! Linter configuration options

use std::collections::HashMap;
use super::diagnostic::Severity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintLevel {
    Allow,
    Warn,
    Deny,
    Forbid,
}

impl Default for LintLevel {
    fn default() -> Self {
        LintLevel::Warn
    }
}

impl LintLevel {
    pub fn to_severity(&self) -> Severity {
        match self {
            LintLevel::Allow => Severity::Info,
            LintLevel::Warn => Severity::Warning,
            LintLevel::Deny | LintLevel::Forbid => Severity::Error,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinterConfig {
    pub rules: HashMap<String, LintLevel>,
    pub max_warnings: Option<usize>,
    pub max_errors: Option<usize>,
    pub exclude_patterns: Vec<String>,
    pub include_patterns: Vec<String>,
    pub fix_safe: bool,
    pub fix_unsafe: bool,
    pub show_source: bool,
    pub color_output: bool,
    pub format: OutputFormat,
}

impl Default for LinterConfig {
    fn default() -> Self {
        let mut rules = HashMap::new();
        
        rules.insert("unused-variable".to_string(), LintLevel::Warn);
        rules.insert("unused-import".to_string(), LintLevel::Warn);
        rules.insert("unused-parameter".to_string(), LintLevel::Allow);
        rules.insert("dead-code".to_string(), LintLevel::Warn);
        rules.insert("unreachable-code".to_string(), LintLevel::Warn);
        rules.insert("deprecated-syntax".to_string(), LintLevel::Warn);
        rules.insert("naming-convention".to_string(), LintLevel::Allow);
        rules.insert("missing-doc".to_string(), LintLevel::Allow);
        rules.insert("complex-function".to_string(), LintLevel::Warn);
        rules.insert("magic-number".to_string(), LintLevel::Allow);
        rules.insert("todo-comment".to_string(), LintLevel::Allow);
        rules.insert("potential-panic".to_string(), LintLevel::Warn);
        rules.insert("security".to_string(), LintLevel::Deny);
        rules.insert("performance".to_string(), LintLevel::Warn);

        LinterConfig {
            rules,
            max_warnings: None,
            max_errors: None,
            exclude_patterns: vec![
                "**/target/**".to_string(),
                "**/node_modules/**".to_string(),
                "**/.git/**".to_string(),
            ],
            include_patterns: vec!["**/*.zari".to_string()],
            fix_safe: false,
            fix_unsafe: false,
            show_source: true,
            color_output: true,
            format: OutputFormat::default(),
        }
    }
}

impl LinterConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn strict() -> Self {
        let mut config = Self::default();
        
        for level in config.rules.values_mut() {
            *level = LintLevel::Warn;
        }
        
        config.rules.insert("security".to_string(), LintLevel::Forbid);
        config.rules.insert("potential-panic".to_string(), LintLevel::Deny);
        
        config
    }

    pub fn relaxed() -> Self {
        let mut config = Self::default();
        
        for level in config.rules.values_mut() {
            *level = LintLevel::Allow;
        }
        
        config.rules.insert("security".to_string(), LintLevel::Warn);
        
        config
    }

    pub fn set_rule(&mut self, name: &str, level: LintLevel) {
        self.rules.insert(name.to_string(), level);
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.rules.get(name)
            .map(|l| *l != LintLevel::Allow)
            .unwrap_or(true)
    }

    pub fn get_severity(&self, name: &str) -> Option<Severity> {
        self.rules.get(name).map(|l| l.to_severity())
    }

    pub fn allow(mut self, name: &str) -> Self {
        self.rules.insert(name.to_string(), LintLevel::Allow);
        self
    }

    pub fn warn(mut self, name: &str) -> Self {
        self.rules.insert(name.to_string(), LintLevel::Warn);
        self
    }

    pub fn deny(mut self, name: &str) -> Self {
        self.rules.insert(name.to_string(), LintLevel::Deny);
        self
    }

    pub fn forbid(mut self, name: &str) -> Self {
        self.rules.insert(name.to_string(), LintLevel::Forbid);
        self
    }

    pub fn with_max_warnings(mut self, max: usize) -> Self {
        self.max_warnings = Some(max);
        self
    }

    pub fn with_max_errors(mut self, max: usize) -> Self {
        self.max_errors = Some(max);
        self
    }

    pub fn fix_safe(mut self, fix: bool) -> Self {
        self.fix_safe = fix;
        self
    }

    pub fn fix_all(mut self, fix: bool) -> Self {
        self.fix_safe = fix;
        self.fix_unsafe = fix;
        self
    }

    pub fn exclude(mut self, pattern: &str) -> Self {
        self.exclude_patterns.push(pattern.to_string());
        self
    }

    pub fn include(mut self, pattern: &str) -> Self {
        self.include_patterns.push(pattern.to_string());
        self
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    pub fn from_file(path: &std::path::Path) -> std::result::Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        
        Self::parse_toml(&content)
    }

    fn parse_toml(content: &str) -> std::result::Result<Self, String> {
        let mut config = Self::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "max_warnings" => {
                        if let Ok(n) = value.parse::<usize>() {
                            config.max_warnings = Some(n);
                        }
                    }
                    "max_errors" => {
                        if let Ok(n) = value.parse::<usize>() {
                            config.max_errors = Some(n);
                        }
                    }
                    "fix_safe" => {
                        config.fix_safe = value == "true";
                    }
                    "fix_unsafe" => {
                        config.fix_unsafe = value == "true";
                    }
                    _ => {
                        if value == "allow" {
                            config.set_rule(key, LintLevel::Allow);
                        } else if value == "warn" {
                            config.set_rule(key, LintLevel::Warn);
                        } else if value == "deny" {
                            config.set_rule(key, LintLevel::Deny);
                        } else if value == "forbid" {
                            config.set_rule(key, LintLevel::Forbid);
                        }
                    }
                }
            }
        }

        Ok(config)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Human,
    Json,
    Checkstyle,
    Junit,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Human
    }
}

#[derive(Debug, Clone)]
pub struct LintOptions {
    pub check: bool,
    pub fix: bool,
    pub verbose: bool,
    pub files: Vec<std::path::PathBuf>,
    pub config_path: Option<std::path::PathBuf>,
}

impl Default for LintOptions {
    fn default() -> Self {
        LintOptions {
            check: false,
            fix: false,
            verbose: false,
            files: Vec::new(),
            config_path: None,
        }
    }
}

impl LintOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check(mut self) -> Self {
        self.check = true;
        self
    }

    pub fn fix(mut self) -> Self {
        self.fix = true;
        self
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    pub fn with_files(mut self, files: Vec<std::path::PathBuf>) -> Self {
        self.files = files;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LinterConfig::default();
        
        assert!(config.is_enabled("unused-variable"));
        assert!(config.is_enabled("security"));
        assert!(!config.fix_safe);
    }

    #[test]
    fn test_config_builder() {
        let config = LinterConfig::new()
            .deny("unused-import")
            .allow("missing-doc")
            .with_max_warnings(10);
        
        assert_eq!(config.get_severity("unused-import"), Some(Severity::Error));
        assert!(!config.is_enabled("missing-doc"));
        assert_eq!(config.max_warnings, Some(10));
    }

    #[test]
    fn test_strict_config() {
        let config = LinterConfig::strict();
        
        assert!(config.is_enabled("unused-variable"));
        assert!(config.is_enabled("security"));
    }

    #[test]
    fn test_relaxed_config() {
        let config = LinterConfig::relaxed();
        
        assert!(!config.is_enabled("unused-variable"));
    }

    #[test]
    fn test_parse_toml() {
        let toml = r#"
unused-variable = "deny"
missing-doc = "allow"
max_warnings = 100
"#;
        
        let config = LinterConfig::parse_toml(toml).unwrap();
        assert_eq!(config.get_severity("unused-variable"), Some(Severity::Error));
        assert!(!config.is_enabled("missing-doc"));
        assert_eq!(config.max_warnings, Some(100));
    }
}
