//! Formatter configuration options

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndentStyle {
    Tab,
    Spaces(usize),
}

impl Default for IndentStyle {
    fn default() -> Self {
        IndentStyle::Spaces(4)
    }
}

impl IndentStyle {
    pub fn as_str(&self) -> String {
        match self {
            IndentStyle::Tab => "\t".to_string(),
            IndentStyle::Spaces(n) => " ".repeat(*n),
        }
    }

    pub fn width(&self) -> usize {
        match self {
            IndentStyle::Tab => 1,
            IndentStyle::Spaces(n) => *n,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LineWidth {
    pub max: usize,
    pub ideal: usize,
}

impl Default for LineWidth {
    fn default() -> Self {
        LineWidth {
            max: 100,
            ideal: 80,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BraceStyle {
    SameLine,
    NextLine,
    AlwaysNextLine,
}

impl Default for BraceStyle {
    fn default() -> Self {
        BraceStyle::SameLine
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchArmStyle {
    SameLine,
    NextLine,
}

impl Default for MatchArmStyle {
    fn default() -> Self {
        MatchArmStyle::NextLine
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrailingComma {
    Always,
    Never,
    OnlyMultiline,
}

impl Default for TrailingComma {
    fn default() -> Self {
        TrailingComma::OnlyMultiline
    }
}

#[derive(Debug, Clone)]
pub struct FormatterConfig {
    pub indent_style: IndentStyle,
    pub line_width: LineWidth,
    pub brace_style: BraceStyle,
    pub match_arm_style: MatchArmStyle,
    pub trailing_comma: TrailingComma,
    pub space_around_operators: bool,
    pub space_after_colon: bool,
    pub space_before_colon: bool,
    pub blank_lines_upper_bound: usize,
    pub blank_lines_lower_bound: usize,
    pub wrap_comments: bool,
    pub comment_width: usize,
    pub normalize_comments: bool,
    pub format_strings: bool,
    pub max_empty_lines: usize,
    pub merge_imports: bool,
    pub reorder_imports: bool,
    pub imports_granularity: ImportGranularity,
    pub edition: String,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        FormatterConfig {
            indent_style: IndentStyle::default(),
            line_width: LineWidth::default(),
            brace_style: BraceStyle::default(),
            match_arm_style: MatchArmStyle::default(),
            trailing_comma: TrailingComma::default(),
            space_around_operators: true,
            space_after_colon: true,
            space_before_colon: false,
            blank_lines_upper_bound: 1,
            blank_lines_lower_bound: 0,
            wrap_comments: false,
            comment_width: 80,
            normalize_comments: true,
            format_strings: false,
            max_empty_lines: 1,
            merge_imports: false,
            reorder_imports: false,
            imports_granularity: ImportGranularity::default(),
            edition: "2024".to_string(),
        }
    }
}

impl FormatterConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_indent_style(mut self, style: IndentStyle) -> Self {
        self.indent_style = style;
        self
    }

    pub fn with_line_width(mut self, max: usize) -> Self {
        self.line_width.max = max;
        self.line_width.ideal = (max * 4 / 5).max(40);
        self
    }

    pub fn with_tabs(mut self) -> Self {
        self.indent_style = IndentStyle::Tab;
        self
    }

    pub fn with_spaces(mut self, width: usize) -> Self {
        self.indent_style = IndentStyle::Spaces(width);
        self
    }

    pub fn with_brace_style(mut self, style: BraceStyle) -> Self {
        self.brace_style = style;
        self
    }

    pub fn indent_str(&self) -> String {
        self.indent_style.as_str()
    }

    pub fn max_width(&self) -> usize {
        self.line_width.max
    }

    pub fn from_file(path: &std::path::Path) -> std::result::Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        
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
                    "indent_style" => {
                        if value == "\"tab\"" || value == "tab" {
                            config.indent_style = IndentStyle::Tab;
                        } else if value.starts_with("\"spaces") {
                            if let Some(n) = value.strip_prefix("\"spaces ").and_then(|s| s.strip_suffix('"')) {
                                if let Ok(spaces) = n.parse::<usize>() {
                                    config.indent_style = IndentStyle::Spaces(spaces);
                                }
                            }
                        }
                    }
                    "max_width" | "line_width" => {
                        if let Ok(width) = value.parse::<usize>() {
                            config.line_width.max = width;
                            config.line_width.ideal = (width * 4 / 5).max(40);
                        }
                    }
                    "space_around_operators" => {
                        config.space_around_operators = value == "true";
                    }
                    "space_after_colon" => {
                        config.space_after_colon = value == "true";
                    }
                    "wrap_comments" => {
                        config.wrap_comments = value == "true";
                    }
                    _ => {}
                }
            }
        }

        Ok(config)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportGranularity {
    Item,
    Module,
    Crate,
    Preserve,
}

impl Default for ImportGranularity {
    fn default() -> Self {
        ImportGranularity::Module
    }
}

#[derive(Debug, Clone)]
pub struct FormatOptions {
    pub check: bool,
    pub emit_stdout: bool,
    pub backup: bool,
    pub verbose: bool,
    pub files: Vec<PathBuf>,
    pub config_path: Option<PathBuf>,
}

impl Default for FormatOptions {
    fn default() -> Self {
        FormatOptions {
            check: false,
            emit_stdout: false,
            backup: false,
            verbose: false,
            files: Vec::new(),
            config_path: None,
        }
    }
}

impl FormatOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check(mut self) -> Self {
        self.check = true;
        self
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    pub fn with_files(mut self, files: Vec<PathBuf>) -> Self {
        self.files = files;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indent_style() {
        let tab = IndentStyle::Tab;
        assert_eq!(tab.as_str(), "\t");
        assert_eq!(tab.width(), 1);

        let spaces = IndentStyle::Spaces(2);
        assert_eq!(spaces.as_str(), "  ");
        assert_eq!(spaces.width(), 2);
    }

    #[test]
    fn test_default_config() {
        let config = FormatterConfig::default();
        
        assert_eq!(config.indent_style, IndentStyle::Spaces(4));
        assert_eq!(config.line_width.max, 100);
        assert!(config.space_around_operators);
        assert!(config.space_after_colon);
        assert!(!config.space_before_colon);
    }

    #[test]
    fn test_config_builder() {
        let config = FormatterConfig::new()
            .with_tabs()
            .with_line_width(120);
        
        assert_eq!(config.indent_style, IndentStyle::Tab);
        assert_eq!(config.line_width.max, 120);
    }

    #[test]
    fn test_parse_toml() {
        let toml = r#"
max_width = 120
indent_style = "tab"
space_around_operators = true
"#;
        
        let config = FormatterConfig::parse_toml(toml).unwrap();
        assert_eq!(config.line_width.max, 120);
        assert_eq!(config.indent_style, IndentStyle::Tab);
    }
}
