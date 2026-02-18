//! Indentation management for the formatter

use super::config::IndentStyle;

pub struct IndentManager {
    style: IndentStyle,
    level: usize,
}

impl IndentManager {
    pub fn new(style: IndentStyle) -> Self {
        IndentManager {
            style,
            level: 0,
        }
    }

    pub fn current(&self) -> String {
        self.style.as_str().repeat(self.level)
    }

    pub fn current_width(&self) -> usize {
        self.level * self.style.width()
    }

    pub fn increase(&mut self) {
        self.level += 1;
    }

    pub fn decrease(&mut self) {
        self.level = self.level.saturating_sub(1);
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn set_level(&mut self, level: usize) {
        self.level = level;
    }

    pub fn indent(&self, s: &str) -> String {
        let indent_str = self.current();
        s.lines()
            .map(|line| {
                if line.is_empty() {
                    String::new()
                } else {
                    format!("{}{}", indent_str, line)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn dedent(&self, s: &str) -> String {
        let indent_width = self.style.width();
        s.lines()
            .map(|line| {
                match &self.style {
                    IndentStyle::Tab => {
                        if line.starts_with('\t') {
                            line[1..].to_string()
                        } else {
                            line.to_string()
                        }
                    }
                    IndentStyle::Spaces(_) => {
                        let spaces = " ".repeat(indent_width);
                        if line.starts_with(&spaces) {
                            line[indent_width..].to_string()
                        } else {
                            line.to_string()
                        }
                    }
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indent_spaces() {
        let mut indent = IndentManager::new(IndentStyle::Spaces(4));
        
        assert_eq!(indent.current(), "");
        assert_eq!(indent.current_width(), 0);
        
        indent.increase();
        assert_eq!(indent.current(), "    ");
        assert_eq!(indent.current_width(), 4);
        
        indent.increase();
        assert_eq!(indent.current(), "        ");
        assert_eq!(indent.current_width(), 8);
        
        indent.decrease();
        assert_eq!(indent.current(), "    ");
    }

    #[test]
    fn test_indent_tabs() {
        let mut indent = IndentManager::new(IndentStyle::Tab);
        
        indent.increase();
        assert_eq!(indent.current(), "\t");
        
        indent.increase();
        assert_eq!(indent.current(), "\t\t");
    }

    #[test]
    fn test_indent_string() {
        let mut indent = IndentManager::new(IndentStyle::Spaces(2));
        indent.increase();
        
        let input = "line1\nline2\nline3";
        let result = indent.indent(input);
        
        assert_eq!(result, "  line1\n  line2\n  line3");
    }

    #[test]
    fn test_dedent_string() {
        let indent = IndentManager::new(IndentStyle::Spaces(2));
        
        let input = "  line1\n  line2\nline3";
        let result = indent.dedent(input);
        
        assert_eq!(result, "line1\nline2\nline3");
    }
}
