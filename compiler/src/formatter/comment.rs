//! Comment formatting utilities

use super::config::FormatterConfig;

pub struct CommentFormatter;

impl CommentFormatter {
    pub fn format(comment: &str, config: &FormatterConfig) -> String {
        if comment.starts_with("///") {
            Self::format_doc_comment(comment, config)
        } else if comment.starts_with("##") {
            Self::format_doc_comment(comment, config)
        } else if comment.starts_with("#!") {
            Self::format_inner_doc(comment, config)
        } else {
            Self::format_line_comment(comment, config)
        }
    }

    fn format_doc_comment(comment: &str, config: &FormatterConfig) -> String {
        if config.normalize_comments {
            let content = comment.trim_start_matches('#').trim_start_matches('/').trim();
            
            if content.is_empty() {
                return comment.to_string();
            }

            if comment.starts_with("##") {
                format!("## {}", content)
            } else {
                format!("/// {}", content)
            }
        } else {
            comment.to_string()
        }
    }

    fn format_inner_doc(comment: &str, _config: &FormatterConfig) -> String {
        comment.to_string()
    }

    fn format_line_comment(comment: &str, config: &FormatterConfig) -> String {
        if config.normalize_comments {
            let content = comment.trim_start_matches('#').trim();
            
            if content.is_empty() {
                return "# ".to_string();
            }

            format!("# {}", content)
        } else {
            comment.to_string()
        }
    }

    pub fn wrap_comment(comment: &str, max_width: usize, indent: &str) -> String {
        if comment.len() <= max_width {
            return comment.to_string();
        }

        let prefix = if comment.starts_with("///") {
            "///"
        } else if comment.starts_with("##") {
            "##"
        } else if comment.starts_with("#!") {
            "#!"
        } else {
            "#"
        };

        let content = comment.trim_start_matches(|c| c == '#' || c == '/').trim();
        let content_width = max_width.saturating_sub(prefix.len() + 1);

        if content.len() <= content_width {
            return comment.to_string();
        }

        let words: Vec<&str> = content.split_whitespace().collect();
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in words {
            if current_line.is_empty() {
                current_line.push_str(word);
            } else if current_line.len() + 1 + word.len() <= content_width {
                current_line.push(' ');
                current_line.push_str(word);
            } else {
                lines.push(format!("{} {}", prefix, current_line));
                current_line = word.to_string();
            }
        }

        if !current_line.is_empty() {
            lines.push(format!("{} {}", prefix, current_line));
        }

        lines.join(&format!("\n{}", indent))
    }

    pub fn format_block(comment: &str, config: &FormatterConfig) -> String {
        let lines: Vec<&str> = comment.lines().collect();
        
        lines.iter()
            .map(|line| Self::format(line, config))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_line_comment() {
        let config = FormatterConfig::default();
        
        assert_eq!(
            CommentFormatter::format("#hello", &config),
            "# hello"
        );
        
        assert_eq!(
            CommentFormatter::format("#  hello", &config),
            "# hello"
        );
    }

    #[test]
    fn test_format_doc_comment() {
        let config = FormatterConfig::default();
        
        assert_eq!(
            CommentFormatter::format("##hello", &config),
            "## hello"
        );
        
        assert_eq!(
            CommentFormatter::format("///  hello", &config),
            "/// hello"
        );
    }

    #[test]
    fn test_wrap_comment() {
        let long_comment = "# This is a very long comment that should be wrapped to fit within the specified maximum width limit";
        let wrapped = CommentFormatter::wrap_comment(long_comment, 40, "");
        
        assert!(wrapped.lines().all(|l| l.len() <= 42)); // 40 + "# "
    }

    #[test]
    fn test_preserve_short_comment() {
        let config = FormatterConfig::default();
        
        let short = "# short";
        assert_eq!(CommentFormatter::format(short, &config), "# short");
    }
}
