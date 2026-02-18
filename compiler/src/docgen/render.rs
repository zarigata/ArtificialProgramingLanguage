//! Documentation rendering trait and common utilities

use super::{Documentation, DocResult};

pub trait DocRenderer {
    fn render(&self, docs: &Documentation) -> DocResult<String>;
    fn render_index(&self, files: &[std::path::PathBuf]) -> DocResult<String>;
    fn file_extension(&self) -> &str;
}

pub fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub fn format_type(type_name: &str) -> String {
    type_name.to_string()
}

pub fn slugify(name: &str) -> String {
    name.to_lowercase()
        .replace(' ', "-")
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "")
}

pub fn highlight_code(code: &str, _language: &str) -> String {
    code.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
    }

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(slugify("My Function!"), "my-function");
    }
}
