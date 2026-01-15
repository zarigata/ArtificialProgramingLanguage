//! C++-style syntax adapter for VeZ

use crate::error::{Error, ErrorKind, Result};
use crate::parser::ast::*;

/// Placeholder for C++-style parser
pub fn parse(source: &str) -> Result<Program> {
    Err(Error::new(
        ErrorKind::InvalidSyntax,
        "C++-style parser not yet fully implemented"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpp_style_placeholder() {
        let source = r#"
int add(int x, int y) {
    return x + y;
}
"#;
        let result = parse(source);
        assert!(result.is_err());
    }
}
