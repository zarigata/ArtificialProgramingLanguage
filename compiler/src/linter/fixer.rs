//! Auto-fix support for the linter

use super::diagnostic::Fix;

pub struct Fixer {
    source: String,
    edits: Vec<Edit>,
}

#[derive(Debug, Clone)]
struct Edit {
    offset: usize,
    length: usize,
    replacement: String,
}

impl Fixer {
    pub fn new(source: &str) -> Self {
        Fixer {
            source: source.to_string(),
            edits: Vec::new(),
        }
    }

    pub fn apply(&mut self, fix: &Fix) -> bool {
        let offset = self.position_to_offset(fix.span.start.line, fix.span.start.column);
        let end_offset = self.position_to_offset(fix.span.end.line, fix.span.end.column);
        let length = end_offset.saturating_sub(offset);

        self.edits.push(Edit {
            offset,
            length,
            replacement: fix.replacement.clone(),
        });

        true
    }

    pub fn finish(self) -> String {
        let mut result = self.source;
        
        self.edits.iter()
            .rev()
            .for_each(|edit| {
                if edit.offset <= result.len() {
                    let end = (edit.offset + edit.length).min(result.len());
                    result.replace_range(edit.offset..end, &edit.replacement);
                }
            });

        result
    }

    fn position_to_offset(&self, line: usize, column: usize) -> usize {
        let mut offset = 0;
        let mut current_line = 1;

        for ch in self.source.chars() {
            if current_line == line {
                return offset + column.saturating_sub(1);
            }
            
            if ch == '\n' {
                current_line += 1;
            }
            offset += 1;
        }

        if current_line == line {
            return offset + column.saturating_sub(1);
        }

        offset
    }
}

pub fn apply_fixes(source: &str, fixes: &[Fix]) -> String {
    let mut fixer = Fixer::new(source);
    
    for fix in fixes {
        fixer.apply(fix);
    }
    
    fixer.finish()
}

pub fn apply_safe_fixes(source: &str, diagnostics: &[super::diagnostic::Diagnostic]) -> (String, Vec<Fix>) {
    let mut fixer = Fixer::new(source);
    let mut applied = Vec::new();

    for diag in diagnostics {
        if let Some(fix) = &diag.fix {
            if fixer.apply(fix) {
                applied.push(fix.clone());
            }
        }
    }

    (fixer.finish(), applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::diagnostic::Span;

    #[test]
    fn test_fixer_replace() {
        let source = "let x = 5";
        let mut fixer = Fixer::new(source);
        
        let fix = Fix::new(Span::at(1, 5), "y", "Rename variable");
        fixer.apply(&fix);
        
        let result = fixer.finish();
        assert_eq!(result, "let y = 5");
    }

    #[test]
    fn test_fixer_remove() {
        let source = "let x = 5";
        let mut fixer = Fixer::new(source);
        
        let fix = Fix::remove(Span::range(1, 1, 1, 9));
        fixer.apply(&fix);
        
        let result = fixer.finish();
        assert_eq!(result, "");
    }

    #[test]
    fn test_fixer_multiple_edits() {
        let source = "let x = 1\nlet y = 2";
        let mut fixer = Fixer::new(source);
        
        fixer.apply(&Fix::new(Span::at(1, 5), "a", ""));
        fixer.apply(&Fix::new(Span::at(2, 5), "b", ""));
        
        let result = fixer.finish();
        assert!(result.contains("let a = 1"));
        assert!(result.contains("let b = 2"));
    }

    #[test]
    fn test_position_to_offset() {
        let source = "line1\nline2\nline3";
        let fixer = Fixer::new(source);
        
        assert_eq!(fixer.position_to_offset(1, 1), 0);
        assert_eq!(fixer.position_to_offset(2, 1), 6);
        assert_eq!(fixer.position_to_offset(3, 1), 12);
    }

    #[test]
    fn test_apply_fixes() {
        let source = "let x = 5";
        let fixes = vec![
            Fix::new(Span::at(1, 5), "y", ""),
        ];
        
        let result = apply_fixes(source, &fixes);
        assert_eq!(result, "let y = 5");
    }
}
