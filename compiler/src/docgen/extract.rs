//! Documentation extraction from VeZ source code

use std::path::Path;
use std::collections::HashMap;
use super::{DocResult, DocError, Documentation, ModuleDoc, ItemDoc, ItemKind, Visibility, ParamDoc, ReturnDoc, CodeExample, AttributeDoc};

pub struct DocExtractor {
    comment_patterns: Vec<(String, String)>,
}

impl DocExtractor {
    pub fn new() -> Self {
        DocExtractor {
            comment_patterns: vec![
                ("##".to_string(), "## ".to_string()),
                ("///".to_string(), "/// ".to_string()),
            ],
        }
    }

    pub fn extract_from_file(&self, path: &Path) -> DocResult<Documentation> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DocError::IoError(e.to_string()))?;
        
        self.extract_from_source(&content, path)
    }

    pub fn extract_from_source(&self, source: &str, path: &Path) -> DocResult<Documentation> {
        let module_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        let mut items = Vec::new();
        let mut current_doc: Option<DocComment> = None;
        let mut in_doc_block = false;
        let mut doc_lines: Vec<String> = Vec::new();
        
        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("##") || trimmed.starts_with("///") {
                if !in_doc_block {
                    in_doc_block = true;
                    doc_lines.clear();
                }
                let doc_content = if trimmed.starts_with("## ") {
                    trimmed[3..].to_string()
                } else if trimmed.starts_with("##") {
                    trimmed[2..].to_string()
                } else if trimmed.starts_with("/// ") {
                    trimmed[4..].to_string()
                } else {
                    trimmed[3..].to_string()
                };
                doc_lines.push(doc_content);
            } else if in_doc_block && !trimmed.is_empty() {
                in_doc_block = false;
                current_doc = Some(DocComment {
                    lines: doc_lines.clone(),
                    line_number: line_num,
                });
                
                if let Some(item) = self.parse_item(trimmed, current_doc.take(), line_num + 1) {
                    items.push(item);
                }
            } else if in_doc_block && trimmed.is_empty() {
                in_doc_block = false;
            }
            
            if trimmed.starts_with("def ") || trimmed.starts_with("fn ") {
                if let Some(doc) = current_doc.take() {
                    if let Some(item) = self.parse_function(trimmed, doc, line_num + 1) {
                        items.push(item);
                    }
                } else {
                    if let Some(item) = self.parse_function(trimmed, DocComment::empty(line_num + 1), line_num + 1) {
                        items.push(item);
                    }
                }
            } else if trimmed.starts_with("struct ") {
                if let Some(doc) = current_doc.take() {
                    if let Some(item) = self.parse_struct(trimmed, doc, line_num + 1) {
                        items.push(item);
                    }
                }
            } else if trimmed.starts_with("enum ") {
                if let Some(doc) = current_doc.take() {
                    if let Some(item) = self.parse_enum(trimmed, doc, line_num + 1) {
                        items.push(item);
                    }
                }
            } else if trimmed.starts_with("const ") {
                if let Some(doc) = current_doc.take() {
                    if let Some(item) = self.parse_const(trimmed, doc, line_num + 1) {
                        items.push(item);
                    }
                }
            }
        }
        
        let module_doc = self.extract_module_doc(source, &module_name, path);
        
        Ok(Documentation {
            module: module_doc,
            items,
            cross_refs: HashMap::new(),
        })
    }

    fn extract_module_doc(&self, source: &str, name: &str, path: &Path) -> ModuleDoc {
        let mut description = None;
        let mut examples = Vec::new();
        
        let mut in_module_doc = false;
        let mut doc_lines = Vec::new();
        
        for line in source.lines() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("//!") {
                in_module_doc = true;
                let content = if trimmed.starts_with("//! ") {
                    &trimmed[4..]
                } else {
                    &trimmed[3..]
                };
                doc_lines.push(content.to_string());
            } else if in_module_doc {
                break;
            }
        }
        
        if !doc_lines.is_empty() {
            description = Some(doc_lines.join("\n"));
        }
        
        ModuleDoc {
            name: name.to_string(),
            path: path.to_path_buf(),
            description,
            examples,
        }
    }

    fn parse_item(&self, _line: &str, _doc: Option<DocComment>, _line_num: usize) -> Option<ItemDoc> {
        None
    }

    fn parse_function(&self, line: &str, doc: DocComment, line_num: usize) -> Option<ItemDoc> {
        let rest = if line.starts_with("def ") {
            &line[4..]
        } else if line.starts_with("fn ") {
            &line[3..]
        } else {
            return None;
        };
        
        let name_end = rest.find('(').unwrap_or(rest.len());
        let name = rest[..name_end].trim().to_string();
        
        let params_str = rest.get(name_end..).unwrap_or("");
        let params = self.parse_params(params_str);
        
        let returns = if let Some(ret_start) = rest.find("->") {
            let ret_type = rest[ret_start + 2..].split(':').next().unwrap_or("").trim();
            Some(ReturnDoc {
                type_name: ret_type.to_string(),
                description: doc.get_param_doc("return"),
            })
        } else {
            None
        };
        
        Some(ItemDoc {
            name,
            kind: ItemKind::Function,
            visibility: Visibility::Public,
            description: doc.description(),
            params,
            returns,
            examples: doc.examples(),
            attributes: Vec::new(),
            source_line: line_num,
        })
    }

    fn parse_struct(&self, line: &str, doc: DocComment, line_num: usize) -> Option<ItemDoc> {
        let rest = line.strip_prefix("struct ")?;
        let name = rest.split('{').next().unwrap_or(rest).trim().to_string();
        
        Some(ItemDoc {
            name,
            kind: ItemKind::Struct,
            visibility: Visibility::Public,
            description: doc.description(),
            params: Vec::new(),
            returns: None,
            examples: doc.examples(),
            attributes: Vec::new(),
            source_line: line_num,
        })
    }

    fn parse_enum(&self, line: &str, doc: DocComment, line_num: usize) -> Option<ItemDoc> {
        let rest = line.strip_prefix("enum ")?;
        let name = rest.split('{').next().unwrap_or(rest).trim().to_string();
        
        Some(ItemDoc {
            name,
            kind: ItemKind::Enum,
            visibility: Visibility::Public,
            description: doc.description(),
            params: Vec::new(),
            returns: None,
            examples: doc.examples(),
            attributes: Vec::new(),
            source_line: line_num,
        })
    }

    fn parse_const(&self, line: &str, doc: DocComment, line_num: usize) -> Option<ItemDoc> {
        let rest = line.strip_prefix("const ")?;
        let name_end = rest.find(':').unwrap_or(rest.find('=').unwrap_or(rest.len()));
        let name = rest[..name_end].trim().to_string();
        
        Some(ItemDoc {
            name,
            kind: ItemKind::Const,
            visibility: Visibility::Public,
            description: doc.description(),
            params: Vec::new(),
            returns: None,
            examples: doc.examples(),
            attributes: Vec::new(),
            source_line: line_num,
        })
    }

    fn parse_params(&self, params_str: &str) -> Vec<ParamDoc> {
        let mut params = Vec::new();
        
        let inner = params_str.trim_start_matches('(').trim_end_matches(')');
        if inner.is_empty() {
            return params;
        }
        
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            
            let (name, type_name) = if let Some(colon_pos) = part.find(':') {
                (part[..colon_pos].trim(), part[colon_pos + 1..].trim())
            } else {
                (part, "unknown")
            };
            
            params.push(ParamDoc {
                name: name.to_string(),
                type_name: type_name.to_string(),
                description: None,
            });
        }
        
        params
    }
}

impl Default for DocExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct DocComment {
    pub lines: Vec<String>,
    pub line_number: usize,
}

impl DocComment {
    pub fn empty(line_number: usize) -> Self {
        DocComment {
            lines: Vec::new(),
            line_number,
        }
    }

    pub fn description(&self) -> Option<String> {
        let desc_lines: Vec<&str> = self.lines.iter()
            .take_while(|l| !l.starts_with('@'))
            .map(|s| s.as_str())
            .collect();
        
        if desc_lines.is_empty() {
            None
        } else {
            Some(desc_lines.join("\n"))
        }
    }

    pub fn get_param_doc(&self, param_name: &str) -> Option<String> {
        let prefix = format!("@{} ", param_name);
        for line in &self.lines {
            if line.starts_with(&prefix) {
                return Some(line[prefix.len()..].to_string());
            }
        }
        None
    }

    pub fn examples(&self) -> Vec<CodeExample> {
        let mut examples = Vec::new();
        let mut in_example = false;
        let mut example_code = String::new();
        let mut example_desc = None;
        
        for line in &self.lines {
            if line.starts_with("@example") {
                if in_example && !example_code.is_empty() {
                    examples.push(CodeExample {
                        code: example_code.trim().to_string(),
                        description: example_desc,
                        language: "zari".to_string(),
                    });
                }
                in_example = true;
                example_code = String::new();
                example_desc = Some(line[8..].trim().to_string());
            } else if in_example {
                if line.starts_with('@') && !line.starts_with("@example") {
                    examples.push(CodeExample {
                        code: example_code.trim().to_string(),
                        description: example_desc,
                        language: "zari".to_string(),
                    });
                    in_example = false;
                    example_code = String::new();
                    example_desc = None;
                } else {
                    example_code.push_str(line);
                    example_code.push('\n');
                }
            }
        }
        
        if in_example && !example_code.is_empty() {
            examples.push(CodeExample {
                code: example_code.trim().to_string(),
                description: example_desc,
                language: "zari".to_string(),
            });
        }
        
        examples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_function() {
        let extractor = DocExtractor::new();
        let source = r#"
## Calculate the sum of two numbers
## @param a First number
## @param b Second number
## @return The sum
def add(a: int, b: int) -> int:
    return a + b
"#;
        let result = extractor.extract_from_source(source, Path::new("test.zari"));
        assert!(result.is_ok());
        
        let docs = result.unwrap();
        assert_eq!(docs.items.len(), 1);
        assert_eq!(docs.items[0].name, "add");
        assert_eq!(docs.items[0].kind, ItemKind::Function);
    }

    #[test]
    fn test_doc_comment_description() {
        let doc = DocComment {
            lines: vec!["Hello".to_string(), "World".to_string()],
            line_number: 1,
        };
        assert_eq!(doc.description(), Some("Hello\nWorld".to_string()));
    }

    #[test]
    fn test_doc_comment_param() {
        let doc = DocComment {
            lines: vec!["@param x The value".to_string()],
            line_number: 1,
        };
        assert_eq!(doc.get_param_doc("param x"), Some("The value".to_string()));
    }
}
