//! Markdown documentation renderer

use std::path::PathBuf;
use super::{Documentation, DocResult, DocRenderer, ItemDoc, ItemKind, render};

pub struct MarkdownRenderer {
    heading_level: usize,
}

impl MarkdownRenderer {
    pub fn new() -> Self {
        MarkdownRenderer { heading_level: 1 }
    }

    pub fn with_heading_level(mut self, level: usize) -> Self {
        self.heading_level = level;
        self
    }
}

impl Default for MarkdownRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl DocRenderer for MarkdownRenderer {
    fn render(&self, docs: &Documentation) -> DocResult<String> {
        let mut md = String::new();
        
        md.push_str(&format!(
            "{} {}\n\n",
            "#".repeat(self.heading_level),
            docs.module.name
        ));
        
        if let Some(desc) = &docs.module.description {
            md.push_str(&format!("{}\n\n", desc));
        }
        
        md.push_str("## Contents\n\n");
        for item in &docs.items {
            let slug = render::slugify(&item.name);
            let kind = item_kind_str(&item.kind);
            md.push_str(&format!(
                "- [{} `{}`](#{})\n",
                kind, item.name, slug
            ));
        }
        md.push('\n');
        
        for item in &docs.items {
            md.push_str(&self.render_item(item));
        }
        
        Ok(md)
    }

    fn render_index(&self, files: &[PathBuf]) -> DocResult<String> {
        let mut md = String::new();
        
        md.push_str("# VeZ Documentation\n\n");
        md.push_str("## Modules\n\n");
        
        for file in files {
            if let Some(name) = file.file_stem().and_then(|s| s.to_str()) {
                if name != "index" {
                    md.push_str(&format!("- [{}]({}.md)\n", name, name));
                }
            }
        }
        
        Ok(md)
    }

    fn file_extension(&self) -> &str {
        "md"
    }
}

impl MarkdownRenderer {
    fn render_item(&self, item: &ItemDoc) -> String {
        let mut md = String::new();
        let kind = item_kind_str(&item.kind);
        let heading = "#".repeat(self.heading_level + 1);
        
        md.push_str(&format!(
            "{} {} `{}`\n\n",
            heading, kind, item.name
        ));
        
        if let Some(desc) = &item.description {
            md.push_str(&format!("{}\n\n", desc));
        }
        
        if !item.params.is_empty() {
            md.push_str("### Parameters\n\n");
            md.push_str("| Name | Type | Description |\n");
            md.push_str("|------|------|-------------|\n");
            for param in &item.params {
                let desc = param.description.as_deref().unwrap_or("");
                md.push_str(&format!(
                    "| `{}` | `{}` | {} |\n",
                    param.name, param.type_name, desc
                ));
            }
            md.push('\n');
        }
        
        if let Some(ret) = &item.returns {
            md.push_str("### Returns\n\n");
            md.push_str(&format!("`{}`", ret.type_name));
            if let Some(desc) = &ret.description {
                md.push_str(&format!(" - {}", desc));
            }
            md.push_str("\n\n");
        }
        
        if !item.examples.is_empty() {
            md.push_str("### Examples\n\n");
            for example in &item.examples {
                if let Some(desc) = &example.description {
                    md.push_str(&format!("{}:\n\n", desc));
                }
                md.push_str(&format!(
                    "```{}\n{}\n```\n\n",
                    example.language, example.code
                ));
            }
        }
        
        md.push_str(&format!("*Source: line {}*\n\n---\n\n", item.source_line));
        
        md
    }
}

fn item_kind_str(kind: &ItemKind) -> &'static str {
    match kind {
        ItemKind::Function => "Function",
        ItemKind::Struct => "Struct",
        ItemKind::Enum => "Enum",
        ItemKind::Trait => "Trait",
        ItemKind::Const => "Const",
        ItemKind::Static => "Static",
        ItemKind::Type => "Type",
        ItemKind::Module => "Module",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docgen::{ModuleDoc, Visibility, ParamDoc};
    use std::path::PathBuf;

    fn make_test_docs() -> Documentation {
        Documentation {
            module: ModuleDoc {
                name: "test".to_string(),
                path: PathBuf::from("test.zari"),
                description: Some("A test module".to_string()),
                examples: vec![],
            },
            items: vec![ItemDoc {
                name: "add".to_string(),
                kind: ItemKind::Function,
                visibility: Visibility::Public,
                description: Some("Add two numbers".to_string()),
                params: vec![
                    ParamDoc {
                        name: "a".to_string(),
                        type_name: "int".to_string(),
                        description: Some("First number".to_string()),
                    },
                    ParamDoc {
                        name: "b".to_string(),
                        type_name: "int".to_string(),
                        description: Some("Second number".to_string()),
                    },
                ],
                returns: Some(crate::docgen::ReturnDoc {
                    type_name: "int".to_string(),
                    description: Some("The sum".to_string()),
                }),
                examples: vec![],
                attributes: vec![],
                source_line: 10,
            }],
            cross_refs: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_markdown_render() {
        let renderer = MarkdownRenderer::new();
        let docs = make_test_docs();
        let md = renderer.render(&docs).unwrap();
        
        assert!(md.contains("# test"));
        assert!(md.contains("Function `add`"));
        assert!(md.contains("| `a` | `int` | First number |"));
    }

    #[test]
    fn test_markdown_index() {
        let renderer = MarkdownRenderer::new();
        let files = vec![PathBuf::from("output/module1.md")];
        let md = renderer.render_index(&files).unwrap();
        
        assert!(md.contains("# VeZ Documentation"));
        assert!(md.contains("[module1](module1.md)"));
    }
}
