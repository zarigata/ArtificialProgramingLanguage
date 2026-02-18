//! Documentation Generator for VeZ
//!
//! Generates HTML and Markdown documentation from VeZ source code.

pub mod extract;
pub mod render;
pub mod html;
pub mod markdown;

pub use extract::{DocExtractor, DocComment};
pub use render::DocRenderer;
pub use html::HtmlRenderer;
pub use markdown::MarkdownRenderer;

use std::path::{Path, PathBuf};
use std::collections::HashMap;

pub type DocResult<T> = std::result::Result<T, DocError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocError {
    IoError(String),
    ParseError(String),
    InvalidPath(String),
    TemplateError(String),
}

impl std::fmt::Display for DocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocError::IoError(msg) => write!(f, "IO error: {}", msg),
            DocError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            DocError::InvalidPath(msg) => write!(f, "Invalid path: {}", msg),
            DocError::TemplateError(msg) => write!(f, "Template error: {}", msg),
        }
    }
}

impl std::error::Error for DocError {}

#[derive(Debug, Clone)]
pub struct Documentation {
    pub module: ModuleDoc,
    pub items: Vec<ItemDoc>,
    pub cross_refs: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ModuleDoc {
    pub name: String,
    pub path: PathBuf,
    pub description: Option<String>,
    pub examples: Vec<CodeExample>,
}

#[derive(Debug, Clone)]
pub struct ItemDoc {
    pub name: String,
    pub kind: ItemKind,
    pub visibility: Visibility,
    pub description: Option<String>,
    pub params: Vec<ParamDoc>,
    pub returns: Option<ReturnDoc>,
    pub examples: Vec<CodeExample>,
    pub attributes: Vec<AttributeDoc>,
    pub source_line: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Function,
    Struct,
    Enum,
    Trait,
    Const,
    Static,
    Type,
    Module,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
    Internal,
}

#[derive(Debug, Clone)]
pub struct ParamDoc {
    pub name: String,
    pub type_name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ReturnDoc {
    pub type_name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CodeExample {
    pub code: String,
    pub description: Option<String>,
    pub language: String,
}

#[derive(Debug, Clone)]
pub struct AttributeDoc {
    pub name: String,
    pub args: Vec<String>,
}

pub struct DocGenerator {
    extractor: DocExtractor,
    renderer: Box<dyn DocRenderer>,
}

impl DocGenerator {
    pub fn html() -> Self {
        DocGenerator {
            extractor: DocExtractor::new(),
            renderer: Box::new(HtmlRenderer::new()),
        }
    }

    pub fn markdown() -> Self {
        DocGenerator {
            extractor: DocExtractor::new(),
            renderer: Box::new(MarkdownRenderer::new()),
        }
    }

    pub fn with_renderer(mut self, renderer: Box<dyn DocRenderer>) -> Self {
        self.renderer = renderer;
        self
    }

    pub fn generate(&self, source: &Path, output: &Path) -> DocResult<()> {
        let docs = self.extractor.extract_from_file(source)?;
        let rendered = self.renderer.render(&docs)?;
        
        std::fs::create_dir_all(output.parent().unwrap_or(output))
            .map_err(|e| DocError::IoError(e.to_string()))?;
        
        std::fs::write(output, rendered)
            .map_err(|e| DocError::IoError(e.to_string()))?;
        
        Ok(())
    }

    pub fn generate_project(&self, src_dir: &Path, output_dir: &Path) -> DocResult<Vec<PathBuf>> {
        let mut generated = Vec::new();
        
        if !src_dir.exists() {
            return Err(DocError::InvalidPath(format!("{:?} does not exist", src_dir)));
        }
        
        std::fs::create_dir_all(output_dir)
            .map_err(|e| DocError::IoError(e.to_string()))?;
        
        let entries = std::fs::read_dir(src_dir)
            .map_err(|e| DocError::IoError(e.to_string()))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| DocError::IoError(e.to_string()))?;
            let path = entry.path();
            
            if path.extension().map(|e| e == "zari").unwrap_or(false) {
                let stem = path.file_stem().unwrap().to_string_lossy();
                let output_path = output_dir.join(format!("{}.html", stem));
                
                self.generate(&path, &output_path)?;
                generated.push(output_path);
            } else if path.is_dir() {
                let subdir_output = output_dir.join(path.file_name().unwrap());
                let sub_generated = self.generate_project(&path, &subdir_output)?;
                generated.extend(sub_generated);
            }
        }
        
        let index_path = output_dir.join("index.html");
        let index_content = self.renderer.render_index(&generated)?;
        std::fs::write(&index_path, index_content)
            .map_err(|e| DocError::IoError(e.to_string()))?;
        generated.push(index_path);
        
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_error_display() {
        let err = DocError::IoError("file not found".to_string());
        assert_eq!(err.to_string(), "IO error: file not found");
    }

    #[test]
    fn test_item_kind() {
        assert_eq!(ItemKind::Function, ItemKind::Function);
        assert_ne!(ItemKind::Function, ItemKind::Struct);
    }

    #[test]
    fn test_visibility() {
        assert_eq!(Visibility::Public, Visibility::Public);
    }
}
