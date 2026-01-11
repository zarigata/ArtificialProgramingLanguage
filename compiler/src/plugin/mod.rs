// VeZ Plugin System
// Extensible architecture for adding new language features without core modifications

pub mod api;
pub mod loader;
pub mod registry;
pub mod hooks;

use crate::parser::ast::*;
use crate::error::{Error, ErrorKind, Result};
use std::collections::HashMap;
use std::path::PathBuf;

// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub dependencies: Vec<PluginDependency>,
    pub api_version: String,
    pub capabilities: Vec<PluginCapability>,
}

#[derive(Debug, Clone)]
pub struct PluginDependency {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PluginCapability {
    SyntaxExtension,      // Add new syntax
    TypeSystem,           // Extend type system
    Optimization,         // Add optimization passes
    CodeGeneration,       // Custom code generation
    Analysis,             // Static analysis
    Transformation,       // AST transformation
    Macro,               // Macro expansion
    Linting,             // Code quality checks
    Formatting,          // Code formatting
    Documentation,       // Doc generation
}

// Plugin interface
pub trait Plugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn initialize(&mut self, context: &mut PluginContext) -> Result<()>;
    fn shutdown(&mut self) -> Result<()>;
}

// Syntax extension plugin
pub trait SyntaxPlugin: Plugin {
    fn parse_syntax(&self, input: &str) -> Result<Expr>;
    fn syntax_keywords(&self) -> Vec<String>;
    fn syntax_operators(&self) -> Vec<String>;
}

// Type system plugin
pub trait TypePlugin: Plugin {
    fn define_types(&self) -> Vec<TypeDefinition>;
    fn type_check(&self, expr: &Expr, context: &TypeContext) -> Result<Type>;
    fn type_inference(&self, expr: &Expr, context: &TypeContext) -> Result<Type>;
}

#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub name: String,
    pub kind: TypeKind,
    pub methods: Vec<MethodSignature>,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    Struct(Vec<Field>),
    Enum(Vec<Variant>),
    Trait(Vec<TraitMethod>),
    Alias(Type),
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: String,
    pub signature: MethodSignature,
}

#[derive(Debug, Clone)]
pub struct MethodSignature {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
}

// Optimization plugin
pub trait OptimizationPlugin: Plugin {
    fn optimize(&self, module: &mut Module) -> Result<bool>;
    fn optimization_level(&self) -> OptLevel;
    fn optimization_name(&self) -> String;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
}

// Code generation plugin
pub trait CodegenPlugin: Plugin {
    fn generate_code(&self, module: &Module, target: &Target) -> Result<String>;
    fn supported_targets(&self) -> Vec<Target>;
}

#[derive(Debug, Clone)]
pub struct Target {
    pub arch: String,
    pub os: String,
    pub features: Vec<String>,
}

// Analysis plugin
pub trait AnalysisPlugin: Plugin {
    fn analyze(&self, module: &Module) -> Result<AnalysisReport>;
    fn analysis_name(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct AnalysisReport {
    pub findings: Vec<Finding>,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub message: String,
    pub location: Span,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

// Transformation plugin
pub trait TransformationPlugin: Plugin {
    fn transform(&self, expr: &Expr) -> Result<Expr>;
    fn transform_module(&self, module: &Module) -> Result<Module>;
}

// Macro plugin
pub trait MacroPlugin: Plugin {
    fn expand_macro(&self, name: &str, args: &[Expr]) -> Result<Expr>;
    fn macro_names(&self) -> Vec<String>;
}

// Plugin context
pub struct PluginContext {
    pub compiler_version: String,
    pub target: Target,
    pub config: HashMap<String, String>,
    pub hooks: PluginHooks,
}

impl PluginContext {
    pub fn new() -> Self {
        PluginContext {
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target: Target {
                arch: std::env::consts::ARCH.to_string(),
                os: std::env::consts::OS.to_string(),
                features: Vec::new(),
            },
            config: HashMap::new(),
            hooks: PluginHooks::new(),
        }
    }
    
    pub fn set_config(&mut self, key: String, value: String) {
        self.config.insert(key, value);
    }
    
    pub fn get_config(&self, key: &str) -> Option<&String> {
        self.config.get(key)
    }
}

// Plugin hooks for compiler phases
pub struct PluginHooks {
    pub pre_parse: Vec<Box<dyn Fn(&str) -> Result<String>>>,
    pub post_parse: Vec<Box<dyn Fn(&Module) -> Result<()>>>,
    pub pre_typecheck: Vec<Box<dyn Fn(&Module) -> Result<()>>>,
    pub post_typecheck: Vec<Box<dyn Fn(&Module) -> Result<()>>>,
    pub pre_codegen: Vec<Box<dyn Fn(&Module) -> Result<()>>>,
    pub post_codegen: Vec<Box<dyn Fn(&str) -> Result<()>>>,
}

impl PluginHooks {
    pub fn new() -> Self {
        PluginHooks {
            pre_parse: Vec::new(),
            post_parse: Vec::new(),
            pre_typecheck: Vec::new(),
            post_typecheck: Vec::new(),
            pre_codegen: Vec::new(),
            post_codegen: Vec::new(),
        }
    }
}

// Type context for type plugins
pub struct TypeContext {
    pub types: HashMap<String, Type>,
    pub variables: HashMap<String, Type>,
}

impl TypeContext {
    pub fn new() -> Self {
        TypeContext {
            types: HashMap::new(),
            variables: HashMap::new(),
        }
    }
    
    pub fn add_type(&mut self, name: String, ty: Type) {
        self.types.insert(name, ty);
    }
    
    pub fn get_type(&self, name: &str) -> Option<&Type> {
        self.types.get(name)
    }
}

// Plugin manager
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn Plugin>>,
    load_order: Vec<String>,
}

impl PluginManager {
    pub fn new() -> Self {
        PluginManager {
            plugins: HashMap::new(),
            load_order: Vec::new(),
        }
    }
    
    pub fn register_plugin(&mut self, plugin: Box<dyn Plugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();
        
        // Check dependencies
        for dep in &plugin.metadata().dependencies {
            if !self.plugins.contains_key(&dep.name) {
                return Err(Error::new(
                    ErrorKind::UndefinedSymbol,
                    format!("Missing dependency: {} v{}", dep.name, dep.version),
                ));
            }
        }
        
        self.plugins.insert(name.clone(), plugin);
        self.load_order.push(name);
        
        Ok(())
    }
    
    pub fn initialize_all(&mut self, context: &mut PluginContext) -> Result<()> {
        for name in &self.load_order.clone() {
            if let Some(plugin) = self.plugins.get_mut(name) {
                plugin.initialize(context)?;
            }
        }
        Ok(())
    }
    
    pub fn shutdown_all(&mut self) -> Result<()> {
        for name in self.load_order.iter().rev() {
            if let Some(plugin) = self.plugins.get_mut(name) {
                plugin.shutdown()?;
            }
        }
        Ok(())
    }
    
    pub fn get_plugin(&self, name: &str) -> Option<&dyn Plugin> {
        self.plugins.get(name).map(|p| p.as_ref())
    }
    
    pub fn get_plugins_by_capability(&self, capability: PluginCapability) -> Vec<&dyn Plugin> {
        self.plugins.values()
            .filter(|p| p.metadata().capabilities.contains(&capability))
            .map(|p| p.as_ref())
            .collect()
    }
}

// Module definition for plugins
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Trait(Trait),
    Impl(Impl),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestPlugin {
        metadata: PluginMetadata,
    }
    
    impl Plugin for TestPlugin {
        fn metadata(&self) -> &PluginMetadata {
            &self.metadata
        }
        
        fn initialize(&mut self, _context: &mut PluginContext) -> Result<()> {
            Ok(())
        }
        
        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }
    
    #[test]
    fn test_plugin_manager() {
        let mut manager = PluginManager::new();
        
        let plugin = TestPlugin {
            metadata: PluginMetadata {
                name: "test_plugin".to_string(),
                version: "1.0.0".to_string(),
                author: "Test Author".to_string(),
                description: "Test plugin".to_string(),
                dependencies: Vec::new(),
                api_version: "1.0".to_string(),
                capabilities: vec![PluginCapability::Analysis],
            },
        };
        
        assert!(manager.register_plugin(Box::new(plugin)).is_ok());
        assert!(manager.get_plugin("test_plugin").is_some());
    }
}
