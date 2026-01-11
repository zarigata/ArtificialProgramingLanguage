// Plugin API
// High-level API for plugin development

use super::*;

// Plugin SDK - Simplified API for plugin developers
pub struct PluginSDK {
    context: PluginContext,
}

impl PluginSDK {
    pub fn new(context: PluginContext) -> Self {
        PluginSDK { context }
    }
    
    // Compiler information
    pub fn compiler_version(&self) -> &str {
        &self.context.compiler_version
    }
    
    pub fn target_arch(&self) -> &str {
        &self.context.target.arch
    }
    
    pub fn target_os(&self) -> &str {
        &self.context.target.os
    }
    
    // Configuration
    pub fn get_config(&self, key: &str) -> Option<&String> {
        self.context.get_config(key)
    }
    
    // Logging
    pub fn log_info(&self, message: &str) {
        println!("[INFO] {}", message);
    }
    
    pub fn log_warning(&self, message: &str) {
        eprintln!("[WARN] {}", message);
    }
    
    pub fn log_error(&self, message: &str) {
        eprintln!("[ERROR] {}", message);
    }
    
    // AST utilities
    pub fn create_literal_int(&self, value: i64) -> Expr {
        Expr::Literal {
            value: Literal::Int(value),
            span: Span::dummy(),
        }
    }
    
    pub fn create_literal_string(&self, value: String) -> Expr {
        Expr::Literal {
            value: Literal::String(value),
            span: Span::dummy(),
        }
    }
    
    pub fn create_variable(&self, name: String) -> Expr {
        Expr::Variable {
            name,
            span: Span::dummy(),
        }
    }
    
    pub fn create_binary_op(&self, op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span: Span::dummy(),
        }
    }
    
    pub fn create_function_call(&self, func: Expr, args: Vec<Expr>) -> Expr {
        Expr::Call {
            func: Box::new(func),
            args,
            span: Span::dummy(),
        }
    }
}

// Macro for creating plugins easily
#[macro_export]
macro_rules! vez_plugin {
    (
        name: $name:expr,
        version: $version:expr,
        author: $author:expr,
        description: $description:expr,
        capabilities: [$($cap:expr),*],
        
        initialize: $init:expr,
        shutdown: $shutdown:expr
    ) => {
        {
            struct GeneratedPlugin {
                metadata: PluginMetadata,
                init_fn: Box<dyn Fn(&mut PluginContext) -> Result<()>>,
                shutdown_fn: Box<dyn Fn() -> Result<()>>,
            }
            
            impl Plugin for GeneratedPlugin {
                fn metadata(&self) -> &PluginMetadata {
                    &self.metadata
                }
                
                fn initialize(&mut self, context: &mut PluginContext) -> Result<()> {
                    (self.init_fn)(context)
                }
                
                fn shutdown(&mut self) -> Result<()> {
                    (self.shutdown_fn)()
                }
            }
            
            GeneratedPlugin {
                metadata: PluginMetadata {
                    name: $name.to_string(),
                    version: $version.to_string(),
                    author: $author.to_string(),
                    description: $description.to_string(),
                    dependencies: Vec::new(),
                    api_version: "1.0".to_string(),
                    capabilities: vec![$($cap),*],
                },
                init_fn: Box::new($init),
                shutdown_fn: Box::new($shutdown),
            }
        }
    };
}

// Helper functions for common plugin tasks
pub mod helpers {
    use super::*;
    
    // AST traversal
    pub fn visit_expr<F>(expr: &Expr, visitor: &mut F)
    where
        F: FnMut(&Expr),
    {
        visitor(expr);
        
        match expr {
            Expr::Binary { left, right, .. } => {
                visit_expr(left, visitor);
                visit_expr(right, visitor);
            }
            Expr::Unary { operand, .. } => {
                visit_expr(operand, visitor);
            }
            Expr::Call { func, args, .. } => {
                visit_expr(func, visitor);
                for arg in args {
                    visit_expr(arg, visitor);
                }
            }
            Expr::Block { stmts, .. } => {
                for stmt in stmts {
                    visit_stmt(stmt, visitor);
                }
            }
            _ => {}
        }
    }
    
    pub fn visit_stmt<F>(stmt: &Stmt, visitor: &mut F)
    where
        F: FnMut(&Expr),
    {
        match stmt {
            Stmt::Expr(expr) => visit_expr(expr, visitor),
            Stmt::Let { init, .. } => {
                if let Some(init_expr) = init {
                    visit_expr(init_expr, visitor);
                }
            }
            _ => {}
        }
    }
    
    // Type utilities
    pub fn is_numeric_type(ty: &Type) -> bool {
        matches!(
            ty,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 |
            Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128 |
            Type::F32 | Type::F64
        )
    }
    
    pub fn is_integer_type(ty: &Type) -> bool {
        matches!(
            ty,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 |
            Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128
        )
    }
    
    pub fn is_float_type(ty: &Type) -> bool {
        matches!(ty, Type::F32 | Type::F64)
    }
    
    // Code generation utilities
    pub fn indent(code: &str, level: usize) -> String {
        let indent_str = "    ".repeat(level);
        code.lines()
            .map(|line| format!("{}{}", indent_str, line))
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    pub fn format_type(ty: &Type) -> String {
        match ty {
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::String => "String".to_string(),
            _ => "unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_sdk() {
        let context = PluginContext::new();
        let sdk = PluginSDK::new(context);
        
        assert!(!sdk.compiler_version().is_empty());
        assert!(!sdk.target_arch().is_empty());
    }
    
    #[test]
    fn test_ast_creation() {
        let context = PluginContext::new();
        let sdk = PluginSDK::new(context);
        
        let expr = sdk.create_literal_int(42);
        match expr {
            Expr::Literal { value: Literal::Int(n), .. } => assert_eq!(n, 42),
            _ => panic!("Expected integer literal"),
        }
    }
}
