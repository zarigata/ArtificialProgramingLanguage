//! Style converter - Convert between different syntax styles

use crate::error::Result;
use crate::parser::ast::*;
use super::SyntaxStyle;

/// Convert VeZ AST to source code in a specific style
pub fn ast_to_source(program: &Program, style: SyntaxStyle) -> Result<String> {
    match style {
        SyntaxStyle::Native => to_native_vez(program),
        SyntaxStyle::Python => to_python(program),
        SyntaxStyle::JavaScript => to_javascript(program),
        SyntaxStyle::Go => to_go(program),
        SyntaxStyle::Cpp => to_cpp(program),
        SyntaxStyle::Ruby => to_ruby(program),
    }
}

fn to_native_vez(program: &Program) -> Result<String> {
    let mut output = String::new();
    
    for item in &program.items {
        output.push_str(&format_item_native(item));
        output.push('\n');
    }
    
    Ok(output)
}

fn format_item_native(item: &Item) -> String {
    match item {
        Item::Function(func) => {
            let mut s = format!("fn {}(", func.name);
            
            for (i, param) in func.params.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}: {}", param.name, format_type(&param.ty)));
            }
            
            s.push(')');
            
            if let Some(ret_ty) = &func.return_type {
                s.push_str(&format!(" -> {}", format_type(ret_ty)));
            }
            
            s.push_str(" {\n");
            
            for stmt in &func.body {
                s.push_str(&format!("    {}\n", format_stmt(stmt)));
            }
            
            s.push_str("}\n");
            s
        }
        Item::Struct(st) => {
            let mut s = format!("struct {} {{\n", st.name);
            
            for field in &st.fields {
                s.push_str(&format!("    {}: {},\n", field.name, format_type(&field.ty)));
            }
            
            s.push_str("}\n");
            s
        }
        _ => String::new(),
    }
}

fn to_python(program: &Program) -> Result<String> {
    let mut output = String::new();
    
    for item in &program.items {
        output.push_str(&format_item_python(item));
        output.push('\n');
    }
    
    Ok(output)
}

fn format_item_python(item: &Item) -> String {
    match item {
        Item::Function(func) => {
            let mut s = format!("def {}(", func.name);
            
            for (i, param) in func.params.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}: {}", param.name, format_type_python(&param.ty)));
            }
            
            s.push(')');
            
            if let Some(ret_ty) = &func.return_type {
                s.push_str(&format!(" -> {}", format_type_python(ret_ty)));
            }
            
            s.push_str(":\n");
            
            for stmt in &func.body {
                s.push_str(&format!("    {}\n", format_stmt_python(stmt)));
            }
            
            s
        }
        Item::Struct(st) => {
            let mut s = format!("class {}:\n", st.name);
            
            if st.fields.is_empty() {
                s.push_str("    pass\n");
            } else {
                s.push_str("    def __init__(self");
                for field in &st.fields {
                    s.push_str(&format!(", {}: {}", field.name, format_type_python(&field.ty)));
                }
                s.push_str("):\n");
                
                for field in &st.fields {
                    s.push_str(&format!("        self.{} = {}\n", field.name, field.name));
                }
            }
            
            s
        }
        _ => String::new(),
    }
}

fn to_javascript(program: &Program) -> Result<String> {
    let mut output = String::new();
    
    for item in &program.items {
        output.push_str(&format_item_javascript(item));
        output.push('\n');
    }
    
    Ok(output)
}

fn format_item_javascript(item: &Item) -> String {
    match item {
        Item::Function(func) => {
            let mut s = format!("function {}(", func.name);
            
            for (i, param) in func.params.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&param.name);
            }
            
            s.push_str(") {\n");
            
            for stmt in &func.body {
                s.push_str(&format!("    {};\n", format_stmt_javascript(stmt)));
            }
            
            s.push_str("}\n");
            s
        }
        Item::Struct(st) => {
            let mut s = format!("class {} {{\n", st.name);
            
            s.push_str("    constructor(");
            for (i, field) in st.fields.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&field.name);
            }
            s.push_str(") {\n");
            
            for field in &st.fields {
                s.push_str(&format!("        this.{} = {};\n", field.name, field.name));
            }
            
            s.push_str("    }\n}\n");
            s
        }
        _ => String::new(),
    }
}

fn to_go(_program: &Program) -> Result<String> {
    Ok("// Go-style output not yet implemented\n".to_string())
}

fn to_cpp(_program: &Program) -> Result<String> {
    Ok("// C++-style output not yet implemented\n".to_string())
}

fn to_ruby(_program: &Program) -> Result<String> {
    Ok("# Ruby-style output not yet implemented\n".to_string())
}

fn format_type(ty: &Type) -> String {
    match ty {
        Type::Named(name) => name.clone(),
        Type::Generic(name, args) => {
            let args_str = args.iter()
                .map(|t| format_type(t))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}<{}>", name, args_str)
        }
        Type::Reference(inner) => format!("&{}", format_type(inner)),
        Type::MutableReference(inner) => format!("&mut {}", format_type(inner)),
        Type::Array(inner, size) => format!("[{}; {}]", format_type(inner), size),
        Type::Tuple(types) => {
            let types_str = types.iter()
                .map(|t| format_type(t))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({})", types_str)
        }
        _ => "unknown".to_string(),
    }
}

fn format_type_python(ty: &Type) -> String {
    match ty {
        Type::Named(name) => {
            match name.as_str() {
                "i32" | "i64" | "u32" | "u64" => "int".to_string(),
                "f32" | "f64" => "float".to_string(),
                "bool" => "bool".to_string(),
                "String" => "str".to_string(),
                _ => name.clone(),
            }
        }
        Type::Generic(name, args) => {
            let args_str = args.iter()
                .map(|t| format_type_python(t))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}[{}]", name, args_str)
        }
        _ => "Any".to_string(),
    }
}

fn format_stmt(stmt: &Stmt) -> String {
    match stmt {
        Stmt::Return(Some(expr)) => format!("return {};", format_expr(expr)),
        Stmt::Return(None) => "return;".to_string(),
        Stmt::Let(name, ty, init) => {
            let mut s = format!("let {}", name);
            if let Some(t) = ty {
                s.push_str(&format!(": {}", format_type(t)));
            }
            if let Some(e) = init {
                s.push_str(&format!(" = {}", format_expr(e)));
            }
            s.push(';');
            s
        }
        Stmt::Expr(expr) => format!("{};", format_expr(expr)),
    }
}

fn format_stmt_python(stmt: &Stmt) -> String {
    match stmt {
        Stmt::Return(Some(expr)) => format!("return {}", format_expr(expr)),
        Stmt::Return(None) => "return".to_string(),
        Stmt::Let(name, _, init) => {
            if let Some(e) = init {
                format!("{} = {}", name, format_expr(expr))
            } else {
                format!("{} = None", name)
            }
        }
        Stmt::Expr(expr) => format_expr(expr),
    }
}

fn format_stmt_javascript(stmt: &Stmt) -> String {
    match stmt {
        Stmt::Return(Some(expr)) => format!("return {}", format_expr(expr)),
        Stmt::Return(None) => "return".to_string(),
        Stmt::Let(name, _, init) => {
            if let Some(e) = init {
                format!("const {} = {}", name, format_expr(e))
            } else {
                format!("let {}", name)
            }
        }
        Stmt::Expr(expr) => format_expr(expr),
    }
}

fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Literal(lit) => format_literal(lit),
        Expr::Ident(name) => name.clone(),
        Expr::Binary(left, op, right) => {
            format!("{} {} {}", format_expr(left), format_binop(op), format_expr(right))
        }
        Expr::Unary(op, expr) => {
            format!("{}{}", format_unop(op), format_expr(expr))
        }
        Expr::Call(func, args) => {
            let args_str = args.iter()
                .map(|a| format_expr(a))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", format_expr(func), args_str)
        }
        Expr::Field(obj, field) => format!("{}.{}", format_expr(obj), field),
        Expr::Index(arr, idx) => format!("{}[{}]", format_expr(arr), format_expr(idx)),
        Expr::Array(elements) => {
            let elems_str = elements.iter()
                .map(|e| format_expr(e))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{}]", elems_str)
        }
        _ => "...".to_string(),
    }
}

fn format_literal(lit: &Literal) -> String {
    match lit {
        Literal::Int(n) => n.to_string(),
        Literal::Float(f) => f.to_string(),
        Literal::String(s) => format!("\"{}\"", s),
        Literal::Char(c) => format!("'{}'", c),
        Literal::Bool(b) => b.to_string(),
    }
}

fn format_binop(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

fn format_unop(op: &UnOp) -> &'static str {
    match op {
        UnOp::Neg => "-",
        UnOp::Not => "!",
        UnOp::Deref => "*",
        UnOp::Ref => "&",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_vez_output() {
        let program = Program {
            items: vec![
                Item::Function(Function {
                    name: "add".to_string(),
                    attributes: vec![],
                    generics: vec![],
                    params: vec![
                        Param { name: "x".to_string(), ty: Type::Named("i32".to_string()) },
                        Param { name: "y".to_string(), ty: Type::Named("i32".to_string()) },
                    ],
                    return_type: Some(Type::Named("i32".to_string())),
                    where_clause: None,
                    body: vec![
                        Stmt::Return(Some(Expr::Binary(
                            Box::new(Expr::Ident("x".to_string())),
                            BinOp::Add,
                            Box::new(Expr::Ident("y".to_string())),
                        ))),
                    ],
                }),
            ],
        };

        let result = ast_to_source(&program, SyntaxStyle::Native);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("fn add"));
        assert!(output.contains("return x + y"));
    }

    #[test]
    fn test_python_output() {
        let program = Program {
            items: vec![
                Item::Function(Function {
                    name: "add".to_string(),
                    attributes: vec![],
                    generics: vec![],
                    params: vec![
                        Param { name: "x".to_string(), ty: Type::Named("i32".to_string()) },
                        Param { name: "y".to_string(), ty: Type::Named("i32".to_string()) },
                    ],
                    return_type: Some(Type::Named("i32".to_string())),
                    where_clause: None,
                    body: vec![
                        Stmt::Return(Some(Expr::Binary(
                            Box::new(Expr::Ident("x".to_string())),
                            BinOp::Add,
                            Box::new(Expr::Ident("y".to_string())),
                        ))),
                    ],
                }),
            ],
        };

        let result = ast_to_source(&program, SyntaxStyle::Python);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("def add"));
        assert!(output.contains("return x + y"));
    }
}
