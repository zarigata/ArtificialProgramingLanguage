//! Integration tests for VeZ compiler with new features

use vez_compiler::style_adapters::*;
use vez_compiler::parser::ast::*;

#[test]
fn test_full_pipeline_python_to_native() {
    let python_source = r#"
def add(x: int, y: int) -> int:
    return x + y
"#;
    
    // Parse Python source
    let program = python::parse(python_source);
    assert!(program.is_ok(), "Failed to parse Python source");
    
    let ast = program.unwrap();
    
    // Convert to native VeZ
    let native_source = converter::ast_to_source(&ast, SyntaxStyle::Native);
    assert!(native_source.is_ok(), "Failed to convert to native VeZ");
    
    let output = native_source.unwrap();
    assert!(output.contains("fn add"), "Output should contain function definition");
}

#[test]
fn test_full_pipeline_javascript_to_python() {
    let js_source = r#"
function multiply(a, b) {
    return a * b;
}
"#;
    
    // Parse JavaScript source
    let program = javascript::parse(js_source);
    assert!(program.is_ok(), "Failed to parse JavaScript source");
    
    let ast = program.unwrap();
    
    // Convert to Python
    let python_source = converter::ast_to_source(&ast, SyntaxStyle::Python);
    assert!(python_source.is_ok(), "Failed to convert to Python");
    
    let output = python_source.unwrap();
    assert!(output.contains("def multiply"), "Output should contain function definition");
}

#[test]
fn test_style_detection_and_parsing() {
    let test_cases = vec![
        ("test.pyvez", SyntaxStyle::Python),
        ("test.jsvez", SyntaxStyle::JavaScript),
        ("test.vez", SyntaxStyle::Native),
    ];
    
    for (filename, expected_style) in test_cases {
        let ext = filename.split('.').last().unwrap();
        let detected = SyntaxStyle::from_extension(ext);
        assert_eq!(detected, Some(expected_style), "Failed to detect style for {}", filename);
    }
}

#[test]
fn test_multi_function_python() {
    let source = r#"
def add(x: int, y: int) -> int:
    return x + y

def subtract(x: int, y: int) -> int:
    return x - y

def multiply(x: int, y: int) -> int:
    return x * y
"#;
    
    let result = python::parse(source);
    assert!(result.is_ok());
    
    let program = result.unwrap();
    assert_eq!(program.items.len(), 3, "Should parse three functions");
}

#[test]
fn test_multi_function_javascript() {
    let source = r#"
function add(x, y) {
    return x + y;
}

function subtract(x, y) {
    return x - y;
}

function multiply(x, y) {
    return x * y;
}
"#;
    
    let result = javascript::parse(source);
    assert!(result.is_ok());
    
    let program = result.unwrap();
    assert_eq!(program.items.len(), 3, "Should parse three functions");
}

#[test]
fn test_converter_preserves_semantics() {
    // Create a simple program
    let program = Program {
        items: vec![
            Item::Function(Function {
                name: "test".to_string(),
                attributes: vec![],
                generics: vec![],
                params: vec![
                    Param { 
                        name: "x".to_string(), 
                        ty: Type::Named("i32".to_string()) 
                    },
                ],
                return_type: Some(Type::Named("i32".to_string())),
                where_clause: None,
                body: vec![
                    Stmt::Return(Some(Expr::Binary(
                        Box::new(Expr::Ident("x".to_string())),
                        BinOp::Mul,
                        Box::new(Expr::Literal(Literal::Int(2))),
                    ))),
                ],
            }),
        ],
    };
    
    // Convert to different styles
    let native = converter::ast_to_source(&program, SyntaxStyle::Native);
    let python = converter::ast_to_source(&program, SyntaxStyle::Python);
    let javascript = converter::ast_to_source(&program, SyntaxStyle::JavaScript);
    
    assert!(native.is_ok());
    assert!(python.is_ok());
    assert!(javascript.is_ok());
    
    // All should contain the function name
    assert!(native.unwrap().contains("test"));
    assert!(python.unwrap().contains("test"));
    assert!(javascript.unwrap().contains("test"));
}

#[test]
fn test_complex_ast_conversion() {
    let program = Program {
        items: vec![
            Item::Function(Function {
                name: "complex".to_string(),
                attributes: vec![],
                generics: vec![],
                params: vec![
                    Param { name: "a".to_string(), ty: Type::Named("i32".to_string()) },
                    Param { name: "b".to_string(), ty: Type::Named("i32".to_string()) },
                    Param { name: "c".to_string(), ty: Type::Named("i32".to_string()) },
                ],
                return_type: Some(Type::Named("i32".to_string())),
                where_clause: None,
                body: vec![
                    Stmt::Let(
                        "temp".to_string(),
                        Some(Type::Named("i32".to_string())),
                        Some(Expr::Binary(
                            Box::new(Expr::Ident("a".to_string())),
                            BinOp::Add,
                            Box::new(Expr::Ident("b".to_string())),
                        )),
                    ),
                    Stmt::Return(Some(Expr::Binary(
                        Box::new(Expr::Ident("temp".to_string())),
                        BinOp::Mul,
                        Box::new(Expr::Ident("c".to_string())),
                    ))),
                ],
            }),
        ],
    };
    
    let result = converter::ast_to_source(&program, SyntaxStyle::Native);
    assert!(result.is_ok());
}

#[test]
fn test_error_handling_invalid_python() {
    let invalid_source = r#"
def broken syntax here
"#;
    
    let result = python::parse(invalid_source);
    assert!(result.is_err(), "Should fail on invalid syntax");
}

#[test]
fn test_error_handling_invalid_javascript() {
    let invalid_source = r#"
function broken { syntax }
"#;
    
    let result = javascript::parse(invalid_source);
    assert!(result.is_err(), "Should fail on invalid syntax");
}

#[test]
fn test_parse_with_style_function() {
    let python_source = r#"
def test() -> int:
    return 42
"#;
    
    let result = parse_with_style(python_source, SyntaxStyle::Python);
    assert!(result.is_ok());
}

#[test]
fn test_all_operators_python() {
    let source = r#"
def test_ops(a: int, b: int) -> bool:
    return a + b * 2 - 1 / 3 == 5
"#;
    
    let result = python::parse(source);
    assert!(result.is_ok());
}

#[test]
fn test_all_operators_javascript() {
    let source = r#"
function testOps(a, b) {
    return a + b * 2 - 1 / 3 === 5;
}
"#;
    
    let result = javascript::parse(source);
    assert!(result.is_ok());
}
