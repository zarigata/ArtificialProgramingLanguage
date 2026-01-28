//! Integration tests for VeZ compiler

use vezc::lexer::Lexer;
use vezc::parser::Parser;
use vezc::semantic::Resolver;
use vezc::semantic::TypeChecker;
use vezc::borrow::BorrowChecker;
use vezc::ir::IrBuilder;

/// Test complete compilation pipeline
fn compile_source(source: &str) -> Result<String, String> {
    // Lexer
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()
        .map_err(|e| format!("Lexer error: {:?}", e))?;
    
    // Parser
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()
        .map_err(|e| format!("Parser error: {:?}", e))?;
    
    // Semantic Analysis
    let resolver = Resolver::new();
    let symbol_table = resolver.resolve(&ast)
        .map_err(|e| format!("Resolver error: {:?}", e))?;
    
    // Type Checking
    let mut type_checker = TypeChecker::new(symbol_table.clone());
    type_checker.check_program(&ast)
        .map_err(|e| format!("Type checker error: {:?}", e))?;
    
    // Borrow Checking
    let mut borrow_checker = BorrowChecker::new(symbol_table);
    borrow_checker.check_program(&ast)
        .map_err(|e| format!("Borrow checker error: {:?}", e))?;
    
    // IR Generation
    let ir_builder = IrBuilder::new("test".to_string());
    let module = ir_builder.build_program(&ast)
        .map_err(|e| format!("IR builder error: {:?}", e))?;
    
    Ok(format!("Successfully compiled! Generated {} functions", module.functions.len()))
}

#[test]
fn test_simple_function() {
    let source = r#"
        fn add(a: i32, b: i32) -> i32 {
            a + b
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Simple function: {}", result.unwrap());
}

#[test]
fn test_factorial() {
    let source = r#"
        fn factorial(n: i32) -> i32 {
            if n <= 1 {
                1
            } else {
                n * factorial(n - 1)
            }
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Factorial: {}", result.unwrap());
}

#[test]
fn test_fibonacci() {
    let source = r#"
        fn fibonacci(n: i32) -> i32 {
            if n <= 1 {
                n
            } else {
                fibonacci(n - 1) + fibonacci(n - 2)
            }
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Fibonacci: {}", result.unwrap());
}

#[test]
fn test_generic_function() {
    let source = r#"
        fn identity<T>(x: T) -> T {
            x
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Generic function: {}", result.unwrap());
}

#[test]
fn test_struct_definition() {
    let source = r#"
        struct Point {
            x: f64,
            y: f64
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Struct definition: {}", result.unwrap());
}

#[test]
fn test_trait_definition() {
    let source = r#"
        trait Display {
            fn display(self) -> String;
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Trait definition: {}", result.unwrap());
}

#[test]
fn test_impl_block() {
    let source = r#"
        struct Point {
            x: f64,
            y: f64
        }
        
        impl Point {
            fn new(x: f64, y: f64) -> Point {
                Point { x, y }
            }
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Impl block: {}", result.unwrap());
}

#[test]
fn test_pattern_matching() {
    let source = r#"
        fn classify(x: i32) -> i32 {
            match x {
                0 => 0,
                1 => 1,
                _ => 2
            }
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Pattern matching: {}", result.unwrap());
}

#[test]
fn test_loops() {
    let source = r#"
        fn sum_to_n(n: i32) -> i32 {
            let mut total = 0;
            let mut i = 0;
            while i < n {
                total = total + i;
                i = i + 1;
            }
            total
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Loops: {}", result.unwrap());
}

#[test]
fn test_complete_program() {
    let source = r#"
        struct Point<T> {
            x: T,
            y: T
        }
        
        impl<T> Point<T> {
            fn new(x: T, y: T) -> Point<T> {
                Point { x, y }
            }
        }
        
        fn factorial(n: i32) -> i32 {
            if n <= 1 {
                1
            } else {
                n * factorial(n - 1)
            }
        }
        
        fn main() {
            let p = Point::new(1, 2);
            let result = factorial(5);
        }
    "#;
    
    let result = compile_source(source);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    println!("✅ Complete program: {}", result.unwrap());
}
