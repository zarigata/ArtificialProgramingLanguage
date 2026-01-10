//! Comprehensive parser tests

use super::*;
use crate::lexer::Lexer;

fn parse_source(source: &str) -> Result<Program> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod expression_tests {
    use super::*;

    #[test]
    fn test_literal_expressions() {
        let program = parse_source("fn main() { 42; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_binary_expressions() {
        let program = parse_source("fn main() { 1 + 2; }").unwrap();
        assert_eq!(program.items.len(), 1);
        
        let program = parse_source("fn main() { 1 + 2 * 3; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_unary_expressions() {
        let program = parse_source("fn main() { -42; }").unwrap();
        assert_eq!(program.items.len(), 1);
        
        let program = parse_source("fn main() { !true; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_function_calls() {
        let program = parse_source("fn main() { foo(); }").unwrap();
        assert_eq!(program.items.len(), 1);
        
        let program = parse_source("fn main() { foo(1, 2, 3); }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_method_calls() {
        let program = parse_source("fn main() { obj.method(); }").unwrap();
        assert_eq!(program.items.len(), 1);
        
        let program = parse_source("fn main() { obj.method(1, 2); }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_field_access() {
        let program = parse_source("fn main() { obj.field; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_array_indexing() {
        let program = parse_source("fn main() { arr[0]; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_array_literals() {
        let program = parse_source("fn main() { [1, 2, 3]; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_parenthesized_expressions() {
        let program = parse_source("fn main() { (1 + 2) * 3; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod control_flow_tests {
    use super::*;

    #[test]
    fn test_if_expression() {
        let program = parse_source("fn main() { if true { 1 } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_if_else_expression() {
        let program = parse_source("fn main() { if true { 1 } else { 2 } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_if_else_if_chain() {
        let program = parse_source(
            "fn main() { if x { 1 } else if y { 2 } else { 3 } }"
        ).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_match_expression() {
        let program = parse_source(
            "fn main() { match x { 1 => 2, _ => 3 } }"
        ).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_loop_expression() {
        let program = parse_source("fn main() { loop { break; } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_while_expression() {
        let program = parse_source("fn main() { while true { break; } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_for_expression() {
        let program = parse_source("fn main() { for i in 0..10 { } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_break_continue() {
        let program = parse_source("fn main() { loop { break; } }").unwrap();
        assert_eq!(program.items.len(), 1);
        
        let program = parse_source("fn main() { loop { continue; } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_return_expression() {
        let program = parse_source("fn main() { return 42; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod statement_tests {
    use super::*;

    #[test]
    fn test_let_binding() {
        let program = parse_source("fn main() { let x = 42; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_let_with_type() {
        let program = parse_source("fn main() { let x: i32 = 42; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_let_without_init() {
        let program = parse_source("fn main() { let x: i32; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_expression_statement() {
        let program = parse_source("fn main() { 42; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_return_statement() {
        let program = parse_source("fn main() { return 42; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod declaration_tests {
    use super::*;

    #[test]
    fn test_empty_function() {
        let program = parse_source("fn main() { }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_function_with_params() {
        let program = parse_source("fn add(a: i32, b: i32) { }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_function_with_return_type() {
        let program = parse_source("fn add(a: i32, b: i32) -> i32 { }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_empty_struct() {
        let program = parse_source("struct Point { }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_struct_with_fields() {
        let program = parse_source("struct Point { x: f64, y: f64 }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_multiple_items() {
        let program = parse_source(
            "fn foo() { } struct Bar { } fn baz() { }"
        ).unwrap();
        assert_eq!(program.items.len(), 3);
    }
}

#[cfg(test)]
mod type_tests {
    use super::*;

    #[test]
    fn test_named_type() {
        let program = parse_source("fn main() { let x: i32; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_reference_type() {
        let program = parse_source("fn main() { let x: &i32; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_array_type() {
        let program = parse_source("fn main() { let x: [i32; 10]; }").unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod pattern_tests {
    use super::*;

    #[test]
    fn test_wildcard_pattern() {
        let program = parse_source("fn main() { match x { _ => 1 } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_literal_pattern() {
        let program = parse_source("fn main() { match x { 42 => 1 } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_ident_pattern() {
        let program = parse_source("fn main() { match x { y => 1 } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_tuple_pattern() {
        let program = parse_source("fn main() { match x { (a, b) => 1 } }").unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        let source = r#"
            fn fibonacci(n: u32) -> u32 {
                if n <= 1 {
                    n
                } else {
                    fibonacci(n - 1) + fibonacci(n - 2)
                }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_struct_with_methods() {
        let source = r#"
            struct Point {
                x: f64,
                y: f64
            }
            
            fn distance(p1: Point, p2: Point) -> f64 {
                let dx = p1.x - p2.x;
                let dy = p1.y - p2.y;
                return dx * dx + dy * dy;
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 2);
    }

    #[test]
    fn test_complex_expression() {
        let source = r#"
            fn main() {
                let result = (a + b) * c - d / e;
                let array = [1, 2, 3, 4, 5];
                let first = array[0];
                let method_result = obj.method(1, 2);
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_nested_control_flow() {
        let source = r#"
            fn main() {
                for i in 0..10 {
                    if i % 2 == 0 {
                        continue;
                    } else {
                        while true {
                            break;
                        }
                    }
                }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_match_with_guards() {
        let source = r#"
            fn main() {
                match value {
                    x if x > 0 => 1,
                    x if x < 0 => -1,
                    _ => 0
                }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_missing_semicolon() {
        // This should still parse (expression statement)
        let result = parse_source("fn main() { 42 }");
        assert!(result.is_ok());
    }

    #[test]
    fn test_unexpected_token() {
        let result = parse_source("fn main() { @ }");
        assert!(result.is_err());
    }

    #[test]
    fn test_unmatched_paren() {
        let result = parse_source("fn main() { (1 + 2 }");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_function() {
        let result = parse_source("fn { }");
        assert!(result.is_err());
    }
}
