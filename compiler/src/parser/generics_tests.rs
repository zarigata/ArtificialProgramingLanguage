//! Tests for generics, traits, and implementations

use super::*;
use crate::lexer::Lexer;

fn parse_source(source: &str) -> Result<Program> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod generic_tests {
    use super::*;

    #[test]
    fn test_generic_function() {
        let source = "fn identity<T>(x: T) -> T { x }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_function_multiple_params() {
        let source = "fn pair<T, U>(x: T, y: U) -> (T, U) { (x, y) }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_function_with_bounds() {
        let source = "fn print<T: Display>(x: T) { }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_function_multiple_bounds() {
        let source = "fn complex<T: Display + Clone>(x: T) { }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_struct() {
        let source = "struct Box<T> { value: T }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_struct_multiple_params() {
        let source = "struct Pair<T, U> { first: T, second: U }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_enum() {
        let source = "enum Option<T> { Some(T), None }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_where_clause() {
        let source = r#"
            fn complex<T, U>(x: T, y: U) 
            where T: Display, U: Clone {
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_where_clause_multiple_bounds() {
        let source = r#"
            fn complex<T>(x: T) 
            where T: Display + Clone + Debug {
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    #[test]
    fn test_simple_trait() {
        let source = r#"
            trait Display {
                fn display(self) -> String;
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_trait_with_multiple_methods() {
        let source = r#"
            trait Shape {
                fn area(self) -> f64;
                fn perimeter(self) -> f64;
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_trait_with_supertrait() {
        let source = r#"
            trait Drawable: Display {
                fn draw(self);
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_trait_with_multiple_supertraits() {
        let source = r#"
            trait Complex: Display + Clone + Debug {
                fn complex_method(self);
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_trait() {
        let source = r#"
            trait Container<T> {
                fn get(self) -> T;
                fn set(self, value: T);
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_trait_with_associated_type() {
        let source = r#"
            trait Iterator {
                type Item;
                fn next(self) -> Item;
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod impl_tests {
    use super::*;

    #[test]
    fn test_simple_impl() {
        let source = r#"
            impl Point {
                fn new(x: f64, y: f64) -> Point { }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_trait_impl() {
        let source = r#"
            impl Display for Point {
                fn display(self) -> String { }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_impl() {
        let source = r#"
            impl<T> Box<T> {
                fn new(value: T) -> Box<T> { }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_generic_trait_impl() {
        let source = r#"
            impl<T: Display> Display for Box<T> {
                fn display(self) -> String { }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_impl_with_where_clause() {
        let source = r#"
            impl<T> Display for Box<T> 
            where T: Display {
                fn display(self) -> String { }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_impl_with_associated_type() {
        let source = r#"
            impl Iterator for MyIterator {
                type Item = i32;
                fn next(self) -> Item { }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod type_tests {
    use super::*;

    #[test]
    fn test_generic_type() {
        let source = "fn main() { let x: Vec<i32>; }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_nested_generic_type() {
        let source = "fn main() { let x: Vec<Vec<i32>>; }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_multiple_type_args() {
        let source = "fn main() { let x: HashMap<String, i32>; }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_mutable_reference() {
        let source = "fn main() { let x: &mut i32; }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_tuple_type() {
        let source = "fn main() { let x: (i32, String, f64); }";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod use_tests {
    use super::*;

    #[test]
    fn test_simple_use() {
        let source = "use std::io;";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_nested_use() {
        let source = "use std::collections::HashMap;";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_use_with_alias() {
        let source = "use std::io as stdio;";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod mod_tests {
    use super::*;

    #[test]
    fn test_mod_declaration() {
        let source = "mod utils;";
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_inline_mod() {
        let source = r#"
            mod utils {
                fn helper() { }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_nested_mods() {
        let source = r#"
            mod outer {
                mod inner {
                    fn helper() { }
                }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 1);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_generic_program() {
        let source = r#"
            trait Display {
                fn display(self) -> String;
            }
            
            struct Point<T> {
                x: T,
                y: T
            }
            
            impl<T: Display> Display for Point<T> {
                fn display(self) -> String {
                    return "Point";
                }
            }
            
            fn main() {
                let p: Point<f64> = Point { x: 1.0, y: 2.0 };
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 4);
    }

    #[test]
    fn test_option_enum() {
        let source = r#"
            enum Option<T> {
                Some(T),
                None
            }
            
            impl<T> Option<T> {
                fn is_some(self) -> bool {
                    match self {
                        Some(_) => true,
                        None => false
                    }
                }
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 2);
    }

    #[test]
    fn test_result_enum() {
        let source = r#"
            enum Result<T, E> {
                Ok(T),
                Err(E)
            }
            
            fn divide(a: i32, b: i32) -> Result<i32, String> {
                if b == 0 {
                    return Err("Division by zero");
                }
                return Ok(a / b);
            }
        "#;
        let program = parse_source(source).unwrap();
        assert_eq!(program.items.len(), 2);
    }
}
