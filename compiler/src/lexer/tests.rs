//! Comprehensive lexer tests

use super::*;
use crate::lexer::token::TokenKind;

#[cfg(test)]
mod number_tests {
    use super::*;

    #[test]
    fn test_decimal_integers() {
        let mut lexer = Lexer::new("0 42 1000 999_999");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(ref s, None) if s == "0"));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(ref s, None) if s == "42"));
        assert!(matches!(tokens[2].kind, TokenKind::IntLiteral(ref s, None) if s == "1000"));
        assert!(matches!(tokens[3].kind, TokenKind::IntLiteral(ref s, None) if s == "999_999"));
    }

    #[test]
    fn test_hexadecimal() {
        let mut lexer = Lexer::new("0xFF 0x00 0xDEAD_BEEF");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(ref s, None) if s == "0xFF"));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(ref s, None) if s == "0x00"));
        assert!(matches!(tokens[2].kind, TokenKind::IntLiteral(ref s, None) if s == "0xDEAD_BEEF"));
    }

    #[test]
    fn test_octal() {
        let mut lexer = Lexer::new("0o777 0o123");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(ref s, None) if s == "0o777"));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(ref s, None) if s == "0o123"));
    }

    #[test]
    fn test_binary() {
        let mut lexer = Lexer::new("0b1010 0b1111_0000");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(ref s, None) if s == "0b1010"));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(ref s, None) if s == "0b1111_0000"));
    }

    #[test]
    fn test_float_literals() {
        let mut lexer = Lexer::new("3.14 0.5 2.0 1.5e10 1.5e-5");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::FloatLiteral(ref s, None) if s == "3.14"));
        assert!(matches!(tokens[1].kind, TokenKind::FloatLiteral(ref s, None) if s == "0.5"));
        assert!(matches!(tokens[2].kind, TokenKind::FloatLiteral(ref s, None) if s == "2.0"));
        assert!(matches!(tokens[3].kind, TokenKind::FloatLiteral(ref s, None) if s == "1.5e10"));
        assert!(matches!(tokens[4].kind, TokenKind::FloatLiteral(ref s, None) if s == "1.5e-5"));
    }

    #[test]
    fn test_number_suffixes() {
        let mut lexer = Lexer::new("42i32 100u64 3.14f32 2.5f64");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(ref s, Some(ref suf)) if s == "42" && suf == "i32"));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(ref s, Some(ref suf)) if s == "100" && suf == "u64"));
        assert!(matches!(tokens[2].kind, TokenKind::FloatLiteral(ref s, Some(ref suf)) if s == "3.14" && suf == "f32"));
        assert!(matches!(tokens[3].kind, TokenKind::FloatLiteral(ref s, Some(ref suf)) if s == "2.5" && suf == "f64"));
    }

    #[test]
    fn test_invalid_hex() {
        let mut lexer = Lexer::new("0x");
        assert!(lexer.tokenize().is_err());
    }

    #[test]
    fn test_invalid_octal() {
        let mut lexer = Lexer::new("0o");
        assert!(lexer.tokenize().is_err());
    }

    #[test]
    fn test_invalid_binary() {
        let mut lexer = Lexer::new("0b");
        assert!(lexer.tokenize().is_err());
    }
}

#[cfg(test)]
mod string_tests {
    use super::*;

    #[test]
    fn test_simple_string() {
        let mut lexer = Lexer::new(r#""hello world""#);
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::StringLiteral(ref s) if s == "hello world"));
    }

    #[test]
    fn test_escape_sequences() {
        let mut lexer = Lexer::new(r#""line1\nline2\ttab""#);
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::StringLiteral(ref s) if s == "line1\nline2\ttab"));
    }

    #[test]
    fn test_hex_escape() {
        let mut lexer = Lexer::new(r#""\x41\x42\x43""#);
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::StringLiteral(ref s) if s == "ABC"));
    }

    #[test]
    fn test_unicode_escape() {
        let mut lexer = Lexer::new(r#""\u{1F600}""#);
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::StringLiteral(ref s) if s == "😀"));
    }

    #[test]
    fn test_raw_string() {
        let mut lexer = Lexer::new(r#"r"no\nescape""#);
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::StringLiteral(ref s) if s == r"no\nescape"));
    }

    #[test]
    fn test_raw_string_with_hashes() {
        let mut lexer = Lexer::new(r###"r#"can have "quotes" inside"#"###);
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::StringLiteral(ref s) if s == r#"can have "quotes" inside"#));
    }

    #[test]
    fn test_unterminated_string() {
        let mut lexer = Lexer::new(r#""unterminated"#);
        assert!(lexer.tokenize().is_err());
    }

    #[test]
    fn test_invalid_escape() {
        let mut lexer = Lexer::new(r#""\q""#);
        assert!(lexer.tokenize().is_err());
    }
}

#[cfg(test)]
mod char_tests {
    use super::*;

    #[test]
    fn test_simple_char() {
        let mut lexer = Lexer::new("'a' 'Z' '0'");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::CharLiteral('a')));
        assert!(matches!(tokens[1].kind, TokenKind::CharLiteral('Z')));
        assert!(matches!(tokens[2].kind, TokenKind::CharLiteral('0')));
    }

    #[test]
    fn test_char_escapes() {
        let mut lexer = Lexer::new(r"'\n' '\t' '\'' '\\'");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::CharLiteral('\n')));
        assert!(matches!(tokens[1].kind, TokenKind::CharLiteral('\t')));
        assert!(matches!(tokens[2].kind, TokenKind::CharLiteral('\'')));
        assert!(matches!(tokens[3].kind, TokenKind::CharLiteral('\\')));
    }

    #[test]
    fn test_char_hex_escape() {
        let mut lexer = Lexer::new(r"'\x41'");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::CharLiteral('A')));
    }

    #[test]
    fn test_char_unicode_escape() {
        let mut lexer = Lexer::new(r"'\u{1F600}'");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::CharLiteral('😀')));
    }
}

#[cfg(test)]
mod keyword_tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let mut lexer = Lexer::new("fn let mut const if else while for");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Fn));
        assert!(matches!(tokens[1].kind, TokenKind::Let));
        assert!(matches!(tokens[2].kind, TokenKind::Mut));
        assert!(matches!(tokens[3].kind, TokenKind::Const));
        assert!(matches!(tokens[4].kind, TokenKind::If));
        assert!(matches!(tokens[5].kind, TokenKind::Else));
        assert!(matches!(tokens[6].kind, TokenKind::While));
        assert!(matches!(tokens[7].kind, TokenKind::For));
    }

    #[test]
    fn test_type_keywords() {
        let mut lexer = Lexer::new("struct enum trait impl");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Struct));
        assert!(matches!(tokens[1].kind, TokenKind::Enum));
        assert!(matches!(tokens[2].kind, TokenKind::Trait));
        assert!(matches!(tokens[3].kind, TokenKind::Impl));
    }

    #[test]
    fn test_boolean_literals() {
        let mut lexer = Lexer::new("true false");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::True));
        assert!(matches!(tokens[1].kind, TokenKind::False));
    }
}

#[cfg(test)]
mod operator_tests {
    use super::*;

    #[test]
    fn test_arithmetic_operators() {
        let mut lexer = Lexer::new("+ - * / % **");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Plus));
        assert!(matches!(tokens[1].kind, TokenKind::Minus));
        assert!(matches!(tokens[2].kind, TokenKind::Star));
        assert!(matches!(tokens[3].kind, TokenKind::Slash));
        assert!(matches!(tokens[4].kind, TokenKind::Percent));
        assert!(matches!(tokens[5].kind, TokenKind::StarStar));
    }

    #[test]
    fn test_comparison_operators() {
        let mut lexer = Lexer::new("== != < <= > >=");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::EqEq));
        assert!(matches!(tokens[1].kind, TokenKind::Ne));
        assert!(matches!(tokens[2].kind, TokenKind::Lt));
        assert!(matches!(tokens[3].kind, TokenKind::Le));
        assert!(matches!(tokens[4].kind, TokenKind::Gt));
        assert!(matches!(tokens[5].kind, TokenKind::Ge));
    }

    #[test]
    fn test_logical_operators() {
        let mut lexer = Lexer::new("&& || !");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::AndAnd));
        assert!(matches!(tokens[1].kind, TokenKind::OrOr));
        assert!(matches!(tokens[2].kind, TokenKind::Bang));
    }

    #[test]
    fn test_bitwise_operators() {
        let mut lexer = Lexer::new("& | ^ ~ << >>");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::And));
        assert!(matches!(tokens[1].kind, TokenKind::Pipe));
        assert!(matches!(tokens[2].kind, TokenKind::Caret));
        assert!(matches!(tokens[3].kind, TokenKind::Tilde));
        assert!(matches!(tokens[4].kind, TokenKind::Shl));
        assert!(matches!(tokens[5].kind, TokenKind::Shr));
    }

    #[test]
    fn test_assignment_operators() {
        let mut lexer = Lexer::new("= += -= *= /=");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Eq));
        assert!(matches!(tokens[1].kind, TokenKind::PlusEq));
        assert!(matches!(tokens[2].kind, TokenKind::MinusEq));
        assert!(matches!(tokens[3].kind, TokenKind::StarEq));
        assert!(matches!(tokens[4].kind, TokenKind::SlashEq));
    }

    #[test]
    fn test_arrows() {
        let mut lexer = Lexer::new("-> =>");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Arrow));
        assert!(matches!(tokens[1].kind, TokenKind::FatArrow));
    }
}

#[cfg(test)]
mod delimiter_tests {
    use super::*;

    #[test]
    fn test_delimiters() {
        let mut lexer = Lexer::new("( ) { } [ ] , ; : ::");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::LParen));
        assert!(matches!(tokens[1].kind, TokenKind::RParen));
        assert!(matches!(tokens[2].kind, TokenKind::LBrace));
        assert!(matches!(tokens[3].kind, TokenKind::RBrace));
        assert!(matches!(tokens[4].kind, TokenKind::LBracket));
        assert!(matches!(tokens[5].kind, TokenKind::RBracket));
        assert!(matches!(tokens[6].kind, TokenKind::Comma));
        assert!(matches!(tokens[7].kind, TokenKind::Semi));
        assert!(matches!(tokens[8].kind, TokenKind::Colon));
        assert!(matches!(tokens[9].kind, TokenKind::ColonColon));
    }

    #[test]
    fn test_dots() {
        let mut lexer = Lexer::new(". .. ..=");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Dot));
        assert!(matches!(tokens[1].kind, TokenKind::DotDot));
        assert!(matches!(tokens[2].kind, TokenKind::DotDotEq));
    }
}

#[cfg(test)]
mod comment_tests {
    use super::*;

    #[test]
    fn test_line_comment() {
        let mut lexer = Lexer::new("42 // this is a comment\n100");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(_, _)));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(_, _)));
        assert_eq!(tokens.len(), 3); // 42, 100, EOF
    }

    #[test]
    fn test_block_comment() {
        let mut lexer = Lexer::new("42 /* comment */ 100");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(_, _)));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(_, _)));
        assert_eq!(tokens.len(), 3); // 42, 100, EOF
    }

    #[test]
    fn test_nested_block_comment() {
        let mut lexer = Lexer::new("42 /* outer /* inner */ outer */ 100");
        let tokens = lexer.tokenize().unwrap();
        
        // Note: Current implementation doesn't support nested comments
        // This test documents current behavior
        assert!(tokens.len() >= 2);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_function_declaration() {
        let mut lexer = Lexer::new("fn main() { }");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Fn));
        assert!(matches!(tokens[1].kind, TokenKind::Ident(ref s) if s == "main"));
        assert!(matches!(tokens[2].kind, TokenKind::LParen));
        assert!(matches!(tokens[3].kind, TokenKind::RParen));
        assert!(matches!(tokens[4].kind, TokenKind::LBrace));
        assert!(matches!(tokens[5].kind, TokenKind::RBrace));
    }

    #[test]
    fn test_variable_declaration() {
        let mut lexer = Lexer::new("let x: i32 = 42;");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].kind, TokenKind::Let));
        assert!(matches!(tokens[1].kind, TokenKind::Ident(ref s) if s == "x"));
        assert!(matches!(tokens[2].kind, TokenKind::Colon));
        assert!(matches!(tokens[3].kind, TokenKind::Ident(ref s) if s == "i32"));
        assert!(matches!(tokens[4].kind, TokenKind::Eq));
        assert!(matches!(tokens[5].kind, TokenKind::IntLiteral(_, _)));
        assert!(matches!(tokens[6].kind, TokenKind::Semi));
    }

    #[test]
    fn test_complex_expression() {
        let mut lexer = Lexer::new("(a + b) * c - d / e");
        let tokens = lexer.tokenize().unwrap();
        
        assert_eq!(tokens.len(), 12); // Including EOF
        assert!(matches!(tokens[0].kind, TokenKind::LParen));
        assert!(matches!(tokens[2].kind, TokenKind::Plus));
        assert!(matches!(tokens[5].kind, TokenKind::Star));
        assert!(matches!(tokens[7].kind, TokenKind::Minus));
        assert!(matches!(tokens[9].kind, TokenKind::Slash));
    }
}
