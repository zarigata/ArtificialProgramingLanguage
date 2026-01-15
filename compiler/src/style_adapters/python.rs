//! Python-style syntax adapter for VeZ
//! 
//! Allows writing VeZ code using Python syntax that transpiles to native VeZ AST.

use crate::error::{Error, ErrorKind, Result};
use crate::parser::ast::*;
use crate::span::{Position, Span};

/// Python-style lexer
struct PyLexer {
    input: Vec<char>,
    position: usize,
    current_pos: Position,
    indent_stack: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum PyToken {
    // Keywords
    Def, Class, If, Elif, Else, For, While, Return, Pass,
    Import, From, As, In, And, Or, Not, Is, None, True, False,
    Async, Await, With, Try, Except, Finally, Raise, Lambda,
    
    // Identifiers and literals
    Ident(String),
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    
    // Operators
    Plus, Minus, Star, Slash, DoubleSlash, Percent, StarStar,
    Eq, EqEq, NotEq, Lt, Le, Gt, Ge,
    Arrow, Colon, Comma, Dot, LParen, RParen, LBracket, RBracket,
    
    // Special
    Newline, Indent, Dedent, Eof,
}

impl PyLexer {
    fn new(input: &str) -> Self {
        PyLexer {
            input: input.chars().collect(),
            position: 0,
            current_pos: Position::initial(),
            indent_stack: vec![0],
        }
    }

    fn tokenize(&mut self) -> Result<Vec<PyToken>> {
        let mut tokens = Vec::new();
        let mut line_start = true;

        while !self.is_at_end() {
            if line_start {
                let indent = self.count_indent();
                line_start = false;

                let current_indent = *self.indent_stack.last().unwrap();
                if indent > current_indent {
                    self.indent_stack.push(indent);
                    tokens.push(PyToken::Indent);
                } else if indent < current_indent {
                    while let Some(&stack_indent) = self.indent_stack.last() {
                        if stack_indent <= indent {
                            break;
                        }
                        self.indent_stack.pop();
                        tokens.push(PyToken::Dedent);
                    }
                }
            }

            self.skip_whitespace_inline();

            if self.is_at_end() {
                break;
            }

            let ch = self.current_char();

            if ch == '#' {
                self.skip_comment();
                continue;
            }

            if ch == '\n' {
                self.advance();
                tokens.push(PyToken::Newline);
                line_start = true;
                continue;
            }

            tokens.push(self.next_token()?);
        }

        // Add dedents for remaining indentation levels
        while self.indent_stack.len() > 1 {
            self.indent_stack.pop();
            tokens.push(PyToken::Dedent);
        }

        tokens.push(PyToken::Eof);
        Ok(tokens)
    }

    fn count_indent(&mut self) -> usize {
        let mut count = 0;
        while !self.is_at_end() {
            match self.current_char() {
                ' ' => {
                    count += 1;
                    self.advance();
                }
                '\t' => {
                    count += 4; // Tab = 4 spaces
                    self.advance();
                }
                _ => break,
            }
        }
        count
    }

    fn skip_whitespace_inline(&mut self) {
        while !self.is_at_end() {
            match self.current_char() {
                ' ' | '\t' => self.advance(),
                _ => break,
            }
        }
    }

    fn skip_comment(&mut self) {
        while !self.is_at_end() && self.current_char() != '\n' {
            self.advance();
        }
    }

    fn next_token(&mut self) -> Result<PyToken> {
        let ch = self.current_char();

        match ch {
            '(' => { self.advance(); Ok(PyToken::LParen) }
            ')' => { self.advance(); Ok(PyToken::RParen) }
            '[' => { self.advance(); Ok(PyToken::LBracket) }
            ']' => { self.advance(); Ok(PyToken::RBracket) }
            ',' => { self.advance(); Ok(PyToken::Comma) }
            ':' => { self.advance(); Ok(PyToken::Colon) }
            '.' => { self.advance(); Ok(PyToken::Dot) }
            '+' => { self.advance(); Ok(PyToken::Plus) }
            '%' => { self.advance(); Ok(PyToken::Percent) }
            
            '-' => {
                self.advance();
                if !self.is_at_end() && self.current_char() == '>' {
                    self.advance();
                    Ok(PyToken::Arrow)
                } else {
                    Ok(PyToken::Minus)
                }
            }
            
            '*' => {
                self.advance();
                if !self.is_at_end() && self.current_char() == '*' {
                    self.advance();
                    Ok(PyToken::StarStar)
                } else {
                    Ok(PyToken::Star)
                }
            }
            
            '/' => {
                self.advance();
                if !self.is_at_end() && self.current_char() == '/' {
                    self.advance();
                    Ok(PyToken::DoubleSlash)
                } else {
                    Ok(PyToken::Slash)
                }
            }
            
            '=' => {
                self.advance();
                if !self.is_at_end() && self.current_char() == '=' {
                    self.advance();
                    Ok(PyToken::EqEq)
                } else {
                    Ok(PyToken::Eq)
                }
            }
            
            '!' => {
                self.advance();
                if !self.is_at_end() && self.current_char() == '=' {
                    self.advance();
                    Ok(PyToken::NotEq)
                } else {
                    Err(Error::new(ErrorKind::InvalidCharacter, "Unexpected '!'"))
                }
            }
            
            '<' => {
                self.advance();
                if !self.is_at_end() && self.current_char() == '=' {
                    self.advance();
                    Ok(PyToken::Le)
                } else {
                    Ok(PyToken::Lt)
                }
            }
            
            '>' => {
                self.advance();
                if !self.is_at_end() && self.current_char() == '=' {
                    self.advance();
                    Ok(PyToken::Ge)
                } else {
                    Ok(PyToken::Gt)
                }
            }
            
            '"' | '\'' => self.lex_string(ch),
            
            '0'..='9' => self.lex_number(),
            
            'a'..='z' | 'A'..='Z' | '_' => self.lex_identifier(),
            
            _ => Err(Error::new(
                ErrorKind::InvalidCharacter,
                format!("Unexpected character: '{}'", ch)
            )),
        }
    }

    fn lex_string(&mut self, quote: char) -> Result<PyToken> {
        self.advance(); // Skip opening quote
        let mut value = String::new();

        while !self.is_at_end() && self.current_char() != quote {
            if self.current_char() == '\\' {
                self.advance();
                if !self.is_at_end() {
                    let escaped = match self.current_char() {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        c => c,
                    };
                    value.push(escaped);
                    self.advance();
                }
            } else {
                value.push(self.current_char());
                self.advance();
            }
        }

        if self.is_at_end() {
            return Err(Error::new(ErrorKind::InvalidSyntax, "Unterminated string"));
        }

        self.advance(); // Skip closing quote
        Ok(PyToken::StringLit(value))
    }

    fn lex_number(&mut self) -> Result<PyToken> {
        let mut value = String::new();
        let mut is_float = false;

        while !self.is_at_end() {
            let ch = self.current_char();
            if ch.is_ascii_digit() {
                value.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                is_float = true;
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            let num = value.parse::<f64>()
                .map_err(|_| Error::new(ErrorKind::InvalidSyntax, "Invalid float literal"))?;
            Ok(PyToken::FloatLit(num))
        } else {
            let num = value.parse::<i64>()
                .map_err(|_| Error::new(ErrorKind::InvalidSyntax, "Invalid integer literal"))?;
            Ok(PyToken::IntLit(num))
        }
    }

    fn lex_identifier(&mut self) -> Result<PyToken> {
        let mut value = String::new();

        while !self.is_at_end() {
            let ch = self.current_char();
            if ch.is_alphanumeric() || ch == '_' {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        let token = match value.as_str() {
            "def" => PyToken::Def,
            "class" => PyToken::Class,
            "if" => PyToken::If,
            "elif" => PyToken::Elif,
            "else" => PyToken::Else,
            "for" => PyToken::For,
            "while" => PyToken::While,
            "return" => PyToken::Return,
            "pass" => PyToken::Pass,
            "import" => PyToken::Import,
            "from" => PyToken::From,
            "as" => PyToken::As,
            "in" => PyToken::In,
            "and" => PyToken::And,
            "or" => PyToken::Or,
            "not" => PyToken::Not,
            "is" => PyToken::Is,
            "None" => PyToken::None,
            "True" => PyToken::True,
            "False" => PyToken::False,
            "async" => PyToken::Async,
            "await" => PyToken::Await,
            "with" => PyToken::With,
            "try" => PyToken::Try,
            "except" => PyToken::Except,
            "finally" => PyToken::Finally,
            "raise" => PyToken::Raise,
            "lambda" => PyToken::Lambda,
            _ => PyToken::Ident(value),
        };

        Ok(token)
    }

    fn current_char(&self) -> char {
        self.input[self.position]
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            if self.input[self.position] == '\n' {
                self.current_pos.line += 1;
                self.current_pos.column = 1;
            } else {
                self.current_pos.column += 1;
            }
            self.position += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
}

/// Python-style parser
struct PyParser {
    tokens: Vec<PyToken>,
    position: usize,
}

impl PyParser {
    fn new(tokens: Vec<PyToken>) -> Self {
        PyParser { tokens, position: 0 }
    }

    fn parse(&mut self) -> Result<Program> {
        let mut items = Vec::new();

        while !self.is_at_end() {
            self.skip_newlines();
            if self.is_at_end() {
                break;
            }
            items.push(self.parse_item()?);
        }

        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Item> {
        match &self.current() {
            PyToken::Def => self.parse_function(),
            PyToken::Async => {
                self.advance();
                if matches!(self.current(), PyToken::Def) {
                    self.parse_function()
                } else {
                    Err(Error::new(ErrorKind::InvalidSyntax, "Expected 'def' after 'async'"))
                }
            }
            PyToken::Class => self.parse_class(),
            _ => Err(Error::new(
                ErrorKind::InvalidSyntax,
                format!("Expected item, found {:?}", self.current())
            )),
        }
    }

    fn parse_function(&mut self) -> Result<Item> {
        self.expect(PyToken::Def)?;
        
        let name = self.expect_ident()?;
        
        self.expect(PyToken::LParen)?;
        let mut params = Vec::new();
        
        while !matches!(self.current(), PyToken::RParen) {
            let param_name = self.expect_ident()?;
            
            let param_type = if matches!(self.current(), PyToken::Colon) {
                self.advance();
                self.parse_type_annotation()?
            } else {
                Type::Named("Any".to_string()) // Default to Any if no type annotation
            };
            
            params.push(Param {
                name: param_name,
                ty: param_type,
            });
            
            if matches!(self.current(), PyToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(PyToken::RParen)?;
        
        let return_type = if matches!(self.current(), PyToken::Arrow) {
            self.advance();
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        
        self.expect(PyToken::Colon)?;
        self.expect(PyToken::Newline)?;
        
        let body = self.parse_block()?;
        
        Ok(Item::Function(Function {
            name,
            attributes: Vec::new(),
            generics: Vec::new(),
            params,
            return_type,
            where_clause: None,
            body,
        }))
    }

    fn parse_class(&mut self) -> Result<Item> {
        self.expect(PyToken::Class)?;
        
        let name = self.expect_ident()?;
        
        // Skip optional inheritance for now
        if matches!(self.current(), PyToken::LParen) {
            self.advance();
            while !matches!(self.current(), PyToken::RParen) {
                self.advance();
            }
            self.expect(PyToken::RParen)?;
        }
        
        self.expect(PyToken::Colon)?;
        self.expect(PyToken::Newline)?;
        self.expect(PyToken::Indent)?;
        
        let mut fields = Vec::new();
        
        // Parse class body (simplified - just collect fields)
        while !matches!(self.current(), PyToken::Dedent) {
            self.skip_newlines();
            if matches!(self.current(), PyToken::Dedent) {
                break;
            }
            
            // Skip methods and other items for now
            if matches!(self.current(), PyToken::Def) {
                self.skip_until_dedent();
            } else {
                self.advance();
            }
        }
        
        self.expect(PyToken::Dedent)?;
        
        Ok(Item::Struct(Struct {
            name,
            generics: Vec::new(),
            fields,
            where_clause: None,
        }))
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>> {
        self.expect(PyToken::Indent)?;
        
        let mut stmts = Vec::new();
        
        while !matches!(self.current(), PyToken::Dedent) {
            self.skip_newlines();
            if matches!(self.current(), PyToken::Dedent) {
                break;
            }
            stmts.push(self.parse_stmt()?);
        }
        
        self.expect(PyToken::Dedent)?;
        
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        match &self.current() {
            PyToken::Return => self.parse_return_stmt(),
            PyToken::Pass => {
                self.advance();
                self.skip_newlines();
                Ok(Stmt::Expr(Expr::Literal(Literal::Int(0))))
            }
            _ => {
                let expr = self.parse_expr()?;
                self.skip_newlines();
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt> {
        self.expect(PyToken::Return)?;
        
        let value = if matches!(self.current(), PyToken::Newline) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        
        self.skip_newlines();
        Ok(Stmt::Return(value))
    }

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let mut left = self.parse_addition()?;
        
        loop {
            let op = match self.current() {
                PyToken::EqEq => BinOp::Eq,
                PyToken::NotEq => BinOp::Ne,
                PyToken::Lt => BinOp::Lt,
                PyToken::Le => BinOp::Le,
                PyToken::Gt => BinOp::Gt,
                PyToken::Ge => BinOp::Ge,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_addition()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_addition(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplication()?;
        
        loop {
            let op = match self.current() {
                PyToken::Plus => BinOp::Add,
                PyToken::Minus => BinOp::Sub,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_multiplication()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expr> {
        let mut left = self.parse_power()?;
        
        loop {
            let op = match self.current() {
                PyToken::Star => BinOp::Mul,
                PyToken::Slash => BinOp::Div,
                PyToken::DoubleSlash => BinOp::Div, // Integer division
                PyToken::Percent => BinOp::Mod,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_power()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Expr> {
        let mut left = self.parse_primary()?;
        
        if matches!(self.current(), PyToken::StarStar) {
            self.advance();
            let right = self.parse_power()?;
            // Convert ** to a power function call
            left = Expr::Call(
                Box::new(Expr::Ident("pow".to_string())),
                vec![left, right]
            );
        }
        
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.current().clone() {
            PyToken::IntLit(n) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int(n)))
            }
            PyToken::FloatLit(f) => {
                self.advance();
                Ok(Expr::Literal(Literal::Float(f)))
            }
            PyToken::StringLit(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::String(s)))
            }
            PyToken::True => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(true)))
            }
            PyToken::False => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(false)))
            }
            PyToken::None => {
                self.advance();
                // Represent None as a special identifier
                Ok(Expr::Ident("None".to_string()))
            }
            PyToken::Ident(name) => {
                self.advance();
                
                // Check for function call
                if matches!(self.current(), PyToken::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    
                    while !matches!(self.current(), PyToken::RParen) {
                        args.push(self.parse_expr()?);
                        if matches!(self.current(), PyToken::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    
                    self.expect(PyToken::RParen)?;
                    Ok(Expr::Call(Box::new(Expr::Ident(name)), args))
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            PyToken::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(PyToken::RParen)?;
                Ok(expr)
            }
            PyToken::LBracket => {
                self.advance();
                let mut elements = Vec::new();
                
                while !matches!(self.current(), PyToken::RBracket) {
                    elements.push(self.parse_expr()?);
                    if matches!(self.current(), PyToken::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                
                self.expect(PyToken::RBracket)?;
                Ok(Expr::Array(elements))
            }
            _ => Err(Error::new(
                ErrorKind::InvalidSyntax,
                format!("Expected expression, found {:?}", self.current())
            )),
        }
    }

    fn parse_type_annotation(&mut self) -> Result<Type> {
        let name = self.expect_ident()?;
        
        // Handle generic types like List[int]
        if matches!(self.current(), PyToken::LBracket) {
            self.advance();
            let mut args = Vec::new();
            
            while !matches!(self.current(), PyToken::RBracket) {
                args.push(self.parse_type_annotation()?);
                if matches!(self.current(), PyToken::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            
            self.expect(PyToken::RBracket)?;
            Ok(Type::Generic(name, args))
        } else {
            Ok(Type::Named(name))
        }
    }

    fn expect(&mut self, expected: PyToken) -> Result<()> {
        if std::mem::discriminant(&self.current()) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::ExpectedToken,
                format!("Expected {:?}, found {:?}", expected, self.current())
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.current() {
            PyToken::Ident(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(Error::new(
                ErrorKind::ExpectedToken,
                "Expected identifier"
            ))
        }
    }

    fn current(&self) -> &PyToken {
        &self.tokens[self.position]
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        matches!(self.current(), PyToken::Eof)
    }

    fn skip_newlines(&mut self) {
        while matches!(self.current(), PyToken::Newline) {
            self.advance();
        }
    }

    fn skip_until_dedent(&mut self) {
        let mut depth = 1;
        while !self.is_at_end() && depth > 0 {
            match self.current() {
                PyToken::Indent => depth += 1,
                PyToken::Dedent => depth -= 1,
                _ => {}
            }
            self.advance();
        }
    }
}

/// Parse Python-style source code and convert to VeZ AST
pub fn parse(source: &str) -> Result<Program> {
    let mut lexer = PyLexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = PyParser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let source = r#"
def add(x: int, y: int) -> int:
    return x + y
"#;
        let result = parse(source);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_function_with_multiple_statements() {
        let source = r#"
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"#;
        let result = parse(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_class_definition() {
        let source = r#"
class Point:
    def __init__(self, x: float, y: float):
        pass
"#;
        let result = parse(source);
        assert!(result.is_ok());
    }
}
