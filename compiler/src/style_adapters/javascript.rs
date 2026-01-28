//! JavaScript/TypeScript-style syntax adapter for VeZ

use crate::error::{Error, ErrorKind, Result};
use crate::parser::ast::*;

#[derive(Debug, Clone, PartialEq)]
enum JsToken {
    // Keywords
    Function, Const, Let, Var, Class, If, Else, For, While, Return,
    Import, Export, From, As, Default, Async, Await, New, This,
    Try, Catch, Finally, Throw, Typeof, Instanceof,
    
    // Identifiers and literals
    Ident(String),
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    True, False, Null, Undefined,
    
    // Operators
    Plus, Minus, Star, Slash, Percent,
    Eq, EqEq, EqEqEq, NotEq, NotEqEq,
    Lt, Le, Gt, Ge,
    And, Or, Not, AndAnd, OrOr,
    Arrow, Question, Colon, Comma, Dot, Semi,
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    
    Eof,
}

struct JsLexer {
    input: Vec<char>,
    position: usize,
}

impl JsLexer {
    fn new(input: &str) -> Self {
        JsLexer {
            input: input.chars().collect(),
            position: 0,
        }
    }

    fn tokenize(&mut self) -> Result<Vec<JsToken>> {
        let mut tokens = Vec::new();

        while !self.is_at_end() {
            self.skip_whitespace();
            if self.is_at_end() {
                break;
            }

            if self.current_char() == '/' && self.peek() == Some('/') {
                self.skip_line_comment();
                continue;
            }

            if self.current_char() == '/' && self.peek() == Some('*') {
                self.skip_block_comment()?;
                continue;
            }

            tokens.push(self.next_token()?);
        }

        tokens.push(JsToken::Eof);
        Ok(tokens)
    }

    fn skip_whitespace(&mut self) {
        while !self.is_at_end() && self.current_char().is_whitespace() {
            self.advance();
        }
    }

    fn skip_line_comment(&mut self) {
        while !self.is_at_end() && self.current_char() != '\n' {
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) -> Result<()> {
        self.advance(); // Skip '/'
        self.advance(); // Skip '*'

        while !self.is_at_end() {
            if self.current_char() == '*' && self.peek() == Some('/') {
                self.advance();
                self.advance();
                return Ok(());
            }
            self.advance();
        }

        Err(Error::new(ErrorKind::InvalidSyntax, "Unterminated block comment"))
    }

    fn next_token(&mut self) -> Result<JsToken> {
        let ch = self.current_char();

        match ch {
            '(' => { self.advance(); Ok(JsToken::LParen) }
            ')' => { self.advance(); Ok(JsToken::RParen) }
            '{' => { self.advance(); Ok(JsToken::LBrace) }
            '}' => { self.advance(); Ok(JsToken::RBrace) }
            '[' => { self.advance(); Ok(JsToken::LBracket) }
            ']' => { self.advance(); Ok(JsToken::RBracket) }
            ',' => { self.advance(); Ok(JsToken::Comma) }
            ';' => { self.advance(); Ok(JsToken::Semi) }
            '.' => { self.advance(); Ok(JsToken::Dot) }
            '?' => { self.advance(); Ok(JsToken::Question) }
            ':' => { self.advance(); Ok(JsToken::Colon) }
            '+' => { self.advance(); Ok(JsToken::Plus) }
            '%' => { self.advance(); Ok(JsToken::Percent) }
            
            '-' => {
                self.advance();
                if self.peek() == Some('>') {
                    self.advance();
                    Ok(JsToken::Arrow)
                } else {
                    Ok(JsToken::Minus)
                }
            }
            
            '*' => { self.advance(); Ok(JsToken::Star) }
            '/' => { self.advance(); Ok(JsToken::Slash) }
            
            '=' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Ok(JsToken::EqEqEq)
                    } else {
                        Ok(JsToken::EqEq)
                    }
                } else {
                    Ok(JsToken::Eq)
                }
            }
            
            '!' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Ok(JsToken::NotEqEq)
                    } else {
                        Ok(JsToken::NotEq)
                    }
                } else {
                    Ok(JsToken::Not)
                }
            }
            
            '<' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(JsToken::Le)
                } else {
                    Ok(JsToken::Lt)
                }
            }
            
            '>' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(JsToken::Ge)
                } else {
                    Ok(JsToken::Gt)
                }
            }
            
            '&' => {
                self.advance();
                if self.peek() == Some('&') {
                    self.advance();
                    Ok(JsToken::AndAnd)
                } else {
                    Ok(JsToken::And)
                }
            }
            
            '|' => {
                self.advance();
                if self.peek() == Some('|') {
                    self.advance();
                    Ok(JsToken::OrOr)
                } else {
                    Ok(JsToken::Or)
                }
            }
            
            '"' | '\'' | '`' => self.lex_string(ch),
            
            '0'..='9' => self.lex_number(),
            
            'a'..='z' | 'A'..='Z' | '_' | '$' => self.lex_identifier(),
            
            _ => Err(Error::new(
                ErrorKind::InvalidCharacter,
                format!("Unexpected character: '{}'", ch)
            )),
        }
    }

    fn lex_string(&mut self, quote: char) -> Result<JsToken> {
        self.advance();
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
                        '`' => '`',
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

        self.advance();
        Ok(JsToken::StringLit(value))
    }

    fn lex_number(&mut self) -> Result<JsToken> {
        let mut value = String::new();
        let mut is_float = false;

        while !self.is_at_end() {
            let ch = self.current_char();
            if ch.is_ascii_digit() {
                value.push(ch);
                self.advance();
            } else if ch == '.' && !is_float && self.peek().map_or(false, |c| c.is_ascii_digit()) {
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
            Ok(JsToken::FloatLit(num))
        } else {
            let num = value.parse::<i64>()
                .map_err(|_| Error::new(ErrorKind::InvalidSyntax, "Invalid integer literal"))?;
            Ok(JsToken::IntLit(num))
        }
    }

    fn lex_identifier(&mut self) -> Result<JsToken> {
        let mut value = String::new();

        while !self.is_at_end() {
            let ch = self.current_char();
            if ch.is_alphanumeric() || ch == '_' || ch == '$' {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        let token = match value.as_str() {
            "function" => JsToken::Function,
            "const" => JsToken::Const,
            "let" => JsToken::Let,
            "var" => JsToken::Var,
            "class" => JsToken::Class,
            "if" => JsToken::If,
            "else" => JsToken::Else,
            "for" => JsToken::For,
            "while" => JsToken::While,
            "return" => JsToken::Return,
            "import" => JsToken::Import,
            "export" => JsToken::Export,
            "from" => JsToken::From,
            "as" => JsToken::As,
            "default" => JsToken::Default,
            "async" => JsToken::Async,
            "await" => JsToken::Await,
            "new" => JsToken::New,
            "this" => JsToken::This,
            "try" => JsToken::Try,
            "catch" => JsToken::Catch,
            "finally" => JsToken::Finally,
            "throw" => JsToken::Throw,
            "typeof" => JsToken::Typeof,
            "instanceof" => JsToken::Instanceof,
            "true" => JsToken::True,
            "false" => JsToken::False,
            "null" => JsToken::Null,
            "undefined" => JsToken::Undefined,
            _ => JsToken::Ident(value),
        };

        Ok(token)
    }

    fn current_char(&self) -> char {
        self.input[self.position]
    }

    fn peek(&self) -> Option<char> {
        if self.position + 1 < self.input.len() {
            Some(self.input[self.position + 1])
        } else {
            None
        }
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
}

struct JsParser {
    tokens: Vec<JsToken>,
    position: usize,
}

impl JsParser {
    fn new(tokens: Vec<JsToken>) -> Self {
        JsParser { tokens, position: 0 }
    }

    fn parse(&mut self) -> Result<Program> {
        let mut items = Vec::new();

        while !self.is_at_end() {
            items.push(self.parse_item()?);
        }

        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Item> {
        match &self.current() {
            JsToken::Function => self.parse_function(),
            JsToken::Async => {
                self.advance();
                if matches!(self.current(), JsToken::Function) {
                    self.parse_function()
                } else {
                    Err(Error::new(ErrorKind::InvalidSyntax, "Expected 'function' after 'async'"))
                }
            }
            JsToken::Const | JsToken::Let => {
                self.advance();
                let name = self.expect_ident()?;
                
                if matches!(self.current(), JsToken::Eq) {
                    self.advance();
                    
                    // Check if it's an arrow function
                    if matches!(self.current(), JsToken::LParen) {
                        self.position -= 2; // Go back
                        self.parse_arrow_function(name)
                    } else {
                        // Skip for now
                        while !matches!(self.current(), JsToken::Semi | JsToken::Eof) {
                            self.advance();
                        }
                        if matches!(self.current(), JsToken::Semi) {
                            self.advance();
                        }
                        self.parse_item()
                    }
                } else {
                    Err(Error::new(ErrorKind::InvalidSyntax, "Expected '=' after variable name"))
                }
            }
            JsToken::Class => self.parse_class(),
            _ => Err(Error::new(
                ErrorKind::InvalidSyntax,
                format!("Expected item, found {:?}", self.current())
            )),
        }
    }

    fn parse_function(&mut self) -> Result<Item> {
        self.expect(JsToken::Function)?;
        
        let name = self.expect_ident()?;
        
        self.expect(JsToken::LParen)?;
        let mut params = Vec::new();
        
        while !matches!(self.current(), JsToken::RParen) {
            let param_name = self.expect_ident()?;
            
            let param_type = if matches!(self.current(), JsToken::Colon) {
                self.advance();
                self.parse_type_annotation()?
            } else {
                Type::Named("any".to_string())
            };
            
            params.push(Param {
                name: param_name,
                ty: param_type,
            });
            
            if matches!(self.current(), JsToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(JsToken::RParen)?;
        
        let return_type = if matches!(self.current(), JsToken::Colon) {
            self.advance();
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        
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

    fn parse_arrow_function(&mut self, name: String) -> Result<Item> {
        self.advance(); // Skip const/let
        self.advance(); // Skip name
        self.expect(JsToken::Eq)?;
        
        let mut params = Vec::new();
        
        if matches!(self.current(), JsToken::LParen) {
            self.advance();
            
            while !matches!(self.current(), JsToken::RParen) {
                let param_name = self.expect_ident()?;
                
                let param_type = if matches!(self.current(), JsToken::Colon) {
                    self.advance();
                    self.parse_type_annotation()?
                } else {
                    Type::Named("any".to_string())
                };
                
                params.push(Param {
                    name: param_name,
                    ty: param_type,
                });
                
                if matches!(self.current(), JsToken::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            
            self.expect(JsToken::RParen)?;
        } else {
            // Single parameter without parentheses
            let param_name = self.expect_ident()?;
            params.push(Param {
                name: param_name,
                ty: Type::Named("any".to_string()),
            });
        }
        
        self.expect(JsToken::Arrow)?;
        
        let body = if matches!(self.current(), JsToken::LBrace) {
            self.parse_block()?
        } else {
            // Expression body
            let expr = self.parse_expr()?;
            if matches!(self.current(), JsToken::Semi) {
                self.advance();
            }
            vec![Stmt::Return(Some(expr))]
        };
        
        Ok(Item::Function(Function {
            name,
            attributes: Vec::new(),
            generics: Vec::new(),
            params,
            return_type: None,
            where_clause: None,
            body,
        }))
    }

    fn parse_class(&mut self) -> Result<Item> {
        self.expect(JsToken::Class)?;
        
        let name = self.expect_ident()?;
        
        self.expect(JsToken::LBrace)?;
        
        let mut fields = Vec::new();
        
        while !matches!(self.current(), JsToken::RBrace) {
            // Skip methods and properties for now
            if matches!(self.current(), JsToken::Ident(_)) {
                self.advance();
                
                if matches!(self.current(), JsToken::LParen) {
                    // Method
                    self.skip_until_brace_close();
                } else {
                    // Property
                    while !matches!(self.current(), JsToken::Semi | JsToken::RBrace) {
                        self.advance();
                    }
                    if matches!(self.current(), JsToken::Semi) {
                        self.advance();
                    }
                }
            } else {
                self.advance();
            }
        }
        
        self.expect(JsToken::RBrace)?;
        
        Ok(Item::Struct(Struct {
            name,
            generics: Vec::new(),
            fields,
            where_clause: None,
        }))
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>> {
        self.expect(JsToken::LBrace)?;
        
        let mut stmts = Vec::new();
        
        while !matches!(self.current(), JsToken::RBrace) {
            stmts.push(self.parse_stmt()?);
        }
        
        self.expect(JsToken::RBrace)?;
        
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        match &self.current() {
            JsToken::Return => self.parse_return_stmt(),
            JsToken::Const | JsToken::Let | JsToken::Var => {
                self.advance();
                let name = self.expect_ident()?;
                
                let ty = if matches!(self.current(), JsToken::Colon) {
                    self.advance();
                    Some(self.parse_type_annotation()?)
                } else {
                    None
                };
                
                let init = if matches!(self.current(), JsToken::Eq) {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                
                if matches!(self.current(), JsToken::Semi) {
                    self.advance();
                }
                
                Ok(Stmt::Let(name, ty, init))
            }
            _ => {
                let expr = self.parse_expr()?;
                if matches!(self.current(), JsToken::Semi) {
                    self.advance();
                }
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt> {
        self.expect(JsToken::Return)?;
        
        let value = if matches!(self.current(), JsToken::Semi | JsToken::RBrace) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        
        if matches!(self.current(), JsToken::Semi) {
            self.advance();
        }
        
        Ok(Stmt::Return(value))
    }

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Result<Expr> {
        let mut expr = self.parse_logical_or()?;
        
        if matches!(self.current(), JsToken::Question) {
            self.advance();
            let then_expr = self.parse_expr()?;
            self.expect(JsToken::Colon)?;
            let else_expr = self.parse_expr()?;
            
            expr = Expr::If(
                Box::new(expr),
                Box::new(then_expr),
                Some(Box::new(else_expr))
            );
        }
        
        Ok(expr)
    }

    fn parse_logical_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_logical_and()?;
        
        while matches!(self.current(), JsToken::OrOr) {
            self.advance();
            let right = self.parse_logical_and()?;
            left = Expr::Binary(Box::new(left), BinOp::Or, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_logical_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_equality()?;
        
        while matches!(self.current(), JsToken::AndAnd) {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::Binary(Box::new(left), BinOp::And, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr> {
        let mut left = self.parse_comparison()?;
        
        loop {
            let op = match self.current() {
                JsToken::EqEq | JsToken::EqEqEq => BinOp::Eq,
                JsToken::NotEq | JsToken::NotEqEq => BinOp::Ne,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let mut left = self.parse_addition()?;
        
        loop {
            let op = match self.current() {
                JsToken::Lt => BinOp::Lt,
                JsToken::Le => BinOp::Le,
                JsToken::Gt => BinOp::Gt,
                JsToken::Ge => BinOp::Ge,
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
                JsToken::Plus => BinOp::Add,
                JsToken::Minus => BinOp::Sub,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_multiplication()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        
        loop {
            let op = match self.current() {
                JsToken::Star => BinOp::Mul,
                JsToken::Slash => BinOp::Div,
                JsToken::Percent => BinOp::Mod,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.current() {
            JsToken::Not => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary(UnOp::Not, Box::new(expr)))
            }
            JsToken::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary(UnOp::Neg, Box::new(expr)))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;
        
        loop {
            match self.current() {
                JsToken::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    
                    while !matches!(self.current(), JsToken::RParen) {
                        args.push(self.parse_expr()?);
                        if matches!(self.current(), JsToken::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    
                    self.expect(JsToken::RParen)?;
                    expr = Expr::Call(Box::new(expr), args);
                }
                JsToken::Dot => {
                    self.advance();
                    let field = self.expect_ident()?;
                    expr = Expr::Field(Box::new(expr), field);
                }
                JsToken::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(JsToken::RBracket)?;
                    expr = Expr::Index(Box::new(expr), Box::new(index));
                }
                _ => break,
            }
        }
        
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.current().clone() {
            JsToken::IntLit(n) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int(n)))
            }
            JsToken::FloatLit(f) => {
                self.advance();
                Ok(Expr::Literal(Literal::Float(f)))
            }
            JsToken::StringLit(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::String(s)))
            }
            JsToken::True => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(true)))
            }
            JsToken::False => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(false)))
            }
            JsToken::Null | JsToken::Undefined => {
                self.advance();
                Ok(Expr::Ident("null".to_string()))
            }
            JsToken::Ident(name) => {
                self.advance();
                Ok(Expr::Ident(name))
            }
            JsToken::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(JsToken::RParen)?;
                Ok(expr)
            }
            JsToken::LBracket => {
                self.advance();
                let mut elements = Vec::new();
                
                while !matches!(self.current(), JsToken::RBracket) {
                    elements.push(self.parse_expr()?);
                    if matches!(self.current(), JsToken::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                
                self.expect(JsToken::RBracket)?;
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
        
        if matches!(self.current(), JsToken::Lt) {
            self.advance();
            let mut args = Vec::new();
            
            while !matches!(self.current(), JsToken::Gt) {
                args.push(self.parse_type_annotation()?);
                if matches!(self.current(), JsToken::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            
            self.expect(JsToken::Gt)?;
            Ok(Type::Generic(name, args))
        } else {
            Ok(Type::Named(name))
        }
    }

    fn expect(&mut self, expected: JsToken) -> Result<()> {
        if std::mem::discriminant(self.current()) == std::mem::discriminant(&expected) {
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
            JsToken::Ident(name) => {
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

    fn current(&self) -> &JsToken {
        &self.tokens[self.position]
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        matches!(self.current(), JsToken::Eof)
    }

    fn skip_until_brace_close(&mut self) {
        let mut depth = 0;
        while !self.is_at_end() {
            match self.current() {
                JsToken::LBrace => depth += 1,
                JsToken::RBrace => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                }
                _ => {}
            }
            self.advance();
        }
    }
}

pub fn parse(source: &str) -> Result<Program> {
    let mut lexer = JsLexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = JsParser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_declaration() {
        let source = r#"
function add(x, y) {
    return x + y;
}
"#;
        let result = parse(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_arrow_function() {
        let source = r#"
const multiply = (a, b) => a * b;
"#;
        let result = parse(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_class_definition() {
        let source = r#"
class Point {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}
"#;
        let result = parse(source);
        assert!(result.is_ok());
    }
}
