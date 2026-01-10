//! Parser for VeZ

pub mod ast;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod generics_tests;

use crate::error::{Error, ErrorKind, Result};
use crate::lexer::{Token, TokenKind};
pub use ast::*;

/// Parser for VeZ source code
pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, position: 0 }
    }
    
    pub fn parse(&mut self) -> Result<Program> {
        let mut items = Vec::new();
        
        while !self.is_at_end() {
            items.push(self.parse_item()?);
        }
        
        Ok(Program { items })
    }
    
    fn parse_item(&mut self) -> Result<Item> {
        match &self.current().kind {
            TokenKind::Fn => self.parse_function(),
            TokenKind::Struct => self.parse_struct(),
            TokenKind::Enum => self.parse_enum(),
            TokenKind::Trait => self.parse_trait(),
            TokenKind::Impl => self.parse_impl(),
            TokenKind::Use => self.parse_use(),
            TokenKind::Mod => self.parse_mod(),
            _ => Err(Error::new(
                ErrorKind::UnexpectedToken,
                format!("Expected item, found {:?}", self.current().kind)
            ))
        }
    }
    
    fn parse_function(&mut self) -> Result<Item> {
        self.expect(TokenKind::Fn)?;
        
        let name = self.expect_ident()?;
        
        let generics = if matches!(self.current().kind, TokenKind::Lt) {
            self.parse_generics()?
        } else {
            Vec::new()
        };
        
        self.expect(TokenKind::LParen)?;
        
        let mut params = Vec::new();
        while !matches!(self.current().kind, TokenKind::RParen) {
            let param_name = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let param_type = self.parse_type()?;
            
            params.push(Param {
                name: param_name,
                ty: param_type,
            });
            
            if matches!(self.current().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(TokenKind::RParen)?;
        
        let return_type = if matches!(self.current().kind, TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let where_clause = if matches!(self.current().kind, TokenKind::Where) {
            Some(self.parse_where_clause()?)
        } else {
            None
        };
        
        let body = if matches!(self.current().kind, TokenKind::LBrace) {
            self.parse_block()?
        } else {
            Vec::new()
        };
        
        Ok(Item::Function(Function {
            name,
            generics,
            params,
            return_type,
            where_clause,
            body,
        }))
    }
    
    fn parse_struct(&mut self) -> Result<Item> {
        self.expect(TokenKind::Struct)?;
        let name = self.expect_ident()?;
        
        let generics = if matches!(self.current().kind, TokenKind::Lt) {
            self.parse_generics()?
        } else {
            Vec::new()
        };
        
        let where_clause = if matches!(self.current().kind, TokenKind::Where) {
            Some(self.parse_where_clause()?)
        } else {
            None
        };
        
        self.expect(TokenKind::LBrace)?;
        
        let mut fields = Vec::new();
        while !matches!(self.current().kind, TokenKind::RBrace) {
            let field_name = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let field_type = self.parse_type()?;
            
            fields.push(Field {
                name: field_name,
                ty: field_type,
            });
            
            if matches!(self.current().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(TokenKind::RBrace)?;
        
        Ok(Item::Struct(Struct {
            name,
            generics,
            fields,
            where_clause,
        }))
    }
    
    fn parse_enum(&mut self) -> Result<Item> {
        self.expect(TokenKind::Enum)?;
        let name = self.expect_ident()?;
        
        let generics = if matches!(self.current().kind, TokenKind::Lt) {
            self.parse_generics()?
        } else {
            Vec::new()
        };
        
        let where_clause = if matches!(self.current().kind, TokenKind::Where) {
            Some(self.parse_where_clause()?)
        } else {
            None
        };
        
        self.expect(TokenKind::LBrace)?;
        
        let mut variants = Vec::new();
        while !matches!(self.current().kind, TokenKind::RBrace) {
            let variant_name = self.expect_ident()?;
            
            let data = if matches!(self.current().kind, TokenKind::LParen) {
                self.advance();
                let ty = self.parse_type()?;
                self.expect(TokenKind::RParen)?;
                Some(ty)
            } else {
                None
            };
            
            variants.push(Variant {
                name: variant_name,
                data,
            });
            
            if matches!(self.current().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(TokenKind::RBrace)?;
        
        Ok(Item::Enum(Enum {
            name,
            generics,
            variants,
            where_clause,
        }))
    }
    
    fn expect(&mut self, expected: TokenKind) -> Result<Token> {
        if std::mem::discriminant(&self.current().kind) == std::mem::discriminant(&expected) {
            Ok(self.advance())
        } else {
            Err(Error::new(
                ErrorKind::ExpectedToken,
                format!("Expected {:?}, found {:?}", expected, self.current().kind)
            ))
        }
    }
    
    fn expect_ident(&mut self) -> Result<String> {
        match &self.current().kind {
            TokenKind::Ident(name) => {
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
    
    fn current(&self) -> &Token {
        &self.tokens[self.position]
    }
    
    fn advance(&mut self) -> Token {
        let token = self.tokens[self.position].clone();
        if !self.is_at_end() {
            self.position += 1;
        }
        token
    }
    
    fn is_at_end(&self) -> bool {
        matches!(self.current().kind, TokenKind::Eof)
    }
    
    fn parse_type(&mut self) -> Result<Type> {
        match &self.current().kind {
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                
                // Check for generic type arguments
                if matches!(self.current().kind, TokenKind::Lt) {
                    self.advance();
                    let mut args = Vec::new();
                    
                    while !matches!(self.current().kind, TokenKind::Gt) {
                        args.push(self.parse_type()?);
                        
                        if matches!(self.current().kind, TokenKind::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    
                    self.expect(TokenKind::Gt)?;
                    Ok(Type::Generic(name, args))
                } else {
                    Ok(Type::Named(name))
                }
            }
            TokenKind::And => {
                self.advance();
                
                // Check for mutable reference
                if matches!(self.current().kind, TokenKind::Mut) {
                    self.advance();
                    let inner = self.parse_type()?;
                    Ok(Type::MutableReference(Box::new(inner)))
                } else {
                    let inner = self.parse_type()?;
                    Ok(Type::Reference(Box::new(inner)))
                }
            }
            TokenKind::LBracket => {
                self.advance();
                let inner = self.parse_type()?;
                self.expect(TokenKind::Semi)?;
                
                // Parse array size
                let size = match &self.current().kind {
                    TokenKind::IntLiteral(s, _) => {
                        let size = s.parse().map_err(|_| {
                            Error::new(ErrorKind::InvalidSyntax, "Invalid array size")
                        })?;
                        self.advance();
                        size
                    }
                    _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Expected array size")),
                };
                
                self.expect(TokenKind::RBracket)?;
                Ok(Type::Array(Box::new(inner), size))
            }
            TokenKind::LParen => {
                self.advance();
                let mut types = Vec::new();
                
                while !matches!(self.current().kind, TokenKind::RParen) {
                    types.push(self.parse_type()?);
                    
                    if matches!(self.current().kind, TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                
                self.expect(TokenKind::RParen)?;
                Ok(Type::Tuple(types))
            }
            _ => Err(Error::new(
                ErrorKind::InvalidSyntax,
                format!("Expected type, found {:?}", self.current().kind)
            )),
        }
    }
    
    fn parse_block(&mut self) -> Result<Vec<Stmt>> {
        self.expect(TokenKind::LBrace)?;
        
        let mut stmts = Vec::new();
        while !matches!(self.current().kind, TokenKind::RBrace) {
            stmts.push(self.parse_stmt()?);
        }
        
        self.expect(TokenKind::RBrace)?;
        Ok(stmts)
    }
    
    fn parse_stmt(&mut self) -> Result<Stmt> {
        match &self.current().kind {
            TokenKind::Let => self.parse_let_stmt(),
            TokenKind::Return => self.parse_return_stmt(),
            _ => {
                let expr = self.parse_expr()?;
                if matches!(self.current().kind, TokenKind::Semi) {
                    self.advance();
                }
                Ok(Stmt::Expr(expr))
            }
        }
    }
    
    fn parse_let_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenKind::Let)?;
        let name = self.expect_ident()?;
        
        let ty = if matches!(self.current().kind, TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let init = if matches!(self.current().kind, TokenKind::Eq) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::Let(name, ty, init))
    }
    
    fn parse_return_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenKind::Return)?;
        
        let value = if !matches!(self.current().kind, TokenKind::Semi) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::Return(value))
    }
    
    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_expr_with_precedence(0)
    }
    
    fn parse_expr_with_precedence(&mut self, min_precedence: u8) -> Result<Expr> {
        let mut left = self.parse_unary_expr()?;
        
        loop {
            // Handle postfix operators (method calls, field access, indexing)
            left = self.parse_postfix_expr(left.clone())?;
            
            let precedence = self.get_precedence(&self.current().kind);
            if precedence < min_precedence {
                break;
            }
            
            let op_token = self.current().kind.clone();
            self.advance();
            
            let right = self.parse_expr_with_precedence(precedence + 1)?;
            
            let op = match op_token {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::Ne => BinOp::Ne,
                TokenKind::Lt => BinOp::Lt,
                TokenKind::Le => BinOp::Le,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::Ge => BinOp::Ge,
                TokenKind::AndAnd => BinOp::And,
                TokenKind::OrOr => BinOp::Or,
                _ => return Err(Error::new(ErrorKind::InvalidSyntax, "Invalid binary operator")),
            };
            
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_unary_expr(&mut self) -> Result<Expr> {
        match &self.current().kind {
            TokenKind::Minus => {
                self.advance();
                let expr = self.parse_unary_expr()?;
                Ok(Expr::Unary(UnOp::Neg, Box::new(expr)))
            }
            TokenKind::Bang => {
                self.advance();
                let expr = self.parse_unary_expr()?;
                Ok(Expr::Unary(UnOp::Not, Box::new(expr)))
            }
            TokenKind::Star => {
                self.advance();
                let expr = self.parse_unary_expr()?;
                Ok(Expr::Unary(UnOp::Deref, Box::new(expr)))
            }
            TokenKind::And => {
                self.advance();
                let expr = self.parse_unary_expr()?;
                Ok(Expr::Unary(UnOp::Ref, Box::new(expr)))
            }
            _ => self.parse_primary_expr(),
        }
    }
    
    fn parse_postfix_expr(&mut self, mut expr: Expr) -> Result<Expr> {
        loop {
            match &self.current().kind {
                TokenKind::Dot => {
                    self.advance();
                    let field_name = self.expect_ident()?;
                    
                    // Check if it's a method call
                    if matches!(self.current().kind, TokenKind::LParen) {
                        self.advance();
                        let mut args = Vec::new();
                        
                        while !matches!(self.current().kind, TokenKind::RParen) {
                            args.push(self.parse_expr()?);
                            
                            if matches!(self.current().kind, TokenKind::Comma) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        
                        self.expect(TokenKind::RParen)?;
                        expr = Expr::MethodCall(Box::new(expr), field_name, args);
                    } else {
                        expr = Expr::Field(Box::new(expr), field_name);
                    }
                }
                TokenKind::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(TokenKind::RBracket)?;
                    expr = Expr::Index(Box::new(expr), Box::new(index));
                }
                _ => break,
            }
        }
        Ok(expr)
    }
    
    fn parse_primary_expr(&mut self) -> Result<Expr> {
        match &self.current().kind {
            TokenKind::IntLiteral(s, _) => {
                let value = s.parse().map_err(|_| {
                    Error::new(ErrorKind::InvalidSyntax, "Invalid integer literal")
                })?;
                self.advance();
                Ok(Expr::Literal(Literal::Int(value)))
            }
            TokenKind::FloatLiteral(s, _) => {
                let value = s.parse().map_err(|_| {
                    Error::new(ErrorKind::InvalidSyntax, "Invalid float literal")
                })?;
                self.advance();
                Ok(Expr::Literal(Literal::Float(value)))
            }
            TokenKind::StringLiteral(s) => {
                let value = s.clone();
                self.advance();
                Ok(Expr::Literal(Literal::String(value)))
            }
            TokenKind::CharLiteral(c) => {
                let value = *c;
                self.advance();
                Ok(Expr::Literal(Literal::Char(value)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(false)))
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                
                // Check for function call
                if matches!(self.current().kind, TokenKind::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    
                    while !matches!(self.current().kind, TokenKind::RParen) {
                        args.push(self.parse_expr()?);
                        
                        if matches!(self.current().kind, TokenKind::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr::Call(Box::new(Expr::Ident(name)), args))
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBrace => {
                let stmts = self.parse_block()?;
                Ok(Expr::Block(stmts))
            }
            TokenKind::If => self.parse_if_expr(),
            TokenKind::Match => self.parse_match_expr(),
            TokenKind::Loop => self.parse_loop_expr(),
            TokenKind::While => self.parse_while_expr(),
            TokenKind::For => self.parse_for_expr(),
            TokenKind::Break => {
                self.advance();
                let value = if !matches!(self.current().kind, TokenKind::Semi | TokenKind::RBrace) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                Ok(Expr::Break(value))
            }
            TokenKind::Continue => {
                self.advance();
                Ok(Expr::Continue)
            }
            TokenKind::Return => {
                self.advance();
                let value = if !matches!(self.current().kind, TokenKind::Semi | TokenKind::RBrace) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                Ok(Expr::Return(value))
            }
            TokenKind::LBracket => self.parse_array_expr(),
            _ => Err(Error::new(
                ErrorKind::InvalidSyntax,
                format!("Expected expression, found {:?}", self.current().kind)
            )),
        }
    }
    
    fn get_precedence(&self, token: &TokenKind) -> u8 {
        match token {
            TokenKind::OrOr => 1,
            TokenKind::AndAnd => 2,
            TokenKind::EqEq | TokenKind::Ne => 3,
            TokenKind::Lt | TokenKind::Le | TokenKind::Gt | TokenKind::Ge => 4,
            TokenKind::Plus | TokenKind::Minus => 5,
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => 6,
            _ => 0,
        }
    }
    
    fn parse_if_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::If)?;
        let condition = Box::new(self.parse_expr()?);
        
        let then_branch = Box::new(if matches!(self.current().kind, TokenKind::LBrace) {
            Expr::Block(self.parse_block()?)
        } else {
            self.parse_expr()?
        });
        
        let else_branch = if matches!(self.current().kind, TokenKind::Else) {
            self.advance();
            Some(Box::new(if matches!(self.current().kind, TokenKind::If) {
                self.parse_if_expr()?
            } else if matches!(self.current().kind, TokenKind::LBrace) {
                Expr::Block(self.parse_block()?)
            } else {
                self.parse_expr()?
            }))
        } else {
            None
        };
        
        Ok(Expr::If(condition, then_branch, else_branch))
    }
    
    fn parse_match_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Match)?;
        let scrutinee = Box::new(self.parse_expr()?);
        
        self.expect(TokenKind::LBrace)?;
        
        let mut arms = Vec::new();
        while !matches!(self.current().kind, TokenKind::RBrace) {
            let pattern = self.parse_pattern()?;
            
            let guard = if matches!(self.current().kind, TokenKind::If) {
                self.advance();
                Some(self.parse_expr()?)
            } else {
                None
            };
            
            self.expect(TokenKind::FatArrow)?;
            let body = self.parse_expr()?;
            
            arms.push(MatchArm { pattern, guard, body });
            
            if matches!(self.current().kind, TokenKind::Comma) {
                self.advance();
            }
        }
        
        self.expect(TokenKind::RBrace)?;
        Ok(Expr::Match(scrutinee, arms))
    }
    
    fn parse_loop_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Loop)?;
        let body = Box::new(if matches!(self.current().kind, TokenKind::LBrace) {
            Expr::Block(self.parse_block()?)
        } else {
            self.parse_expr()?
        });
        Ok(Expr::Loop(body))
    }
    
    fn parse_while_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::While)?;
        let condition = Box::new(self.parse_expr()?);
        let body = Box::new(if matches!(self.current().kind, TokenKind::LBrace) {
            Expr::Block(self.parse_block()?)
        } else {
            self.parse_expr()?
        });
        Ok(Expr::While(condition, body))
    }
    
    fn parse_for_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::For)?;
        let var = self.expect_ident()?;
        self.expect(TokenKind::In)?;
        let iter = Box::new(self.parse_expr()?);
        let body = Box::new(if matches!(self.current().kind, TokenKind::LBrace) {
            Expr::Block(self.parse_block()?)
        } else {
            self.parse_expr()?
        });
        Ok(Expr::For(var, iter, body))
    }
    
    fn parse_array_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::LBracket)?;
        
        let mut elements = Vec::new();
        while !matches!(self.current().kind, TokenKind::RBracket) {
            elements.push(self.parse_expr()?);
            
            if matches!(self.current().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(TokenKind::RBracket)?;
        Ok(Expr::Array(elements))
    }
    
    fn parse_pattern(&mut self) -> Result<Pattern> {
        match &self.current().kind {
            TokenKind::Ident(name) if name == "_" => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                Ok(Pattern::Ident(name))
            }
            TokenKind::IntLiteral(s, _) => {
                let value = s.parse().map_err(|_| {
                    Error::new(ErrorKind::InvalidSyntax, "Invalid integer in pattern")
                })?;
                self.advance();
                Ok(Pattern::Literal(Literal::Int(value)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::LParen => {
                self.advance();
                let mut patterns = Vec::new();
                
                while !matches!(self.current().kind, TokenKind::RParen) {
                    patterns.push(self.parse_pattern()?);
                    
                    if matches!(self.current().kind, TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                
                self.expect(TokenKind::RParen)?;
                Ok(Pattern::Tuple(patterns))
            }
            _ => Err(Error::new(
                ErrorKind::InvalidSyntax,
                format!("Expected pattern, found {:?}", self.current().kind)
            )),
        }
    }
    
    fn parse_generics(&mut self) -> Result<Vec<GenericParam>> {
        self.expect(TokenKind::Lt)?;
        
        let mut generics = Vec::new();
        while !matches!(self.current().kind, TokenKind::Gt) {
            let name = self.expect_ident()?;
            
            let mut bounds = Vec::new();
            if matches!(self.current().kind, TokenKind::Colon) {
                self.advance();
                
                loop {
                    bounds.push(self.expect_ident()?);
                    
                    if matches!(self.current().kind, TokenKind::Plus) {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            
            generics.push(GenericParam { name, bounds });
            
            if matches!(self.current().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(TokenKind::Gt)?;
        Ok(generics)
    }
    
    fn parse_where_clause(&mut self) -> Result<WhereClause> {
        self.expect(TokenKind::Where)?;
        
        let mut predicates = Vec::new();
        loop {
            let ty = self.parse_type()?;
            self.expect(TokenKind::Colon)?;
            
            let mut bounds = Vec::new();
            loop {
                bounds.push(self.expect_ident()?);
                
                if matches!(self.current().kind, TokenKind::Plus) {
                    self.advance();
                } else {
                    break;
                }
            }
            
            predicates.push(WherePredicate { ty, bounds });
            
            if matches!(self.current().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(WhereClause { predicates })
    }
    
    fn parse_trait(&mut self) -> Result<Item> {
        self.expect(TokenKind::Trait)?;
        let name = self.expect_ident()?;
        
        let generics = if matches!(self.current().kind, TokenKind::Lt) {
            self.parse_generics()?
        } else {
            Vec::new()
        };
        
        let mut supertraits = Vec::new();
        if matches!(self.current().kind, TokenKind::Colon) {
            self.advance();
            
            loop {
                supertraits.push(self.expect_ident()?);
                
                if matches!(self.current().kind, TokenKind::Plus) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        self.expect(TokenKind::LBrace)?;
        
        let mut items = Vec::new();
        while !matches!(self.current().kind, TokenKind::RBrace) {
            if matches!(self.current().kind, TokenKind::Fn) {
                self.advance();
                let fn_name = self.expect_ident()?;
                self.expect(TokenKind::LParen)?;
                
                let mut params = Vec::new();
                while !matches!(self.current().kind, TokenKind::RParen) {
                    let param_name = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let param_type = self.parse_type()?;
                    
                    params.push(Param {
                        name: param_name,
                        ty: param_type,
                    });
                    
                    if matches!(self.current().kind, TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                
                self.expect(TokenKind::RParen)?;
                
                let return_type = if matches!(self.current().kind, TokenKind::Arrow) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };
                
                self.expect(TokenKind::Semi)?;
                items.push(TraitItem::Function(fn_name, params, return_type));
            } else if matches!(self.current().kind, TokenKind::Type) {
                self.advance();
                let type_name = self.expect_ident()?;
                self.expect(TokenKind::Semi)?;
                items.push(TraitItem::Type(type_name));
            } else {
                break;
            }
        }
        
        self.expect(TokenKind::RBrace)?;
        
        Ok(Item::Trait(Trait {
            name,
            generics,
            supertraits,
            items,
        }))
    }
    
    fn parse_impl(&mut self) -> Result<Item> {
        self.expect(TokenKind::Impl)?;
        
        let generics = if matches!(self.current().kind, TokenKind::Lt) {
            self.parse_generics()?
        } else {
            Vec::new()
        };
        
        // Parse trait name or self type
        let first_name = self.expect_ident()?;
        
        let (trait_name, self_ty) = if matches!(self.current().kind, TokenKind::For) {
            self.advance();
            let self_ty = self.parse_type()?;
            (Some(first_name), self_ty)
        } else {
            (None, Type::Named(first_name))
        };
        
        let where_clause = if matches!(self.current().kind, TokenKind::Where) {
            Some(self.parse_where_clause()?)
        } else {
            None
        };
        
        self.expect(TokenKind::LBrace)?;
        
        let mut items = Vec::new();
        while !matches!(self.current().kind, TokenKind::RBrace) {
            if matches!(self.current().kind, TokenKind::Fn) {
                if let Item::Function(func) = self.parse_function()? {
                    items.push(ImplItem::Function(func));
                }
            } else if matches!(self.current().kind, TokenKind::Type) {
                self.advance();
                let type_name = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let ty = self.parse_type()?;
                self.expect(TokenKind::Semi)?;
                items.push(ImplItem::Type(type_name, ty));
            } else {
                break;
            }
        }
        
        self.expect(TokenKind::RBrace)?;
        
        Ok(Item::Impl(Impl {
            generics,
            trait_name,
            self_ty,
            where_clause,
            items,
        }))
    }
    
    fn parse_use(&mut self) -> Result<Item> {
        self.expect(TokenKind::Use)?;
        
        let mut segments = Vec::new();
        loop {
            segments.push(self.expect_ident()?);
            
            if matches!(self.current().kind, TokenKind::ColonColon) {
                self.advance();
            } else {
                break;
            }
        }
        
        let alias = if matches!(self.current().kind, TokenKind::As) {
            self.advance();
            Some(self.expect_ident()?)
        } else {
            None
        };
        
        self.expect(TokenKind::Semi)?;
        
        Ok(Item::Use(UsePath { segments, alias }))
    }
    
    fn parse_mod(&mut self) -> Result<Item> {
        self.expect(TokenKind::Mod)?;
        let name = self.expect_ident()?;
        
        if matches!(self.current().kind, TokenKind::Semi) {
            self.advance();
            Ok(Item::Mod(name, Vec::new()))
        } else {
            self.expect(TokenKind::LBrace)?;
            
            let mut items = Vec::new();
            while !matches!(self.current().kind, TokenKind::RBrace) {
                items.push(self.parse_item()?);
            }
            
            self.expect(TokenKind::RBrace)?;
            Ok(Item::Mod(name, items))
        }
    }
}
