//! Token definitions for VeZ

use crate::span::Span;

/// A token in the source code
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Token types
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Fn, Let, Mut, Const, Type,
    Struct, Enum, Union, Trait, Impl,
    If, Else, Match, Loop, While, For,
    Break, Continue, Return,
    Pub, Use, Mod, Extern, Unsafe, Async, Await,
    As, In, Where, SelfLower, SelfUpper,
    Static, Inline,
    True, False,
    
    // Literals
    IntLiteral(String, Option<String>),   // value, optional suffix (i32, u64, etc.)
    FloatLiteral(String, Option<String>), // value, optional suffix (f32, f64)
    StringLiteral(String),
    CharLiteral(char),
    
    // Identifiers
    Ident(String),
    
    // Operators
    Plus, Minus, Star, Slash, Percent, StarStar,
    Eq, EqEq, Ne, Lt, Le, Gt, Ge,
    And, Or, Bang, AndAnd, OrOr,
    Amp, Pipe, Caret, Tilde, Shl, Shr,
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,
    
    // Delimiters
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,
    Comma, Semi, Colon, ColonColon,
    Dot, DotDot, DotDotEq,
    Arrow, FatArrow,
    
    // Special
    Eof,
}

impl TokenKind {
    pub fn is_keyword(&self) -> bool {
        matches!(self,
            TokenKind::Fn | TokenKind::Let | TokenKind::Mut | TokenKind::Const |
            TokenKind::Struct | TokenKind::Enum | TokenKind::Trait | TokenKind::Impl |
            TokenKind::If | TokenKind::Else | TokenKind::Match | TokenKind::Loop |
            TokenKind::While | TokenKind::For | TokenKind::Break | TokenKind::Continue |
            TokenKind::Return | TokenKind::Pub | TokenKind::Use | TokenKind::Mod
        )
    }
    
    pub fn is_literal(&self) -> bool {
        matches!(self,
            TokenKind::IntLiteral(_) | TokenKind::FloatLiteral(_) |
            TokenKind::StringLiteral(_) | TokenKind::CharLiteral(_) |
            TokenKind::True | TokenKind::False
        )
    }
}
