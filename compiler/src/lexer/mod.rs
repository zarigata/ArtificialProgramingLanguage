//! Lexical analyzer for VeZ

pub mod token;
#[cfg(test)]
mod tests;

use crate::error::{Error, ErrorKind, Result};
use crate::span::{Position, Span};
pub use token::{Token, TokenKind};

/// Lexer for tokenizing VeZ source code
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_pos: Position,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            position: 0,
            current_pos: Position::initial(),
        }
    }
    
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        
        loop {
            self.skip_whitespace();
            
            if self.is_at_end() {
                tokens.push(self.make_token(TokenKind::Eof));
                break;
            }
            
            let token = self.next_token()?;
            tokens.push(token);
        }
        
        Ok(tokens)
    }
    
    fn next_token(&mut self) -> Result<Token> {
        let start_pos = self.current_pos;
        let ch = self.current_char();
        
        let kind = match ch {
            // Single-character tokens
            '(' => { self.advance(); TokenKind::LParen }
            ')' => { self.advance(); TokenKind::RParen }
            '{' => { self.advance(); TokenKind::LBrace }
            '}' => { self.advance(); TokenKind::RBrace }
            '[' => { self.advance(); TokenKind::LBracket }
            ']' => { self.advance(); TokenKind::RBracket }
            ',' => { self.advance(); TokenKind::Comma }
            ';' => { self.advance(); TokenKind::Semi }
            
            // Operators and multi-character tokens
            '+' => self.lex_plus(),
            '-' => self.lex_minus(),
            '*' => self.lex_star(),
            '/' => self.lex_slash()?,
            '%' => { self.advance(); TokenKind::Percent }
            '=' => self.lex_equals(),
            '!' => self.lex_bang(),
            '<' => self.lex_less(),
            '>' => self.lex_greater(),
            '&' => self.lex_ampersand(),
            '|' => self.lex_pipe(),
            '^' => { self.advance(); TokenKind::Caret }
            '~' => { self.advance(); TokenKind::Tilde }
            '.' => self.lex_dot(),
            ':' => self.lex_colon(),
            
            // String literals
            '"' => self.lex_string()?,
            '\'' => self.lex_char()?,
            
            // Numbers
            '0'..='9' => self.lex_number()?,
            
            // Identifiers, keywords, and raw strings
            'a'..='z' | 'A'..='Z' | '_' => {
                // Check for raw string: r"..." or r#"..."#
                if ch == 'r' && !self.is_at_end() {
                    let next_pos = self.position + 1;
                    if next_pos < self.input.len() {
                        let next_ch = self.input[next_pos];
                        if next_ch == '"' || next_ch == '#' {
                            return Ok(Token {
                                kind: self.lex_raw_string()?,
                                span: Span::new(start_pos, self.current_pos),
                            });
                        }
                    }
                }
                self.lex_identifier()
            }
            
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidCharacter,
                    format!("Unexpected character: '{}'", ch)
                ).with_span(Span::new(start_pos, self.current_pos)));
            }
        };
        
        Ok(Token {
            kind,
            span: Span::new(start_pos, self.current_pos),
        })
    }
    
    fn lex_identifier(&mut self) -> TokenKind {
        let start = self.position;
        
        while !self.is_at_end() && self.is_identifier_continue(self.current_char()) {
            self.advance();
        }
        
        let text: String = self.input[start..self.position].iter().collect();
        
        // Check if it's a keyword
        match text.as_str() {
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "const" => TokenKind::Const,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "match" => TokenKind::Match,
            "loop" => TokenKind::Loop,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "struct" => TokenKind::Struct,
            "enum" => TokenKind::Enum,
            "trait" => TokenKind::Trait,
            "impl" => TokenKind::Impl,
            "pub" => TokenKind::Pub,
            "use" => TokenKind::Use,
            "mod" => TokenKind::Mod,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Ident(text),
        }
    }
    
    fn lex_number(&mut self) -> Result<TokenKind> {
        let start = self.position;
        let start_pos = self.current_pos;
        
        // Check for hex, octal, or binary prefix
        if self.current_char() == '0' && !self.is_at_end() {
            self.advance();
            
            if !self.is_at_end() {
                match self.current_char() {
                    'x' | 'X' => return self.lex_hex_number(start, start_pos),
                    'o' | 'O' => return self.lex_octal_number(start, start_pos),
                    'b' | 'B' => return self.lex_binary_number(start, start_pos),
                    _ => {}
                }
            }
        }
        
        // Decimal number (integer or float)
        while !self.is_at_end() && (self.current_char().is_ascii_digit() || self.current_char() == '_') {
            self.advance();
        }
        
        // Check for float
        if !self.is_at_end() && self.current_char() == '.' {
            let next_pos = self.position + 1;
            if next_pos < self.input.len() && self.input[next_pos].is_ascii_digit() {
                self.advance(); // consume '.'
                
                while !self.is_at_end() && (self.current_char().is_ascii_digit() || self.current_char() == '_') {
                    self.advance();
                }
                
                // Check for exponent
                if !self.is_at_end() && (self.current_char() == 'e' || self.current_char() == 'E') {
                    self.advance();
                    
                    if !self.is_at_end() && (self.current_char() == '+' || self.current_char() == '-') {
                        self.advance();
                    }
                    
                    if self.is_at_end() || !self.current_char().is_ascii_digit() {
                        return Err(Error::new(
                            ErrorKind::InvalidNumber,
                            "Invalid exponent in float literal"
                        ).with_span(Span::new(start_pos, self.current_pos)));
                    }
                    
                    while !self.is_at_end() && (self.current_char().is_ascii_digit() || self.current_char() == '_') {
                        self.advance();
                    }
                }
                
                let text: String = self.input[start..self.position].iter().collect();
                let suffix = self.lex_number_suffix();
                return Ok(TokenKind::FloatLiteral(text, suffix));
            }
        }
        
        // Check for exponent in integer (makes it a float)
        if !self.is_at_end() && (self.current_char() == 'e' || self.current_char() == 'E') {
            self.advance();
            
            if !self.is_at_end() && (self.current_char() == '+' || self.current_char() == '-') {
                self.advance();
            }
            
            if self.is_at_end() || !self.current_char().is_ascii_digit() {
                return Err(Error::new(
                    ErrorKind::InvalidNumber,
                    "Invalid exponent in float literal"
                ).with_span(Span::new(start_pos, self.current_pos)));
            }
            
            while !self.is_at_end() && (self.current_char().is_ascii_digit() || self.current_char() == '_') {
                self.advance();
            }
            
            let text: String = self.input[start..self.position].iter().collect();
            let suffix = self.lex_number_suffix();
            return Ok(TokenKind::FloatLiteral(text, suffix));
        }
        
        let text: String = self.input[start..self.position].iter().collect();
        let suffix = self.lex_number_suffix();
        Ok(TokenKind::IntLiteral(text, suffix))
    }
    
    fn lex_hex_number(&mut self, start: usize, start_pos: Position) -> Result<TokenKind> {
        self.advance(); // consume 'x' or 'X'
        
        if self.is_at_end() || !self.current_char().is_ascii_hexdigit() {
            return Err(Error::new(
                ErrorKind::InvalidNumber,
                "Invalid hexadecimal literal"
            ).with_span(Span::new(start_pos, self.current_pos)));
        }
        
        while !self.is_at_end() && (self.current_char().is_ascii_hexdigit() || self.current_char() == '_') {
            self.advance();
        }
        
        let text: String = self.input[start..self.position].iter().collect();
        let suffix = self.lex_number_suffix();
        Ok(TokenKind::IntLiteral(text, suffix))
    }
    
    fn lex_octal_number(&mut self, start: usize, start_pos: Position) -> Result<TokenKind> {
        self.advance(); // consume 'o' or 'O'
        
        if self.is_at_end() || !matches!(self.current_char(), '0'..='7') {
            return Err(Error::new(
                ErrorKind::InvalidNumber,
                "Invalid octal literal"
            ).with_span(Span::new(start_pos, self.current_pos)));
        }
        
        while !self.is_at_end() && (matches!(self.current_char(), '0'..='7') || self.current_char() == '_') {
            self.advance();
        }
        
        let text: String = self.input[start..self.position].iter().collect();
        let suffix = self.lex_number_suffix();
        Ok(TokenKind::IntLiteral(text, suffix))
    }
    
    fn lex_binary_number(&mut self, start: usize, start_pos: Position) -> Result<TokenKind> {
        self.advance(); // consume 'b' or 'B'
        
        if self.is_at_end() || !matches!(self.current_char(), '0' | '1') {
            return Err(Error::new(
                ErrorKind::InvalidNumber,
                "Invalid binary literal"
            ).with_span(Span::new(start_pos, self.current_pos)));
        }
        
        while !self.is_at_end() && (matches!(self.current_char(), '0' | '1') || self.current_char() == '_') {
            self.advance();
        }
        
        let text: String = self.input[start..self.position].iter().collect();
        let suffix = self.lex_number_suffix();
        Ok(TokenKind::IntLiteral(text, suffix))
    }
    
    fn lex_number_suffix(&mut self) -> Option<String> {
        if self.is_at_end() {
            return None;
        }
        
        let start = self.position;
        
        // Check for type suffix (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
        match self.current_char() {
            'i' | 'u' | 'f' => {
                self.advance();
                
                // Parse the number part
                if !self.is_at_end() {
                    if self.current_char() == 's' {
                        // isize or usize
                        self.advance();
                        if !self.is_at_end() && self.current_char() == 'i' {
                            self.advance();
                            if !self.is_at_end() && self.current_char() == 'z' {
                                self.advance();
                                if !self.is_at_end() && self.current_char() == 'e' {
                                    self.advance();
                                    let suffix: String = self.input[start..self.position].iter().collect();
                                    return Some(suffix);
                                }
                            }
                        }
                    } else if self.current_char().is_ascii_digit() {
                        while !self.is_at_end() && self.current_char().is_ascii_digit() {
                            self.advance();
                        }
                        let suffix: String = self.input[start..self.position].iter().collect();
                        return Some(suffix);
                    }
                }
                
                // Reset if not a valid suffix
                self.position = start;
                None
            }
            _ => None,
        }
    }
    
    fn lex_string(&mut self) -> Result<TokenKind> {
        let start_pos = self.current_pos;
        self.advance(); // Skip opening "
        
        let mut result = String::new();
        
        while !self.is_at_end() && self.current_char() != '"' {
            if self.current_char() == '\\' {
                self.advance();
                
                if self.is_at_end() {
                    return Err(Error::new(
                        ErrorKind::UnterminatedString,
                        "Unterminated string literal"
                    ).with_span(Span::new(start_pos, self.current_pos)));
                }
                
                let escaped = match self.current_char() {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    '0' => '\0',
                    'x' => {
                        // Hex escape: \xNN
                        self.advance();
                        let hex_start = self.position;
                        
                        if self.is_at_end() || !self.current_char().is_ascii_hexdigit() {
                            return Err(Error::new(
                                ErrorKind::InvalidEscape,
                                "Invalid hex escape sequence"
                            ).with_span(Span::new(start_pos, self.current_pos)));
                        }
                        self.advance();
                        
                        if self.is_at_end() || !self.current_char().is_ascii_hexdigit() {
                            return Err(Error::new(
                                ErrorKind::InvalidEscape,
                                "Invalid hex escape sequence"
                            ).with_span(Span::new(start_pos, self.current_pos)));
                        }
                        
                        let hex_str: String = self.input[hex_start..=self.position].iter().collect();
                        let code = u8::from_str_radix(&hex_str, 16).map_err(|_| {
                            Error::new(
                                ErrorKind::InvalidEscape,
                                "Invalid hex escape sequence"
                            ).with_span(Span::new(start_pos, self.current_pos))
                        })?;
                        
                        self.advance();
                        code as char
                    }
                    'u' => {
                        // Unicode escape: \u{NNNN}
                        self.advance();
                        
                        if self.is_at_end() || self.current_char() != '{' {
                            return Err(Error::new(
                                ErrorKind::InvalidEscape,
                                "Invalid unicode escape sequence, expected '{'"
                            ).with_span(Span::new(start_pos, self.current_pos)));
                        }
                        self.advance();
                        
                        let unicode_start = self.position;
                        while !self.is_at_end() && self.current_char() != '}' && self.current_char().is_ascii_hexdigit() {
                            self.advance();
                        }
                        
                        if self.is_at_end() || self.current_char() != '}' {
                            return Err(Error::new(
                                ErrorKind::InvalidEscape,
                                "Invalid unicode escape sequence, expected '}'"
                            ).with_span(Span::new(start_pos, self.current_pos)));
                        }
                        
                        let unicode_str: String = self.input[unicode_start..self.position].iter().collect();
                        let code = u32::from_str_radix(&unicode_str, 16).map_err(|_| {
                            Error::new(
                                ErrorKind::InvalidEscape,
                                "Invalid unicode escape sequence"
                            ).with_span(Span::new(start_pos, self.current_pos))
                        })?;
                        
                        let ch = char::from_u32(code).ok_or_else(|| {
                            Error::new(
                                ErrorKind::InvalidEscape,
                                "Invalid unicode code point"
                            ).with_span(Span::new(start_pos, self.current_pos))
                        })?;
                        
                        self.advance();
                        ch
                    }
                    _ => {
                        return Err(Error::new(
                            ErrorKind::InvalidEscape,
                            format!("Unknown escape sequence: \\{}", self.current_char())
                        ).with_span(Span::new(start_pos, self.current_pos)));
                    }
                };
                
                result.push(escaped);
            } else {
                result.push(self.current_char());
                self.advance();
            }
        }
        
        if self.is_at_end() {
            return Err(Error::new(
                ErrorKind::UnterminatedString,
                "Unterminated string literal"
            ).with_span(Span::new(start_pos, self.current_pos)));
        }
        
        self.advance(); // Skip closing "
        Ok(TokenKind::StringLiteral(result))
    }
    
    fn lex_raw_string(&mut self) -> Result<TokenKind> {
        let start_pos = self.current_pos;
        self.advance(); // Skip 'r'
        
        // Count hash marks
        let mut hash_count = 0;
        while !self.is_at_end() && self.current_char() == '#' {
            hash_count += 1;
            self.advance();
        }
        
        if self.is_at_end() || self.current_char() != '"' {
            return Err(Error::new(
                ErrorKind::InvalidCharacter,
                "Expected '\"' after raw string prefix"
            ).with_span(Span::new(start_pos, self.current_pos)));
        }
        
        self.advance(); // Skip opening "
        let mut result = String::new();
        
        // Read until closing delimiter
        loop {
            if self.is_at_end() {
                return Err(Error::new(
                    ErrorKind::UnterminatedString,
                    "Unterminated raw string literal"
                ).with_span(Span::new(start_pos, self.current_pos)));
            }
            
            if self.current_char() == '"' {
                // Check if followed by correct number of hashes
                let check_pos = self.position;
                self.advance();
                
                let mut found_hashes = 0;
                while !self.is_at_end() && self.current_char() == '#' && found_hashes < hash_count {
                    found_hashes += 1;
                    self.advance();
                }
                
                if found_hashes == hash_count {
                    // Found closing delimiter
                    return Ok(TokenKind::StringLiteral(result));
                } else {
                    // Not the closing delimiter, add to result
                    self.position = check_pos;
                    result.push('"');
                    self.advance();
                }
            } else {
                result.push(self.current_char());
                self.advance();
            }
        }
    }
    
    fn lex_char(&mut self) -> Result<TokenKind> {
        let start_pos = self.current_pos;
        self.advance(); // Skip opening '
        
        if self.is_at_end() {
            return Err(Error::new(
                ErrorKind::UnterminatedString,
                "Unterminated char literal"
            ).with_span(Span::new(start_pos, self.current_pos)));
        }
        
        let ch = if self.current_char() == '\\' {
            self.advance();
            
            if self.is_at_end() {
                return Err(Error::new(
                    ErrorKind::UnterminatedString,
                    "Unterminated char literal"
                ).with_span(Span::new(start_pos, self.current_pos)));
            }
            
            match self.current_char() {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '\\' => '\\',
                '"' => '"',
                '\'' => '\'',
                '0' => '\0',
                'x' => {
                    // Hex escape: \xNN
                    self.advance();
                    let hex_start = self.position;
                    
                    if self.is_at_end() || !self.current_char().is_ascii_hexdigit() {
                        return Err(Error::new(
                            ErrorKind::InvalidEscape,
                            "Invalid hex escape in char literal"
                        ).with_span(Span::new(start_pos, self.current_pos)));
                    }
                    self.advance();
                    
                    if self.is_at_end() || !self.current_char().is_ascii_hexdigit() {
                        return Err(Error::new(
                            ErrorKind::InvalidEscape,
                            "Invalid hex escape in char literal"
                        ).with_span(Span::new(start_pos, self.current_pos)));
                    }
                    
                    let hex_str: String = self.input[hex_start..=self.position].iter().collect();
                    let code = u8::from_str_radix(&hex_str, 16).map_err(|_| {
                        Error::new(
                            ErrorKind::InvalidEscape,
                            "Invalid hex escape in char literal"
                        ).with_span(Span::new(start_pos, self.current_pos))
                    })?;
                    
                    code as char
                }
                'u' => {
                    // Unicode escape: \u{NNNN}
                    self.advance();
                    
                    if self.is_at_end() || self.current_char() != '{' {
                        return Err(Error::new(
                            ErrorKind::InvalidEscape,
                            "Invalid unicode escape in char literal"
                        ).with_span(Span::new(start_pos, self.current_pos)));
                    }
                    self.advance();
                    
                    let unicode_start = self.position;
                    while !self.is_at_end() && self.current_char() != '}' && self.current_char().is_ascii_hexdigit() {
                        self.advance();
                    }
                    
                    if self.is_at_end() || self.current_char() != '}' {
                        return Err(Error::new(
                            ErrorKind::InvalidEscape,
                            "Invalid unicode escape in char literal"
                        ).with_span(Span::new(start_pos, self.current_pos)));
                    }
                    
                    let unicode_str: String = self.input[unicode_start..self.position].iter().collect();
                    let code = u32::from_str_radix(&unicode_str, 16).map_err(|_| {
                        Error::new(
                            ErrorKind::InvalidEscape,
                            "Invalid unicode escape in char literal"
                        ).with_span(Span::new(start_pos, self.current_pos))
                    })?;
                    
                    char::from_u32(code).ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidEscape,
                            "Invalid unicode code point"
                        ).with_span(Span::new(start_pos, self.current_pos))
                    })?
                }
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidEscape,
                        format!("Unknown escape sequence in char literal: \\{}", self.current_char())
                    ).with_span(Span::new(start_pos, self.current_pos)));
                }
            }
        } else {
            self.current_char()
        };
        
        self.advance();
        
        if self.is_at_end() || self.current_char() != '\'' {
            return Err(Error::new(
                ErrorKind::InvalidCharacter,
                "Invalid char literal, expected closing '"
            ).with_span(Span::new(start_pos, self.current_pos)));
        }
        
        self.advance(); // Skip closing '
        Ok(TokenKind::CharLiteral(ch))
    }
    
    fn lex_plus(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            TokenKind::PlusEq
        } else {
            TokenKind::Plus
        }
    }
    
    fn lex_minus(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '>' {
            self.advance();
            TokenKind::Arrow
        } else if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            TokenKind::MinusEq
        } else {
            TokenKind::Minus
        }
    }
    
    fn lex_star(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '*' {
            self.advance();
            TokenKind::StarStar
        } else if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            TokenKind::StarEq
        } else {
            TokenKind::Star
        }
    }
    
    fn lex_slash(&mut self) -> Result<TokenKind> {
        self.advance();
        if !self.is_at_end() && self.current_char() == '/' {
            // Line comment - skip to end of line
            while !self.is_at_end() && self.current_char() != '\n' {
                self.advance();
            }
            // Skip whitespace before getting next token
            self.skip_whitespace();
            if self.is_at_end() {
                return Ok(TokenKind::Eof);
            }
            self.next_token().map(|t| t.kind)
        } else if !self.is_at_end() && self.current_char() == '*' {
            // Block comment
            self.advance();
            while !self.is_at_end() {
                if self.current_char() == '*' {
                    self.advance();
                    if !self.is_at_end() && self.current_char() == '/' {
                        self.advance();
                        // Skip whitespace before getting next token
                        self.skip_whitespace();
                        if self.is_at_end() {
                            return Ok(TokenKind::Eof);
                        }
                        return self.next_token().map(|t| t.kind);
                    }
                } else {
                    self.advance();
                }
            }
            Err(Error::new(ErrorKind::UnterminatedString, "Unterminated block comment"))
        } else if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            Ok(TokenKind::SlashEq)
        } else {
            Ok(TokenKind::Slash)
        }
    }
    
    fn lex_equals(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            TokenKind::EqEq
        } else if !self.is_at_end() && self.current_char() == '>' {
            self.advance();
            TokenKind::FatArrow
        } else {
            TokenKind::Eq
        }
    }
    
    fn lex_bang(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            TokenKind::Ne
        } else {
            TokenKind::Bang
        }
    }
    
    fn lex_less(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            TokenKind::Le
        } else if !self.is_at_end() && self.current_char() == '<' {
            self.advance();
            TokenKind::Shl
        } else {
            TokenKind::Lt
        }
    }
    
    fn lex_greater(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '=' {
            self.advance();
            TokenKind::Ge
        } else if !self.is_at_end() && self.current_char() == '>' {
            self.advance();
            TokenKind::Shr
        } else {
            TokenKind::Gt
        }
    }
    
    fn lex_ampersand(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '&' {
            self.advance();
            TokenKind::AndAnd
        } else {
            TokenKind::And
        }
    }
    
    fn lex_pipe(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '|' {
            self.advance();
            TokenKind::OrOr
        } else {
            TokenKind::Pipe
        }
    }
    
    fn lex_dot(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == '.' {
            self.advance();
            if !self.is_at_end() && self.current_char() == '=' {
                self.advance();
                TokenKind::DotDotEq
            } else {
                TokenKind::DotDot
            }
        } else {
            TokenKind::Dot
        }
    }
    
    fn lex_colon(&mut self) -> TokenKind {
        self.advance();
        if !self.is_at_end() && self.current_char() == ':' {
            self.advance();
            TokenKind::ColonColon
        } else {
            TokenKind::Colon
        }
    }
    
    fn skip_whitespace(&mut self) {
        while !self.is_at_end() && self.current_char().is_whitespace() {
            self.advance();
        }
    }
    
    fn is_identifier_continue(&self, ch: char) -> bool {
        ch.is_alphanumeric() || ch == '_'
    }
    
    fn current_char(&self) -> char {
        self.input[self.position]
    }
    
    fn advance(&mut self) {
        if !self.is_at_end() {
            let ch = self.input[self.position];
            self.position += 1;
            
            if ch == '\n' {
                self.current_pos.line += 1;
                self.current_pos.column = 1;
            } else {
                self.current_pos.column += 1;
            }
            self.current_pos.byte_offset += ch.len_utf8();
        }
    }
    
    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
    
    fn make_token(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            span: Span::new(self.current_pos, self.current_pos),
        }
    }
}
