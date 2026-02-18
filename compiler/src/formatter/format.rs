//! Main formatter implementation

use super::config::{FormatterConfig, BraceStyle, IndentStyle};
use super::indent::IndentManager;
use super::comment::CommentFormatter;

pub struct Formatter {
    config: FormatterConfig,
    indent: IndentManager,
    output: String,
    current_line: String,
    line_length: usize,
    pending_space: bool,
    in_string: bool,
    in_comment: bool,
    paren_depth: usize,
    brace_depth: usize,
    bracket_depth: usize,
}

impl Formatter {
    pub fn new(config: &FormatterConfig) -> Self {
        Formatter {
            config: config.clone(),
            indent: IndentManager::new(config.indent_style),
            output: String::new(),
            current_line: String::new(),
            line_length: 0,
            pending_space: false,
            in_string: false,
            in_comment: false,
            paren_depth: 0,
            brace_depth: 0,
            bracket_depth: 0,
        }
    }

    pub fn format(&mut self, source: &str) -> super::FormatResult<String> {
        let tokens = self.tokenize(source);
        self.format_tokens(&tokens);
        Ok(self.output.clone())
    }

    fn tokenize(&self, source: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = source.chars().peekable();
        let mut current_token = String::new();
        let mut start_pos = 0;
        let mut pos = 0;

        while let Some(c) = chars.next() {
            pos += 1;

            if c.is_whitespace() {
                if !current_token.is_empty() {
                    tokens.push(Token::new(&current_token, start_pos));
                    current_token.clear();
                }

                if c == '\n' {
                    tokens.push(Token::new("\n", pos - 1));
                }
                start_pos = pos;
                continue;
            }

            if c == '#' && !self.in_string {
                if !current_token.is_empty() {
                    tokens.push(Token::new(&current_token, start_pos));
                    current_token.clear();
                }

                let mut comment = String::new();
                comment.push(c);
                while let Some(&next) = chars.peek() {
                    if next == '\n' {
                        break;
                    }
                    comment.push(chars.next().unwrap());
                    pos += 1;
                }
                tokens.push(Token::new(&comment, start_pos));
                start_pos = pos + 1;
                continue;
            }

            if c == '"' || c == '\'' {
                if !current_token.is_empty() {
                    tokens.push(Token::new(&current_token, start_pos));
                    current_token.clear();
                }

                let quote = c;
                let mut string_lit = String::new();
                string_lit.push(c);
                
                while let Some(&next) = chars.peek() {
                    pos += 1;
                    let next = chars.next().unwrap();
                    string_lit.push(next);
                    if next == quote {
                        let prev_chars: Vec<char> = string_lit.chars().collect();
                        let backslash_count = prev_chars.iter().rev().skip(1).take_while(|&&c| c == '\\').count();
                        if backslash_count % 2 == 0 {
                            break;
                        }
                    }
                }
                
                tokens.push(Token::new(&string_lit, start_pos));
                start_pos = pos + 1;
                continue;
            }

            if is_punctuation(c) {
                if !current_token.is_empty() {
                    tokens.push(Token::new(&current_token, start_pos));
                    current_token.clear();
                }
                tokens.push(Token::new(&c.to_string(), pos - 1));
                start_pos = pos;
                continue;
            }

            if current_token.is_empty() {
                start_pos = pos - 1;
            }
            current_token.push(c);
        }

        if !current_token.is_empty() {
            tokens.push(Token::new(&current_token, start_pos));
        }

        tokens
    }

    fn format_tokens(&mut self, tokens: &[Token]) {
        let mut i = 0;
        
        while i < tokens.len() {
            let token = &tokens[i];
            
            match token.kind {
                TokenKind::Newline => {
                    self.handle_newline(&tokens[i..]);
                }
                TokenKind::Comment => {
                    self.handle_comment(&token.value);
                }
                TokenKind::Keyword => {
                    self.handle_keyword(&token.value);
                }
                TokenKind::Punctuation => {
                    let next = tokens.get(i + 1);
                    self.handle_punctuation(&token.value, next);
                }
                TokenKind::String | TokenKind::Char => {
                    self.handle_string(&token.value);
                }
                _ => {
                    self.handle_identifier(&token.value);
                }
            }
            
            i += 1;
        }

        if !self.current_line.is_empty() {
            self.flush_line();
        }
    }

    fn handle_newline(&mut self, _tokens: &[Token]) {
        if self.pending_space && !self.current_line.is_empty() {
            self.current_line.push(' ');
            self.line_length += 1;
        }
        self.flush_line();
        self.pending_space = false;
    }

    fn handle_comment(&mut self, comment: &str) {
        if !self.current_line.is_empty() {
            self.current_line.push(' ');
        }
        let formatted = CommentFormatter::format(comment, &self.config);
        self.current_line.push_str(&formatted);
        self.flush_line();
        self.pending_space = false;
    }

    fn handle_keyword(&mut self, keyword: &str) {
        if self.pending_space {
            self.current_line.push(' ');
            self.line_length += 1;
        }
        self.current_line.push_str(keyword);
        self.line_length += keyword.len();
        self.pending_space = true;
    }

    fn handle_identifier(&mut self, ident: &str) {
        if self.pending_space {
            self.current_line.push(' ');
            self.line_length += 1;
        }
        self.current_line.push_str(ident);
        self.line_length += ident.len();
        self.pending_space = true;
    }

    fn handle_punctuation(&mut self, punct: &str, next_token: Option<&Token>) {
        match punct {
            "(" => {
                self.paren_depth += 1;
                if self.pending_space && self.config.space_around_operators {
                    self.current_line.push(' ');
                    self.line_length += 1;
                }
                self.current_line.push('(');
                self.line_length += 1;
                self.pending_space = false;
            }
            ")" => {
                self.paren_depth = self.paren_depth.saturating_sub(1);
                self.current_line.push(')');
                self.line_length += 1;
                self.pending_space = true;
            }
            "{" => {
                self.brace_depth += 1;
                if self.config.brace_style == BraceStyle::NextLine {
                    self.flush_line();
                    self.current_line.push_str(&self.indent.current());
                    self.line_length = self.indent.current_width();
                } else if self.pending_space {
                    self.current_line.push(' ');
                    self.line_length += 1;
                }
                self.current_line.push('{');
                self.line_length += 1;
                self.flush_line();
                self.indent.increase();
                self.current_line.push_str(&self.indent.current());
                self.line_length = self.indent.current_width();
                self.pending_space = false;
            }
            "}" => {
                self.brace_depth = self.brace_depth.saturating_sub(1);
                self.flush_line();
                self.indent.decrease();
                self.current_line.push_str(&self.indent.current());
                self.line_length = self.indent.current_width();
                self.current_line.push('}');
                self.line_length += 1;
                self.pending_space = true;
            }
            "[" => {
                self.bracket_depth += 1;
                self.current_line.push('[');
                self.line_length += 1;
                self.pending_space = false;
            }
            "]" => {
                self.bracket_depth = self.bracket_depth.saturating_sub(1);
                self.current_line.push(']');
                self.line_length += 1;
                self.pending_space = true;
            }
            "," => {
                self.current_line.push(',');
                self.line_length += 1;
                self.pending_space = true;
            }
            ":" => {
                if self.config.space_before_colon && self.pending_space {
                    self.current_line.push(' ');
                    self.line_length += 1;
                }
                self.current_line.push(':');
                self.line_length += 1;
                self.pending_space = self.config.space_after_colon;
            }
            "=" | "+" | "-" | "*" | "/" | "%" | "<" | ">" | "!" | "&" | "|" | "^" => {
                let is_compound = next_token.map_or(false, |t| t.kind == TokenKind::Punctuation && t.value == "=");
                let is_double = next_token.map_or(false, |t| t.value == punct);
                
                if is_compound || is_double {
                    if self.config.space_around_operators && self.pending_space {
                        self.current_line.push(' ');
                        self.line_length += 1;
                    }
                    self.current_line.push_str(punct);
                    self.line_length += punct.len();
                    self.pending_space = false;
                } else {
                    if self.config.space_around_operators && self.pending_space {
                        self.current_line.push(' ');
                        self.line_length += 1;
                    }
                    self.current_line.push_str(punct);
                    self.line_length += punct.len();
                    self.pending_space = true;
                }
            }
            "." => {
                self.pending_space = false;
                self.current_line.push('.');
                self.line_length += 1;
            }
            _ => {
                self.current_line.push_str(punct);
                self.line_length += punct.len();
                self.pending_space = true;
            }
        }
    }

    fn handle_string(&mut self, s: &str) {
        if self.pending_space {
            self.current_line.push(' ');
            self.line_length += 1;
        }
        self.current_line.push_str(s);
        self.line_length += s.len();
        self.pending_space = true;
    }

    fn flush_line(&mut self) {
        let line = self.current_line.trim_end();
        if !line.is_empty() {
            self.output.push_str(line);
        }
        self.output.push('\n');
        self.current_line.clear();
        self.line_length = 0;
    }
}

#[derive(Debug, Clone)]
struct Token {
    value: String,
    kind: TokenKind,
    pos: usize,
}

impl Token {
    fn new(value: &str, pos: usize) -> Self {
        let kind = Self::classify(value);
        Token {
            value: value.to_string(),
            kind,
            pos,
        }
    }

    fn classify(value: &str) -> TokenKind {
        if value == "\n" {
            return TokenKind::Newline;
        }
        if value.starts_with('#') {
            return TokenKind::Comment;
        }
        if value.starts_with('"') || value.starts_with('\'') {
            return if value.starts_with('"') { TokenKind::String } else { TokenKind::Char };
        }
        if is_keyword(value) {
            return TokenKind::Keyword;
        }
        if value.len() == 1 && is_punctuation(value.chars().next().unwrap()) {
            return TokenKind::Punctuation;
        }
        if value.chars().all(|c| c.is_ascii_digit() || c == '.' || c == 'x' || c == 'X') {
            return TokenKind::Number;
        }
        TokenKind::Identifier
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Identifier,
    Keyword,
    Number,
    String,
    Char,
    Punctuation,
    Comment,
    Newline,
}

fn is_keyword(s: &str) -> bool {
    matches!(s, 
        "def" | "fn" | "let" | "const" | "var" | "if" | "else" | "elif" |
        "while" | "for" | "in" | "return" | "break" | "continue" |
        "struct" | "enum" | "trait" | "impl" | "type" | "class" |
        "import" | "export" | "from" | "as" | "pub" | "private" |
        "async" | "await" | "try" | "catch" | "finally" | "throw" |
        "match" | "case" | "default" | "True" | "False" | "None" |
        "and" | "or" | "not" | "is" | "in" | "pass" | "lambda" |
        "with" | "yield" | "global" | "nonlocal" | "assert" |
        "raise" | "del" | "except" | "finally" | "region"
    )
}

fn is_punctuation(c: char) -> bool {
    matches!(c,
        '(' | ')' | '{' | '}' | '[' | ']' | '<' | '>' |
        ':' | ';' | ',' | '.' | '=' | '+' | '-' | '*' |
        '/' | '%' | '!' | '&' | '|' | '^' | '~' | '?' |
        '@' | '$' | '\\' | '#'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_simple() {
        let config = FormatterConfig::default();
        let mut formatter = Formatter::new(&config);
        
        let source = "def add(a:int,b:int)->int:return a+b";
        let result = formatter.format(source).unwrap();
        
        assert!(result.contains("def add(a: int, b: int) -> int:"));
    }

    #[test]
    fn test_format_with_spaces() {
        let config = FormatterConfig::default();
        let mut formatter = Formatter::new(&config);
        
        let source = "x=1+2*3";
        let result = formatter.format(source).unwrap();
        
        assert!(result.contains("x = 1 + 2 * 3"));
    }

    #[test]
    fn test_format_indentation() {
        let config = FormatterConfig::default();
        let mut formatter = Formatter::new(&config);
        
        let source = "def foo():\nbar()";
        let result = formatter.format(source).unwrap();
        
        assert!(result.contains("def foo():"));
        assert!(result.contains("    bar()"));
    }
}
