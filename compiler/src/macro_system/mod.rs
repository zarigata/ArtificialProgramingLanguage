// VeZ Macro System
// Compile-time metaprogramming with hygiene and expansion

pub mod expander;
pub mod hygiene;
pub mod parser;
pub mod builtin;
pub mod reflection;

use crate::span::Span;
use std::collections::HashMap;

// Macro definition
#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: String,
    pub params: Vec<MacroParam>,
    pub body: MacroBody,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct MacroParam {
    pub name: String,
    pub kind: MacroParamKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MacroParamKind {
    Expr,      // Expression
    Stmt,      // Statement
    Type,      // Type
    Pat,       // Pattern
    Ident,     // Identifier
    Block,     // Block
    Item,      // Item (fn, struct, etc.)
    Tt,        // Token tree
}

#[derive(Debug, Clone)]
pub enum MacroBody {
    Rules(Vec<MacroRule>),
    Procedural(String), // Path to procedural macro
}

#[derive(Debug, Clone)]
pub struct MacroRule {
    pub pattern: Vec<MacroToken>,
    pub expansion: Vec<MacroToken>,
}

#[derive(Debug, Clone)]
pub enum MacroToken {
    Literal(String),
    Variable(String, MacroParamKind),
    Repeat(Vec<MacroToken>, RepeatKind),
    Group(Vec<MacroToken>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RepeatKind {
    ZeroOrMore,  // *
    OneOrMore,   // +
    ZeroOrOne,   // ?
}

// Macro registry
pub struct MacroRegistry {
    macros: HashMap<String, MacroDef>,
}

impl MacroRegistry {
    pub fn new() -> Self {
        let mut registry = MacroRegistry {
            macros: HashMap::new(),
        };
        
        // Register built-in macros
        registry.register_builtin_macros();
        registry
    }
    
    pub fn register(&mut self, name: String, def: MacroDef) {
        self.macros.insert(name, def);
    }
    
    pub fn get(&self, name: &str) -> Option<&MacroDef> {
        self.macros.get(name)
    }
    
    fn register_builtin_macros(&mut self) {
        // vec! macro
        self.register(
            "vec".to_string(),
            MacroDef {
                name: "vec".to_string(),
                params: vec![],
                body: MacroBody::Rules(vec![
                    MacroRule {
                        pattern: vec![],
                        expansion: vec![
                            MacroToken::Literal("Vec::new()".to_string()),
                        ],
                    },
                    MacroRule {
                        pattern: vec![
                            MacroToken::Variable("elems".to_string(), MacroParamKind::Expr),
                            MacroToken::Repeat(
                                vec![
                                    MacroToken::Literal(",".to_string()),
                                    MacroToken::Variable("elems".to_string(), MacroParamKind::Expr),
                                ],
                                RepeatKind::ZeroOrMore,
                            ),
                        ],
                        expansion: vec![
                            MacroToken::Literal("{".to_string()),
                            MacroToken::Literal("let mut v = Vec::new();".to_string()),
                            MacroToken::Repeat(
                                vec![
                                    MacroToken::Literal("v.push(".to_string()),
                                    MacroToken::Variable("elems".to_string(), MacroParamKind::Expr),
                                    MacroToken::Literal(");".to_string()),
                                ],
                                RepeatKind::OneOrMore,
                            ),
                            MacroToken::Literal("v".to_string()),
                            MacroToken::Literal("}".to_string()),
                        ],
                    },
                ]),
                span: Span::dummy(),
            },
        );
        
        // println! macro
        self.register(
            "println".to_string(),
            MacroDef {
                name: "println".to_string(),
                params: vec![],
                body: MacroBody::Rules(vec![
                    MacroRule {
                        pattern: vec![],
                        expansion: vec![
                            MacroToken::Literal("stdout().write_str(\"\\n\")".to_string()),
                        ],
                    },
                    MacroRule {
                        pattern: vec![
                            MacroToken::Variable("fmt".to_string(), MacroParamKind::Expr),
                        ],
                        expansion: vec![
                            MacroToken::Literal("stdout().write_str(&format!(".to_string()),
                            MacroToken::Variable("fmt".to_string(), MacroParamKind::Expr),
                            MacroToken::Literal(", \"\\n\"))".to_string()),
                        ],
                    },
                    MacroRule {
                        pattern: vec![
                            MacroToken::Variable("fmt".to_string(), MacroParamKind::Expr),
                            MacroToken::Literal(",".to_string()),
                            MacroToken::Variable("args".to_string(), MacroParamKind::Expr),
                            MacroToken::Repeat(
                                vec![
                                    MacroToken::Literal(",".to_string()),
                                    MacroToken::Variable("args".to_string(), MacroParamKind::Expr),
                                ],
                                RepeatKind::ZeroOrMore,
                            ),
                        ],
                        expansion: vec![
                            MacroToken::Literal("stdout().write_str(&format!(".to_string()),
                            MacroToken::Variable("fmt".to_string(), MacroParamKind::Expr),
                            MacroToken::Literal(",".to_string()),
                            MacroToken::Repeat(
                                vec![
                                    MacroToken::Variable("args".to_string(), MacroParamKind::Expr),
                                    MacroToken::Literal(",".to_string()),
                                ],
                                RepeatKind::OneOrMore,
                            ),
                            MacroToken::Literal("\"\\n\"))".to_string()),
                        ],
                    },
                ]),
                span: Span::dummy(),
            },
        );
        
        // assert! macro
        self.register(
            "assert".to_string(),
            MacroDef {
                name: "assert".to_string(),
                params: vec![],
                body: MacroBody::Rules(vec![
                    MacroRule {
                        pattern: vec![
                            MacroToken::Variable("cond".to_string(), MacroParamKind::Expr),
                        ],
                        expansion: vec![
                            MacroToken::Literal("if !(".to_string()),
                            MacroToken::Variable("cond".to_string(), MacroParamKind::Expr),
                            MacroToken::Literal(") { panic!(\"assertion failed\") }".to_string()),
                        ],
                    },
                ]),
                span: Span::dummy(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_macro_registry() {
        let registry = MacroRegistry::new();
        assert!(registry.get("vec").is_some());
        assert!(registry.get("println").is_some());
        assert!(registry.get("assert").is_some());
    }
}
