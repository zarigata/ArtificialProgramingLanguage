// Macro Expander
// Expands macro invocations into AST nodes

use super::*;
use crate::parser::ast::*;
use crate::error::{Error, Result};

pub struct MacroExpander {
    registry: MacroRegistry,
    expansion_depth: usize,
    max_depth: usize,
}

impl MacroExpander {
    pub fn new(registry: MacroRegistry) -> Self {
        MacroExpander {
            registry,
            expansion_depth: 0,
            max_depth: 128,
        }
    }
    
    pub fn expand_expr(&mut self, expr: &Expr) -> Result<Expr> {
        match expr {
            Expr::MacroCall { name, args, span } => {
                self.expand_macro_call(name, args, *span)
            }
            Expr::Binary { op, left, right, span } => {
                let left = Box::new(self.expand_expr(left)?);
                let right = Box::new(self.expand_expr(right)?);
                Ok(Expr::Binary {
                    op: *op,
                    left,
                    right,
                    span: *span,
                })
            }
            Expr::Call { func, args, span } => {
                let func = Box::new(self.expand_expr(func)?);
                let args = args.iter()
                    .map(|arg| self.expand_expr(arg))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Expr::Call {
                    func,
                    args,
                    span: *span,
                })
            }
            Expr::Block { stmts, span } => {
                let stmts = stmts.iter()
                    .map(|stmt| self.expand_stmt(stmt))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Expr::Block {
                    stmts,
                    span: *span,
                })
            }
            _ => Ok(expr.clone()),
        }
    }
    
    pub fn expand_stmt(&mut self, stmt: &Stmt) -> Result<Stmt> {
        match stmt {
            Stmt::Expr(expr) => {
                Ok(Stmt::Expr(self.expand_expr(expr)?))
            }
            Stmt::Let { pattern, ty, init, span } => {
                let init = init.as_ref()
                    .map(|e| self.expand_expr(e))
                    .transpose()?
                    .map(Box::new);
                Ok(Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init,
                    span: *span,
                })
            }
            _ => Ok(stmt.clone()),
        }
    }
    
    fn expand_macro_call(&mut self, name: &str, args: &[Expr], span: Span) -> Result<Expr> {
        if self.expansion_depth >= self.max_depth {
            return Err(Error::new(
                format!("macro expansion depth exceeded (max: {})", self.max_depth),
                span,
            ));
        }
        
        let macro_def = self.registry.get(name)
            .ok_or_else(|| Error::new(format!("undefined macro: {}", name), span))?;
        
        self.expansion_depth += 1;
        let result = self.expand_macro_def(macro_def, args, span);
        self.expansion_depth -= 1;
        
        result
    }
    
    fn expand_macro_def(&mut self, def: &MacroDef, args: &[Expr], span: Span) -> Result<Expr> {
        match &def.body {
            MacroBody::Rules(rules) => {
                for rule in rules {
                    if let Some(bindings) = self.match_pattern(&rule.pattern, args) {
                        return self.expand_rule(&rule.expansion, &bindings, span);
                    }
                }
                Err(Error::new(
                    format!("no matching macro rule for {}", def.name),
                    span,
                ))
            }
            MacroBody::Procedural(_path) => {
                // Procedural macros would be implemented here
                Err(Error::new("procedural macros not yet implemented", span))
            }
        }
    }
    
    fn match_pattern(&self, pattern: &[MacroToken], args: &[Expr]) -> Option<HashMap<String, Vec<Expr>>> {
        let mut bindings = HashMap::new();
        let mut arg_idx = 0;
        
        for token in pattern {
            match token {
                MacroToken::Variable(name, _kind) => {
                    if arg_idx >= args.len() {
                        return None;
                    }
                    bindings.entry(name.clone())
                        .or_insert_with(Vec::new)
                        .push(args[arg_idx].clone());
                    arg_idx += 1;
                }
                MacroToken::Repeat(inner, kind) => {
                    let mut repeat_bindings = Vec::new();
                    
                    loop {
                        if arg_idx >= args.len() {
                            break;
                        }
                        
                        // Try to match inner pattern
                        if let Some(inner_bindings) = self.match_pattern(inner, &args[arg_idx..]) {
                            repeat_bindings.push(inner_bindings);
                            arg_idx += 1;
                        } else {
                            break;
                        }
                    }
                    
                    // Check repeat kind constraints
                    match kind {
                        RepeatKind::OneOrMore if repeat_bindings.is_empty() => return None,
                        _ => {}
                    }
                    
                    // Merge repeat bindings
                    for inner_bindings in repeat_bindings {
                        for (name, exprs) in inner_bindings {
                            bindings.entry(name)
                                .or_insert_with(Vec::new)
                                .extend(exprs);
                        }
                    }
                }
                MacroToken::Literal(_) => {
                    // Skip literal tokens in pattern matching
                }
                MacroToken::Group(_) => {
                    // Groups would be handled here
                }
            }
        }
        
        if arg_idx == args.len() {
            Some(bindings)
        } else {
            None
        }
    }
    
    fn expand_rule(&mut self, expansion: &[MacroToken], bindings: &HashMap<String, Vec<Expr>>, span: Span) -> Result<Expr> {
        let mut code = String::new();
        
        for token in expansion {
            match token {
                MacroToken::Literal(s) => {
                    code.push_str(s);
                }
                MacroToken::Variable(name, _kind) => {
                    if let Some(exprs) = bindings.get(name) {
                        if let Some(expr) = exprs.first() {
                            code.push_str(&format!("{:?}", expr));
                        }
                    }
                }
                MacroToken::Repeat(inner, _kind) => {
                    // Expand repeated tokens
                    for token in inner {
                        match token {
                            MacroToken::Variable(name, _) => {
                                if let Some(exprs) = bindings.get(name) {
                                    for expr in exprs {
                                        code.push_str(&format!("{:?}", expr));
                                    }
                                }
                            }
                            MacroToken::Literal(s) => {
                                code.push_str(s);
                            }
                            _ => {}
                        }
                    }
                }
                MacroToken::Group(_) => {}
            }
        }
        
        // Parse the expanded code back into an expression
        // This is simplified - in production, we'd use the actual parser
        Ok(Expr::Block {
            stmts: vec![],
            span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_macro_expansion() {
        let registry = MacroRegistry::new();
        let mut expander = MacroExpander::new(registry);
        
        // Test vec! macro expansion
        let expr = Expr::MacroCall {
            name: "vec".to_string(),
            args: vec![],
            span: Span::dummy(),
        };
        
        let result = expander.expand_expr(&expr);
        assert!(result.is_ok());
    }
}
