//! IR builder for converting AST to SSA form

use crate::parser::*;
use crate::error::{Error, ErrorKind, Result};
use super::ssa::{Function as IrFunction, BasicBlock, Value, ValueId, Constant, Module};
use super::types::IrType;
use super::instructions::{Instruction, BinaryOp, UnaryOp};
use std::collections::HashMap;

/// IR builder
pub struct IrBuilder {
    module: Module,
    current_function: Option<IrFunction>,
    current_block: Option<usize>,
    variable_map: HashMap<String, ValueId>,
}

impl IrBuilder {
    pub fn new(module_name: String) -> Self {
        IrBuilder {
            module: Module::new(module_name),
            current_function: None,
            current_block: None,
            variable_map: HashMap::new(),
        }
    }
    
    /// Build IR from a program
    pub fn build_program(mut self, program: &Program) -> Result<Module> {
        for item in &program.items {
            self.build_item(item)?;
        }
        Ok(self.module)
    }
    
    fn build_item(&mut self, item: &Item) -> Result<()> {
        match item {
            Item::Function(func) => self.build_function(func),
            Item::Struct(_) | Item::Enum(_) | Item::Trait(_) => {
                // Type definitions don't generate IR directly
                Ok(())
            }
            Item::Impl(impl_block) => {
                for item in &impl_block.items {
                    if let ImplItem::Function(func) = item {
                        self.build_function(func)?;
                    }
                }
                Ok(())
            }
            Item::Use(_) | Item::Mod(_, _) => Ok(()),
        }
    }
    
    fn build_function(&mut self, func: &Function) -> Result<()> {
        // Convert parameter types (with names)
        let params: Vec<(String, IrType)> = func.params.iter()
            .map(|p| (p.name.clone(), self.convert_type(&p.ty)))
            .collect();

        // Convert return type
        let return_type = func.return_type.as_ref()
            .map(|ty| self.convert_type(ty))
            .unwrap_or(IrType::Void);

        // Create IR function
        let mut ir_func = IrFunction::new(func.name.clone(), params, return_type);
        
        // Create entry block
        let entry_block = ir_func.add_named_block("entry".to_string());
        
        self.current_function = Some(ir_func);
        self.current_block = Some(entry_block);
        self.variable_map.clear();
        
        // Map parameters to values
        for (i, param) in func.params.iter().enumerate() {
            self.variable_map.insert(param.name.clone(), ValueId(i));
        }
        
        // Build function body
        for stmt in &func.body {
            self.build_stmt(stmt)?;
        }
        
        // Ensure block is terminated
        if let Some(func) = &mut self.current_function {
            if let Some(block_id) = self.current_block {
                // Check if block needs termination
                let needs_termination = func.get_block(block_id)
                    .map(|b| !b.is_terminated())
                    .unwrap_or(false);
                
                if needs_termination {
                    // Add implicit return
                    let ret_inst = Instruction::Return { value: None };
                    let value_id = func.add_value(Value::Instruction(ret_inst.clone()));
                    if let Some(block) = func.get_block_mut(block_id) {
                        block.add_instruction(value_id, ret_inst);
                    }
                }
            }
        }
        
        // Move function to module
        if let Some(ir_func) = self.current_function.take() {
            self.module.add_function(ir_func);
        }
        
        self.current_block = None;
        
        Ok(())
    }
    
    fn build_stmt(&mut self, stmt: &Stmt) -> Result<Option<ValueId>> {
        match stmt {
            Stmt::Let(name, _ty, init) => {
                if let Some(expr) = init {
                    let value_id = self.build_expr(expr)?;
                    self.variable_map.insert(name.clone(), value_id);
                }
                Ok(None)
            }
            Stmt::Expr(expr) => {
                self.build_expr(expr)?;
                Ok(None)
            }
            Stmt::Return(expr_opt) => {
                let value = if let Some(expr) = expr_opt {
                    Some(self.build_expr(expr)?)
                } else {
                    None
                };
                
                let ret_inst = Instruction::Return { value };
                self.emit_instruction(ret_inst)?;
                Ok(None)
            }
        }
    }
    
    fn build_expr(&mut self, expr: &Expr) -> Result<ValueId> {
        match expr {
            Expr::Literal(lit) => self.build_literal(lit),
            
            Expr::Ident(name) => {
                self.variable_map.get(name)
                    .copied()
                    .ok_or_else(|| Error::new(
                        ErrorKind::UndefinedSymbol,
                        format!("Undefined variable: {}", name)
                    ))
            }
            
            Expr::Binary(left, op, right) => {
                let lhs = self.build_expr(left)?;
                let rhs = self.build_expr(right)?;
                
                let ir_op = self.convert_binop(op);
                let inst = Instruction::Binary {
                    op: ir_op,
                    lhs,
                    rhs,
                    ty: IrType::I32, // Simplified - would need type info
                };
                
                self.emit_instruction(inst)
            }
            
            Expr::Unary(op, operand) => {
                let operand_val = self.build_expr(operand)?;
                
                let ir_op = self.convert_unop(op);
                let inst = Instruction::Unary {
                    op: ir_op,
                    operand: operand_val,
                    ty: IrType::I32, // Simplified
                };
                
                self.emit_instruction(inst)
            }
            
            Expr::Call(func_expr, args) => {
                let func_val = self.build_expr(func_expr)?;
                let arg_vals: Result<Vec<_>> = args.iter()
                    .map(|arg| self.build_expr(arg))
                    .collect();
                let arg_vals = arg_vals?;
                
                let inst = Instruction::Call {
                    func: func_val,
                    args: arg_vals,
                    ty: IrType::Void, // Simplified
                };
                
                self.emit_instruction(inst)
            }
            
            Expr::Block(stmts) => {
                let mut last_val = None;
                for stmt in stmts {
                    last_val = self.build_stmt(stmt)?;
                }
                
                // Return unit value if no expression
                last_val.ok_or_else(|| Error::new(
                    ErrorKind::InvalidSyntax,
                    "Block must have a value"
                ))
            }
            
            Expr::If(cond, then_branch, else_branch) => {
                let cond_val = self.build_expr(cond)?;
                
                let func = self.current_function.as_mut()
                    .ok_or_else(|| Error::new(ErrorKind::InvalidSyntax, "No current function"))?;
                
                let then_block = func.add_named_block("if.then".to_string());
                let else_block = func.add_named_block("if.else".to_string());
                let merge_block = func.add_named_block("if.merge".to_string());
                
                // Emit branch
                let branch_inst = Instruction::Branch {
                    cond: cond_val,
                    then_block,
                    else_block,
                };
                self.emit_instruction(branch_inst)?;
                
                // Build then branch
                self.current_block = Some(then_block);
                let then_val = self.build_expr(then_branch)?;
                let jump_inst = Instruction::Jump { target: merge_block };
                self.emit_instruction(jump_inst)?;
                
                // Build else branch
                self.current_block = Some(else_block);
                let else_val = if let Some(else_expr) = else_branch {
                    self.build_expr(else_expr)?
                } else {
                    // Create unit value
                    self.build_literal(&Literal::Int(0))?
                };
                let jump_inst = Instruction::Jump { target: merge_block };
                self.emit_instruction(jump_inst)?;
                
                // Merge block with phi
                self.current_block = Some(merge_block);
                let phi_inst = Instruction::Phi {
                    incoming: vec![(then_val, then_block), (else_val, else_block)],
                    ty: IrType::I32, // Simplified
                };
                
                self.emit_instruction(phi_inst)
            }
            
            Expr::Loop(body) => {
                let func = self.current_function.as_mut()
                    .ok_or_else(|| Error::new(ErrorKind::InvalidSyntax, "No current function"))?;
                
                let loop_block = func.add_named_block("loop.body".to_string());
                let exit_block = func.add_named_block("loop.exit".to_string());
                
                // Jump to loop
                let jump_inst = Instruction::Jump { target: loop_block };
                self.emit_instruction(jump_inst)?;
                
                // Build loop body
                self.current_block = Some(loop_block);
                self.build_expr(body)?;
                
                // Jump back to loop start
                let jump_inst = Instruction::Jump { target: loop_block };
                self.emit_instruction(jump_inst)?;
                
                // Exit block (unreachable but needed for CFG)
                self.current_block = Some(exit_block);
                
                // Return unit
                self.build_literal(&Literal::Int(0))
            }
            
            Expr::While(cond, body) => {
                let func = self.current_function.as_mut()
                    .ok_or_else(|| Error::new(ErrorKind::InvalidSyntax, "No current function"))?;
                
                let cond_block = func.add_named_block("while.cond".to_string());
                let body_block = func.add_named_block("while.body".to_string());
                let exit_block = func.add_named_block("while.exit".to_string());
                
                // Jump to condition
                let jump_inst = Instruction::Jump { target: cond_block };
                self.emit_instruction(jump_inst)?;
                
                // Build condition
                self.current_block = Some(cond_block);
                let cond_val = self.build_expr(cond)?;
                let branch_inst = Instruction::Branch {
                    cond: cond_val,
                    then_block: body_block,
                    else_block: exit_block,
                };
                self.emit_instruction(branch_inst)?;
                
                // Build body
                self.current_block = Some(body_block);
                self.build_expr(body)?;
                let jump_inst = Instruction::Jump { target: cond_block };
                self.emit_instruction(jump_inst)?;
                
                // Exit block
                self.current_block = Some(exit_block);
                
                // Return unit
                self.build_literal(&Literal::Int(0))
            }
            
            _ => {
                // Placeholder for other expressions
                self.build_literal(&Literal::Int(0))
            }
        }
    }
    
    fn build_literal(&mut self, lit: &Literal) -> Result<ValueId> {
        let constant = match lit {
            Literal::Int(n) => Constant::Int(*n as i128, IrType::I32),
            Literal::Float(f) => Constant::Float(*f, IrType::F64),
            Literal::Bool(b) => Constant::Bool(*b),
            Literal::String(_) => Constant::Int(0, IrType::I32), // Simplified
            Literal::Char(_) => Constant::Int(0, IrType::I32), // Simplified
        };
        
        let func = self.current_function.as_mut()
            .ok_or_else(|| Error::new(ErrorKind::InvalidSyntax, "No current function"))?;
        
        Ok(func.add_value(Value::Constant(constant)))
    }
    
    fn emit_instruction(&mut self, inst: Instruction) -> Result<ValueId> {
        let func = self.current_function.as_mut()
            .ok_or_else(|| Error::new(ErrorKind::InvalidSyntax, "No current function"))?;
        
        let value_id = func.add_value(Value::Instruction(inst.clone()));
        
        let block_id = self.current_block
            .ok_or_else(|| Error::new(ErrorKind::InvalidSyntax, "No current block"))?;
        
        if let Some(block) = func.get_block_mut(block_id) {
            block.add_instruction(value_id, inst);
        }
        
        Ok(value_id)
    }
    
    fn convert_type(&self, ty: &Type) -> IrType {
        match ty {
            Type::Named(name) => match name.as_str() {
                "i8" => IrType::I8,
                "i16" => IrType::I16,
                "i32" => IrType::I32,
                "i64" => IrType::I64,
                "u8" => IrType::U8,
                "u16" => IrType::U16,
                "u32" => IrType::U32,
                "u64" => IrType::U64,
                "f32" => IrType::F32,
                "f64" => IrType::F64,
                "bool" => IrType::Bool,
                "()" => IrType::Void,
                _ => IrType::I32, // Default
            },
            Type::Reference(inner) => {
                IrType::Pointer(Box::new(self.convert_type(inner)))
            }
            Type::MutableReference(inner) => {
                IrType::Pointer(Box::new(self.convert_type(inner)))
            }
            Type::Array(elem, size) => {
                IrType::Array(Box::new(self.convert_type(elem)), *size)
            }
            _ => IrType::I32, // Simplified
        }
    }
    
    fn convert_binop(&self, op: &BinOp) -> BinaryOp {
        match op {
            BinOp::Add => BinaryOp::Add,
            BinOp::Sub => BinaryOp::Sub,
            BinOp::Mul => BinaryOp::Mul,
            BinOp::Div => BinaryOp::Div,
            BinOp::Mod => BinaryOp::Rem,
            BinOp::Eq => BinaryOp::Eq,
            BinOp::Ne => BinaryOp::Ne,
            BinOp::Lt => BinaryOp::Lt,
            BinOp::Le => BinaryOp::Le,
            BinOp::Gt => BinaryOp::Gt,
            BinOp::Ge => BinaryOp::Ge,
            BinOp::And => BinaryOp::And,
            BinOp::Or => BinaryOp::Or,
        }
    }
    
    fn convert_unop(&self, op: &UnOp) -> UnaryOp {
        match op {
            UnOp::Neg => UnaryOp::Neg,
            UnOp::Not => UnaryOp::Not,
            _ => UnaryOp::Neg, // Simplified
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn build_ir(source: &str) -> Result<Module> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let builder = IrBuilder::new("test".to_string());
        builder.build_program(&program)
    }

    #[test]
    fn test_simple_function() {
        let source = "fn main() { }";
        let module = build_ir(source).unwrap();
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "main");
    }

    #[test]
    fn test_function_with_return() {
        let source = "fn add(a: i32, b: i32) -> i32 { return a; }";
        let module = build_ir(source).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn test_arithmetic() {
        let source = "fn main() { let x = 1 + 2; }";
        let module = build_ir(source).unwrap();
        assert_eq!(module.functions.len(), 1);
    }
}
