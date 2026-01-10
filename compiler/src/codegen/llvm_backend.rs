//! LLVM backend for code generation

use crate::ir::ssa::{Module, Function as IrFunction, BasicBlock, ValueId, Value, Constant};
use crate::ir::instructions::{Instruction, BinaryOp, UnaryOp};
use crate::ir::types::IrType;
use crate::error::{Result, CompilerError};
use std::collections::HashMap;

/// LLVM code generator
pub struct LLVMCodegen {
    module_name: String,
    target_triple: String,
    value_map: HashMap<ValueId, String>,
    next_temp: usize,
}

impl LLVMCodegen {
    /// Create a new LLVM code generator
    pub fn new(module_name: String) -> Self {
        LLVMCodegen {
            module_name,
            target_triple: Self::get_host_triple(),
            value_map: HashMap::new(),
            next_temp: 0,
        }
    }
    
    /// Get the host target triple
    fn get_host_triple() -> String {
        // Simplified - in real implementation would use LLVM API
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_os = "linux")]
        return "x86_64-unknown-linux-gnu".to_string();
        
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_os = "macos")]
        return "x86_64-apple-darwin".to_string();
        
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_os = "windows")]
        return "x86_64-pc-windows-msvc".to_string();
        
        #[cfg(target_arch = "aarch64")]
        return "aarch64-unknown-linux-gnu".to_string();
        
        "unknown-unknown-unknown".to_string()
    }
    
    /// Generate a new temporary name
    fn new_temp(&mut self) -> String {
        let temp = format!("%{}", self.next_temp);
        self.next_temp += 1;
        temp
    }
    
    /// Convert IR type to LLVM type string
    fn type_to_llvm(&self, ty: &IrType) -> String {
        match ty {
            IrType::Void => "void".to_string(),
            IrType::I1 => "i1".to_string(),
            IrType::I8 => "i8".to_string(),
            IrType::I16 => "i16".to_string(),
            IrType::I32 => "i32".to_string(),
            IrType::I64 => "i64".to_string(),
            IrType::I128 => "i128".to_string(),
            IrType::U8 => "i8".to_string(),
            IrType::U16 => "i16".to_string(),
            IrType::U32 => "i32".to_string(),
            IrType::U64 => "i64".to_string(),
            IrType::U128 => "i128".to_string(),
            IrType::F32 => "float".to_string(),
            IrType::F64 => "double".to_string(),
            IrType::Ptr(inner) => format!("{}*", self.type_to_llvm(inner)),
            IrType::Array(inner, size) => format!("[{} x {}]", size, self.type_to_llvm(inner)),
            IrType::Struct(fields) => {
                let field_types: Vec<_> = fields.iter()
                    .map(|f| self.type_to_llvm(f))
                    .collect();
                format!("{{ {} }}", field_types.join(", "))
            }
            IrType::Function(params, ret) => {
                let param_types: Vec<_> = params.iter()
                    .map(|p| self.type_to_llvm(p))
                    .collect();
                format!("{} ({})", self.type_to_llvm(ret), param_types.join(", "))
            }
        }
    }
    
    /// Convert binary operation to LLVM instruction
    fn binary_op_to_llvm(&self, op: BinaryOp, ty: &IrType) -> &str {
        match (op, ty) {
            (BinaryOp::Add, IrType::F32 | IrType::F64) => "fadd",
            (BinaryOp::Add, _) => "add",
            (BinaryOp::Sub, IrType::F32 | IrType::F64) => "fsub",
            (BinaryOp::Sub, _) => "sub",
            (BinaryOp::Mul, IrType::F32 | IrType::F64) => "fmul",
            (BinaryOp::Mul, _) => "mul",
            (BinaryOp::Div, IrType::F32 | IrType::F64) => "fdiv",
            (BinaryOp::Div, _) => "sdiv",
            (BinaryOp::Rem, IrType::F32 | IrType::F64) => "frem",
            (BinaryOp::Rem, _) => "srem",
            (BinaryOp::And, _) => "and",
            (BinaryOp::Or, _) => "or",
            (BinaryOp::Xor, _) => "xor",
            (BinaryOp::Shl, _) => "shl",
            (BinaryOp::Shr, _) => "ashr",
            (BinaryOp::Eq, IrType::F32 | IrType::F64) => "fcmp oeq",
            (BinaryOp::Eq, _) => "icmp eq",
            (BinaryOp::Ne, IrType::F32 | IrType::F64) => "fcmp one",
            (BinaryOp::Ne, _) => "icmp ne",
            (BinaryOp::Lt, IrType::F32 | IrType::F64) => "fcmp olt",
            (BinaryOp::Lt, _) => "icmp slt",
            (BinaryOp::Le, IrType::F32 | IrType::F64) => "fcmp ole",
            (BinaryOp::Le, _) => "icmp sle",
            (BinaryOp::Gt, IrType::F32 | IrType::F64) => "fcmp ogt",
            (BinaryOp::Gt, _) => "icmp sgt",
            (BinaryOp::Ge, IrType::F32 | IrType::F64) => "fcmp oge",
            (BinaryOp::Ge, _) => "icmp sge",
        }
    }
    
    /// Convert constant to LLVM representation
    fn constant_to_llvm(&self, constant: &Constant) -> String {
        match constant {
            Constant::Int(val, ty) => format!("{} {}", self.type_to_llvm(ty), val),
            Constant::Float(val, ty) => format!("{} {}", self.type_to_llvm(ty), val),
            Constant::Bool(val) => format!("i1 {}", if *val { 1 } else { 0 }),
            Constant::Null(ty) => format!("{} null", self.type_to_llvm(ty)),
            Constant::Undef(ty) => format!("{} undef", self.type_to_llvm(ty)),
        }
    }
    
    /// Generate LLVM IR for a module
    pub fn generate(&mut self, module: &Module) -> Result<String> {
        let mut output = String::new();
        
        // Module header
        output.push_str(&format!("; ModuleID = '{}'\n", self.module_name));
        output.push_str(&format!("target triple = \"{}\"\n\n", self.target_triple));
        
        // Generate functions
        for function in &module.functions {
            output.push_str(&self.generate_function(function)?);
            output.push_str("\n");
        }
        
        Ok(output)
    }
    
    /// Generate LLVM IR for a function
    fn generate_function(&mut self, function: &IrFunction) -> Result<String> {
        let mut output = String::new();
        self.value_map.clear();
        self.next_temp = 0;
        
        // Function signature
        let ret_type = self.type_to_llvm(&function.return_type);
        let params: Vec<_> = function.params.iter()
            .enumerate()
            .map(|(i, ty)| {
                let param_name = format!("%arg{}", i);
                self.value_map.insert(ValueId(i), param_name.clone());
                format!("{} {}", self.type_to_llvm(ty), param_name)
            })
            .collect();
        
        output.push_str(&format!("define {} @{}({}) {{\n", 
            ret_type, function.name, params.join(", ")));
        
        // Generate basic blocks
        for (i, block) in function.blocks.iter().enumerate() {
            if i > 0 {
                output.push_str(&format!("bb{}:\n", i));
            } else {
                output.push_str("entry:\n");
            }
            
            output.push_str(&self.generate_block(block, function)?);
        }
        
        output.push_str("}\n");
        Ok(output)
    }
    
    /// Generate LLVM IR for a basic block
    fn generate_block(&mut self, block: &BasicBlock, function: &IrFunction) -> Result<String> {
        let mut output = String::new();
        
        for (value_id, inst) in &block.instructions {
            let inst_str = self.generate_instruction(*value_id, inst, function)?;
            if !inst_str.is_empty() {
                output.push_str("  ");
                output.push_str(&inst_str);
                output.push_str("\n");
            }
        }
        
        Ok(output)
    }
    
    /// Generate LLVM IR for an instruction
    fn generate_instruction(&mut self, value_id: ValueId, inst: &Instruction, function: &IrFunction) -> Result<String> {
        match inst {
            Instruction::Binary { op, lhs, rhs, ty } => {
                let temp = self.new_temp();
                self.value_map.insert(value_id, temp.clone());
                
                let lhs_val = self.get_value_name(*lhs, function);
                let rhs_val = self.get_value_name(*rhs, function);
                let llvm_op = self.binary_op_to_llvm(*op, ty);
                let ty_str = self.type_to_llvm(ty);
                
                Ok(format!("{} = {} {} {}, {}", temp, llvm_op, ty_str, lhs_val, rhs_val))
            }
            Instruction::Unary { op, operand, ty } => {
                let temp = self.new_temp();
                self.value_map.insert(value_id, temp.clone());
                
                let operand_val = self.get_value_name(*operand, function);
                let ty_str = self.type_to_llvm(ty);
                
                let llvm_inst = match op {
                    UnaryOp::Neg => format!("{} = sub {} 0, {}", temp, ty_str, operand_val),
                    UnaryOp::Not => format!("{} = xor {} {}, -1", temp, ty_str, operand_val),
                };
                
                Ok(llvm_inst)
            }
            Instruction::Alloca { ty } => {
                let temp = self.new_temp();
                self.value_map.insert(value_id, temp.clone());
                let ty_str = self.type_to_llvm(ty);
                Ok(format!("{} = alloca {}", temp, ty_str))
            }
            Instruction::Load { ptr, ty } => {
                let temp = self.new_temp();
                self.value_map.insert(value_id, temp.clone());
                let ptr_val = self.get_value_name(*ptr, function);
                let ty_str = self.type_to_llvm(ty);
                Ok(format!("{} = load {}, {}* {}", temp, ty_str, ty_str, ptr_val))
            }
            Instruction::Store { ptr, value } => {
                let ptr_val = self.get_value_name(*ptr, function);
                let value_val = self.get_value_name(*value, function);
                
                // Get the type of the value being stored
                if let Some(val) = function.get_value(*value) {
                    let ty = self.get_value_type(val, function);
                    let ty_str = self.type_to_llvm(&ty);
                    Ok(format!("store {} {}, {}* {}", ty_str, value_val, ty_str, ptr_val))
                } else {
                    Ok(format!("store i32 {}, i32* {}", value_val, ptr_val))
                }
            }
            Instruction::Call { func, args, ty } => {
                let temp = self.new_temp();
                self.value_map.insert(value_id, temp.clone());
                
                let func_val = self.get_value_name(*func, function);
                let arg_vals: Vec<_> = args.iter()
                    .map(|arg| {
                        let val = self.get_value_name(*arg, function);
                        if let Some(v) = function.get_value(*arg) {
                            let ty = self.get_value_type(v, function);
                            format!("{} {}", self.type_to_llvm(&ty), val)
                        } else {
                            format!("i32 {}", val)
                        }
                    })
                    .collect();
                
                let ret_ty = self.type_to_llvm(ty);
                Ok(format!("{} = call {} {}({})", temp, ret_ty, func_val, arg_vals.join(", ")))
            }
            Instruction::Return { value } => {
                if let Some(val) = value {
                    let val_name = self.get_value_name(*val, function);
                    if let Some(v) = function.get_value(*val) {
                        let ty = self.get_value_type(v, function);
                        let ty_str = self.type_to_llvm(&ty);
                        Ok(format!("ret {} {}", ty_str, val_name))
                    } else {
                        Ok(format!("ret i32 {}", val_name))
                    }
                } else {
                    Ok("ret void".to_string())
                }
            }
            Instruction::Branch { cond, true_bb, false_bb } => {
                let cond_val = self.get_value_name(*cond, function);
                Ok(format!("br i1 {}, label %bb{}, label %bb{}", cond_val, true_bb.0, false_bb.0))
            }
            Instruction::Jump { target } => {
                Ok(format!("br label %bb{}", target.0))
            }
            Instruction::Phi { incoming, ty } => {
                let temp = self.new_temp();
                self.value_map.insert(value_id, temp.clone());
                
                let ty_str = self.type_to_llvm(ty);
                let incoming_strs: Vec<_> = incoming.iter()
                    .map(|(val, bb)| {
                        let val_name = self.get_value_name(*val, function);
                        format!("[ {}, %bb{} ]", val_name, bb.0)
                    })
                    .collect();
                
                Ok(format!("{} = phi {} {}", temp, ty_str, incoming_strs.join(", ")))
            }
            _ => Ok(String::new()),
        }
    }
    
    /// Get the LLVM name for a value
    fn get_value_name(&self, value_id: ValueId, function: &IrFunction) -> String {
        if let Some(name) = self.value_map.get(&value_id) {
            return name.clone();
        }
        
        if let Some(value) = function.get_value(value_id) {
            match value {
                Value::Constant(c) => return self.constant_to_llvm(c),
                Value::Parameter(idx, _) => return format!("%arg{}", idx),
                _ => {}
            }
        }
        
        format!("%{}", value_id.0)
    }
    
    /// Get the type of a value
    fn get_value_type(&self, value: &Value, _function: &IrFunction) -> IrType {
        match value {
            Value::Constant(Constant::Int(_, ty)) => ty.clone(),
            Value::Constant(Constant::Float(_, ty)) => ty.clone(),
            Value::Constant(Constant::Bool(_)) => IrType::I1,
            Value::Parameter(_, ty) => ty.clone(),
            _ => IrType::I32, // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_creation() {
        let codegen = LLVMCodegen::new("test".to_string());
        assert_eq!(codegen.module_name, "test");
    }

    #[test]
    fn test_type_conversion() {
        let codegen = LLVMCodegen::new("test".to_string());
        assert_eq!(codegen.type_to_llvm(&IrType::I32), "i32");
        assert_eq!(codegen.type_to_llvm(&IrType::F64), "double");
        assert_eq!(codegen.type_to_llvm(&IrType::Void), "void");
    }

    #[test]
    fn test_binary_op_conversion() {
        let codegen = LLVMCodegen::new("test".to_string());
        assert_eq!(codegen.binary_op_to_llvm(BinaryOp::Add, &IrType::I32), "add");
        assert_eq!(codegen.binary_op_to_llvm(BinaryOp::Add, &IrType::F32), "fadd");
        assert_eq!(codegen.binary_op_to_llvm(BinaryOp::Eq, &IrType::I32), "icmp eq");
    }

    #[test]
    fn test_constant_conversion() {
        let codegen = LLVMCodegen::new("test".to_string());
        let const_int = Constant::Int(42, IrType::I32);
        assert_eq!(codegen.constant_to_llvm(&const_int), "i32 42");
        
        let const_bool = Constant::Bool(true);
        assert_eq!(codegen.constant_to_llvm(&const_bool), "i1 1");
    }

    #[test]
    fn test_generate_empty_module() {
        let mut codegen = LLVMCodegen::new("test".to_string());
        let module = Module::new("test".to_string());
        
        let result = codegen.generate(&module);
        assert!(result.is_ok());
        
        let llvm_ir = result.unwrap();
        assert!(llvm_ir.contains("ModuleID"));
        assert!(llvm_ir.contains("target triple"));
    }
}
