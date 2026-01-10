//! SSA form IR representation

use super::types::IrType;
use super::instructions::Instruction;
use std::fmt;

/// Value identifier in SSA form
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

/// Value in SSA form
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Instruction result
    Instruction(Instruction),
    /// Constant value
    Constant(Constant),
    /// Function parameter
    Parameter(usize, IrType),
    /// Global value
    Global(String, IrType),
}

/// Constant values
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Int(i128, IrType),
    Float(f64, IrType),
    Bool(bool),
    Null,
    Undef,
}

impl Constant {
    pub fn ty(&self) -> IrType {
        match self {
            Constant::Int(_, ty) => ty.clone(),
            Constant::Float(_, ty) => ty.clone(),
            Constant::Bool(_) => IrType::Bool,
            Constant::Null => IrType::Pointer(Box::new(IrType::Void)),
            Constant::Undef => IrType::Void,
        }
    }
}

/// Basic block in SSA form
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub name: Option<String>,
    pub instructions: Vec<(ValueId, Instruction)>,
    pub predecessors: Vec<usize>,
    pub successors: Vec<usize>,
}

impl BasicBlock {
    pub fn new(id: usize) -> Self {
        BasicBlock {
            id,
            name: None,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }
    
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    pub fn add_instruction(&mut self, value_id: ValueId, inst: Instruction) {
        self.instructions.push((value_id, inst));
    }
    
    pub fn add_predecessor(&mut self, block_id: usize) {
        if !self.predecessors.contains(&block_id) {
            self.predecessors.push(block_id);
        }
    }
    
    pub fn add_successor(&mut self, block_id: usize) {
        if !self.successors.contains(&block_id) {
            self.successors.push(block_id);
        }
    }
    
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instructions.last().map(|(_, inst)| inst)
    }
    
    pub fn is_terminated(&self) -> bool {
        self.terminator().map_or(false, |inst| inst.is_terminator())
    }
}

/// Function in SSA form
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<IrType>,
    pub return_type: IrType,
    pub blocks: Vec<BasicBlock>,
    pub values: Vec<Value>,
    pub next_value_id: usize,
    pub next_block_id: usize,
}

impl Function {
    pub fn new(name: String, params: Vec<IrType>, return_type: IrType) -> Self {
        let mut func = Function {
            name,
            params: params.clone(),
            return_type,
            blocks: Vec::new(),
            values: Vec::new(),
            next_value_id: 0,
            next_block_id: 0,
        };
        
        // Add parameters as values
        for (i, param_ty) in params.iter().enumerate() {
            func.add_value(Value::Parameter(i, param_ty.clone()));
        }
        
        func
    }
    
    pub fn add_value(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        self.values.push(value);
        id
    }
    
    pub fn add_block(&mut self) -> usize {
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.blocks.push(BasicBlock::new(id));
        id
    }
    
    pub fn add_named_block(&mut self, name: String) -> usize {
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.blocks.push(BasicBlock::new(id).with_name(name));
        id
    }
    
    pub fn get_block(&self, id: usize) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }
    
    pub fn get_block_mut(&mut self, id: usize) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }
    
    pub fn get_value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id.0)
    }
    
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.first()
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn {}(", self.name)?;
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", param)?;
        }
        writeln!(f, ") -> {} {{", self.return_type)?;
        
        for block in &self.blocks {
            if let Some(name) = &block.name {
                writeln!(f, "{}:", name)?;
            } else {
                writeln!(f, "bb{}:", block.id)?;
            }
            
            for (value_id, inst) in &block.instructions {
                if inst.result_type().is_some() {
                    writeln!(f, "  v{} = {}", value_id.0, inst)?;
                } else {
                    writeln!(f, "  {}", inst)?;
                }
            }
            writeln!(f)?;
        }
        
        writeln!(f, "}}")
    }
}

/// IR module containing multiple functions
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub functions: Vec<Function>,
    pub globals: Vec<(String, IrType, Option<Constant>)>,
}

impl Module {
    pub fn new(name: String) -> Self {
        Module {
            name,
            functions: Vec::new(),
            globals: Vec::new(),
        }
    }
    
    pub fn add_function(&mut self, func: Function) {
        self.functions.push(func);
    }
    
    pub fn add_global(&mut self, name: String, ty: IrType, init: Option<Constant>) {
        self.globals.push((name, ty, init));
    }
    
    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "module {} {{", self.name)?;
        writeln!(f)?;
        
        for (name, ty, init) in &self.globals {
            write!(f, "global {} : {}", name, ty)?;
            if let Some(constant) = init {
                write!(f, " = {:?}", constant)?;
            }
            writeln!(f)?;
        }
        
        if !self.globals.is_empty() {
            writeln!(f)?;
        }
        
        for func in &self.functions {
            writeln!(f, "{}", func)?;
        }
        
        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instructions::BinaryOp;

    #[test]
    fn test_function_creation() {
        let func = Function::new(
            "test".to_string(),
            vec![IrType::I32, IrType::I32],
            IrType::I32,
        );
        
        assert_eq!(func.name, "test");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.return_type, IrType::I32);
        assert_eq!(func.values.len(), 2); // Parameters
    }

    #[test]
    fn test_basic_block() {
        let mut block = BasicBlock::new(0);
        
        let inst = Instruction::Binary {
            op: BinaryOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: IrType::I32,
        };
        
        block.add_instruction(ValueId(2), inst);
        assert_eq!(block.instructions.len(), 1);
    }

    #[test]
    fn test_block_predecessors_successors() {
        let mut block = BasicBlock::new(0);
        
        block.add_predecessor(1);
        block.add_successor(2);
        
        assert_eq!(block.predecessors, vec![1]);
        assert_eq!(block.successors, vec![2]);
    }

    #[test]
    fn test_module() {
        let mut module = Module::new("test_module".to_string());
        
        let func = Function::new(
            "main".to_string(),
            vec![],
            IrType::Void,
        );
        
        module.add_function(func);
        assert_eq!(module.functions.len(), 1);
        assert!(module.get_function("main").is_some());
    }

    #[test]
    fn test_constant_type() {
        let c = Constant::Int(42, IrType::I32);
        assert_eq!(c.ty(), IrType::I32);
        
        let b = Constant::Bool(true);
        assert_eq!(b.ty(), IrType::Bool);
    }
}
