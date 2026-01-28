//! Intermediate representation

pub mod ssa;
pub mod builder;
pub mod types;
pub mod instructions;

pub use ssa::{Function as IrFunction, BasicBlock, Value, ValueId, Module};
pub use builder::IrBuilder;
pub use types::IrType;
pub use instructions::Instruction;
