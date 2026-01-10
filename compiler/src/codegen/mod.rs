//! Code generation using LLVM backend

pub mod llvm_backend;
pub mod target;
pub mod linker;

pub use llvm_backend::LLVMCodegen;
pub use target::TargetMachine;
pub use linker::Linker;
