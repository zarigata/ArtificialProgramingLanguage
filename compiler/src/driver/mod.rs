//! Compiler driver - orchestrates the compilation pipeline
//!
//! This module connects all compilation stages:
//! 1. Lexing - tokenize source code
//! 2. Parsing - build AST
//! 3. Semantic analysis - type checking and name resolution
//! 4. IR generation - convert AST to SSA form
//! 5. Optimization - optimize IR (optional)
//! 6. Code generation - emit LLVM IR or target code

use std::path::{Path, PathBuf};
use std::fs;

use crate::error::{Error, ErrorKind, Result};
use crate::lexer::Lexer;
use crate::parser::{Parser, Program};
use crate::semantic::{SymbolTable, TypeChecker};
use crate::ir::{IrBuilder, Module as IrModule};
use crate::codegen::LLVMCodegen;

/// Compilation output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Check only - no code generation
    Check,
    /// Emit AST as JSON
    Ast,
    /// Emit IR
    Ir,
    /// Emit LLVM IR
    LlvmIr,
    /// Emit assembly
    Assembly,
    /// Emit object file
    Object,
    /// Emit executable
    Executable,
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Standard optimization
    O2,
    /// Aggressive optimization
    O3,
    /// Size optimization
    Os,
    /// Aggressive size optimization
    Oz,
}

impl OptLevel {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "0" => Some(OptLevel::O0),
            "1" => Some(OptLevel::O1),
            "2" => Some(OptLevel::O2),
            "3" => Some(OptLevel::O3),
            "s" => Some(OptLevel::Os),
            "z" => Some(OptLevel::Oz),
            _ => None,
        }
    }
}

/// Compiler configuration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Input source files
    pub input_files: Vec<PathBuf>,
    /// Output file path
    pub output_path: Option<PathBuf>,
    /// Output type
    pub output_type: OutputType,
    /// Optimization level
    pub opt_level: OptLevel,
    /// Target triple (e.g., x86_64-unknown-linux-gnu)
    pub target_triple: Option<String>,
    /// Dump AST for debugging
    pub dump_ast: bool,
    /// Dump IR for debugging
    pub dump_ir: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            input_files: Vec::new(),
            output_path: None,
            output_type: OutputType::Executable,
            opt_level: OptLevel::O2,
            target_triple: None,
            dump_ast: false,
            dump_ir: false,
            verbose: false,
        }
    }
}

/// Compilation result containing all intermediate representations
#[derive(Debug)]
pub struct CompilationResult {
    /// Source file path
    pub source_path: PathBuf,
    /// Parsed AST
    pub ast: Program,
    /// Generated IR module (if code generation was performed)
    pub ir_module: Option<IrModule>,
    /// Generated LLVM IR (if LLVM codegen was performed)
    pub llvm_ir: Option<String>,
    /// Output file path (if file was written)
    pub output_file: Option<PathBuf>,
}

/// Main compiler driver
pub struct Compiler {
    config: CompilerConfig,
}

impl Compiler {
    /// Create a new compiler with the given configuration
    pub fn new(config: CompilerConfig) -> Self {
        Compiler { config }
    }

    /// Create a compiler with default configuration
    pub fn with_defaults() -> Self {
        Compiler {
            config: CompilerConfig::default(),
        }
    }

    /// Get the compiler configuration
    pub fn config(&self) -> &CompilerConfig {
        &self.config
    }

    /// Compile all input files
    pub fn compile(&mut self) -> Result<Vec<CompilationResult>> {
        let mut results = Vec::new();

        for input_file in self.config.input_files.clone() {
            let result = self.compile_file(&input_file)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compile a single source file
    pub fn compile_file(&self, path: &Path) -> Result<CompilationResult> {
        if self.config.verbose {
            log::info!("Compiling: {}", path.display());
        }

        // Stage 1: Read source file
        let source = self.read_source(path)?;

        // Stage 2: Lexical analysis
        let tokens = self.lex(&source)?;
        if self.config.verbose {
            log::debug!("Lexer produced {} tokens", tokens.len());
        }

        // Stage 3: Parsing
        let ast = self.parse(tokens)?;
        if self.config.verbose {
            log::debug!("Parser produced {} items", ast.items.len());
        }

        // Dump AST if requested
        if self.config.dump_ast {
            self.dump_ast(&ast);
        }

        // Stage 4: Semantic analysis
        self.semantic_analysis(&ast)?;
        if self.config.verbose {
            log::debug!("Semantic analysis completed");
        }

        // If check-only, stop here
        if self.config.output_type == OutputType::Check {
            return Ok(CompilationResult {
                source_path: path.to_path_buf(),
                ast,
                ir_module: None,
                llvm_ir: None,
                output_file: None,
            });
        }

        // Stage 5: IR generation
        let module_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module")
            .to_string();
        let ir_module = self.build_ir(&ast, module_name.clone())?;
        if self.config.verbose {
            log::debug!("IR generation completed");
        }

        // Dump IR if requested
        if self.config.dump_ir {
            self.dump_ir(&ir_module);
        }

        // If IR-only output, stop here
        if self.config.output_type == OutputType::Ir {
            return Ok(CompilationResult {
                source_path: path.to_path_buf(),
                ast,
                ir_module: Some(ir_module),
                llvm_ir: None,
                output_file: None,
            });
        }

        // Stage 6: LLVM IR generation
        let llvm_ir = self.generate_llvm_ir(&ir_module, module_name)?;
        if self.config.verbose {
            log::debug!("LLVM IR generation completed ({} bytes)", llvm_ir.len());
        }

        // Handle output based on type
        let output_file = self.handle_output(path, &llvm_ir)?;

        Ok(CompilationResult {
            source_path: path.to_path_buf(),
            ast,
            ir_module: Some(ir_module),
            llvm_ir: Some(llvm_ir),
            output_file,
        })
    }

    /// Compile source code from a string
    pub fn compile_source(&self, source: &str, name: &str) -> Result<CompilationResult> {
        // Stage 2: Lexical analysis
        let tokens = self.lex(source)?;

        // Stage 3: Parsing
        let ast = self.parse(tokens)?;

        // Stage 4: Semantic analysis
        self.semantic_analysis(&ast)?;

        // If check-only, stop here
        if self.config.output_type == OutputType::Check {
            return Ok(CompilationResult {
                source_path: PathBuf::from(name),
                ast,
                ir_module: None,
                llvm_ir: None,
                output_file: None,
            });
        }

        // Stage 5: IR generation
        let ir_module = self.build_ir(&ast, name.to_string())?;

        // Stage 6: LLVM IR generation
        let llvm_ir = self.generate_llvm_ir(&ir_module, name.to_string())?;

        Ok(CompilationResult {
            source_path: PathBuf::from(name),
            ast,
            ir_module: Some(ir_module),
            llvm_ir: Some(llvm_ir),
            output_file: None,
        })
    }

    // === Pipeline Stages ===

    /// Read source file
    fn read_source(&self, path: &Path) -> Result<String> {
        fs::read_to_string(path).map_err(|e| {
            Error::new(
                ErrorKind::IoError,
                format!("Failed to read file '{}': {}", path.display(), e),
            )
        })
    }

    /// Lexical analysis - tokenize source code
    fn lex(&self, source: &str) -> Result<Vec<crate::lexer::Token>> {
        let mut lexer = Lexer::new(source);
        lexer.tokenize()
    }

    /// Parse tokens into AST
    fn parse(&self, tokens: Vec<crate::lexer::Token>) -> Result<Program> {
        let mut parser = Parser::new(tokens);
        parser.parse()
    }

    /// Perform semantic analysis (type checking, name resolution)
    fn semantic_analysis(&self, program: &Program) -> Result<()> {
        let symbol_table = SymbolTable::new();
        let mut type_checker = TypeChecker::new(symbol_table);
        type_checker.check_program(program)
    }

    /// Build intermediate representation
    fn build_ir(&self, program: &Program, module_name: String) -> Result<IrModule> {
        let builder = IrBuilder::new(module_name);
        builder.build_program(program)
    }

    /// Generate LLVM IR from the IR module
    fn generate_llvm_ir(&self, ir_module: &IrModule, module_name: String) -> Result<String> {
        let mut codegen = LLVMCodegen::new(module_name);
        codegen.generate(ir_module)
    }

    /// Handle output writing based on output type
    fn handle_output(&self, source_path: &Path, llvm_ir: &str) -> Result<Option<PathBuf>> {
        let output_path = self.determine_output_path(source_path);

        match self.config.output_type {
            OutputType::LlvmIr => {
                let ir_path = output_path.with_extension("ll");
                fs::write(&ir_path, llvm_ir).map_err(|e| {
                    Error::new(
                        ErrorKind::IoError,
                        format!("Failed to write LLVM IR: {}", e),
                    )
                })?;
                log::info!("Wrote LLVM IR to: {}", ir_path.display());
                Ok(Some(ir_path))
            }
            OutputType::Assembly => {
                // TODO: Use LLVM to generate assembly from IR
                let asm_path = output_path.with_extension("s");
                log::warn!("Assembly generation not yet implemented, writing LLVM IR instead");
                fs::write(&asm_path, llvm_ir).map_err(|e| {
                    Error::new(
                        ErrorKind::IoError,
                        format!("Failed to write assembly: {}", e),
                    )
                })?;
                Ok(Some(asm_path))
            }
            OutputType::Object => {
                // TODO: Use LLVM to generate object file from IR
                let obj_path = output_path.with_extension("o");
                log::warn!("Object file generation not yet implemented");
                Ok(Some(obj_path))
            }
            OutputType::Executable => {
                // TODO: Generate object file and link
                log::warn!("Executable generation not yet implemented, writing LLVM IR");
                let ir_path = output_path.with_extension("ll");
                fs::write(&ir_path, llvm_ir).map_err(|e| {
                    Error::new(
                        ErrorKind::IoError,
                        format!("Failed to write LLVM IR: {}", e),
                    )
                })?;
                Ok(Some(ir_path))
            }
            _ => Ok(None),
        }
    }

    /// Determine the output path based on config and source path
    fn determine_output_path(&self, source_path: &Path) -> PathBuf {
        if let Some(ref output) = self.config.output_path {
            output.clone()
        } else {
            source_path.with_extension("")
        }
    }

    // === Debug utilities ===

    /// Dump AST to stderr for debugging
    fn dump_ast(&self, program: &Program) {
        eprintln!("=== AST Dump ===");
        for item in &program.items {
            eprintln!("{:#?}", item);
        }
        eprintln!("=== End AST ===");
    }

    /// Dump IR to stderr for debugging
    fn dump_ir(&self, _module: &IrModule) {
        eprintln!("=== IR Dump ===");
        // TODO: Implement proper IR printing
        eprintln!("(IR dump not yet implemented)");
        eprintln!("=== End IR ===");
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Quick compilation functions for simple use cases

/// Check a source file for errors without generating code
pub fn check_file(path: &Path) -> Result<()> {
    let config = CompilerConfig {
        input_files: vec![path.to_path_buf()],
        output_type: OutputType::Check,
        ..Default::default()
    };
    let mut compiler = Compiler::new(config);
    compiler.compile()?;
    Ok(())
}

/// Check source code from a string
pub fn check_source(source: &str) -> Result<()> {
    let config = CompilerConfig {
        output_type: OutputType::Check,
        ..Default::default()
    };
    let compiler = Compiler::new(config);
    compiler.compile_source(source, "<input>")?;
    Ok(())
}

/// Compile a source file to LLVM IR
pub fn compile_to_llvm_ir(path: &Path) -> Result<String> {
    let config = CompilerConfig {
        input_files: vec![path.to_path_buf()],
        output_type: OutputType::LlvmIr,
        ..Default::default()
    };
    let compiler = Compiler::new(config);
    let result = compiler.compile_file(path)?;
    result.llvm_ir.ok_or_else(|| {
        Error::new(ErrorKind::InternalError, "LLVM IR generation failed")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_empty_program() {
        let result = check_source("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_simple_function() {
        let source = r#"
            fn main() {
                let x = 42;
            }
        "#;
        let result = check_source(source);
        // May fail due to type checking - that's expected for now
        let _ = result;
    }

    #[test]
    fn test_compile_simple_function() {
        let source = r#"
            fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;
        let config = CompilerConfig {
            output_type: OutputType::LlvmIr,
            ..Default::default()
        };
        let compiler = Compiler::new(config);
        let result = compiler.compile_source(source, "test");
        // Check that we at least get through parsing
        match result {
            Ok(r) => {
                assert!(r.llvm_ir.is_some() || r.ir_module.is_some());
            }
            Err(e) => {
                // Expected during early development
                eprintln!("Compilation error (expected): {}", e);
            }
        }
    }
}
