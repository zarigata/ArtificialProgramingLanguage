//! VeZ Compiler Driver
//!
//! Main entry point for the VeZ compiler command-line tool.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use vezc::driver::{Compiler, CompilerConfig, OutputType, OptLevel};
use vezc::error::Result;

#[derive(Parser)]
#[command(name = "vezc")]
#[command(version = vezc::VERSION)]
#[command(about = "The VeZ programming language compiler", long_about = None)]
struct Cli {
    /// Input source files (.zari)
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output file path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optimization level (0-3, s, z)
    #[arg(short = 'O', long, default_value = "2")]
    opt_level: String,

    /// Emit type (exe, lib, obj, llvm-ir, asm, check)
    #[arg(long, default_value = "exe")]
    emit: EmitType,

    /// Target triple (e.g., x86_64-unknown-linux-gnu)
    #[arg(long)]
    target: Option<String>,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Check only (no code generation)
    #[arg(long)]
    check: bool,

    /// Dump AST (for debugging)
    #[arg(long)]
    dump_ast: bool,

    /// Dump IR (for debugging)
    #[arg(long)]
    dump_ir: bool,

    /// Number of parallel jobs
    #[arg(short, long)]
    jobs: Option<usize>,
}

#[derive(Debug, Clone, ValueEnum)]
enum EmitType {
    /// Executable binary
    Exe,
    /// Static library
    Lib,
    /// Object file
    Obj,
    /// LLVM IR
    LlvmIr,
    /// Assembly
    Asm,
    /// Check only (no output)
    Check,
}

impl From<EmitType> for OutputType {
    fn from(emit: EmitType) -> Self {
        match emit {
            EmitType::Exe => OutputType::Executable,
            EmitType::Lib => OutputType::Object, // TODO: proper library support
            EmitType::Obj => OutputType::Object,
            EmitType::LlvmIr => OutputType::LlvmIr,
            EmitType::Asm => OutputType::Assembly,
            EmitType::Check => OutputType::Check,
        }
    }
}

fn main() {
    // Parse command-line arguments
    let cli = Cli::parse();

    // Initialize logger
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    log::info!("VeZ Compiler v{}", vezc::VERSION);
    log::debug!("Input files: {:?}", cli.input);

    // Validate input files
    for input in &cli.input {
        if !input.exists() {
            eprintln!("Error: Input file does not exist: {}", input.display());
            std::process::exit(1);
        }

        if input.extension().and_then(|s| s.to_str()) != Some(vezc::FILE_EXTENSION) {
            eprintln!(
                "Warning: Input file does not have .{} extension: {}",
                vezc::FILE_EXTENSION,
                input.display()
            );
        }
    }

    // Parse optimization level
    let opt_level = OptLevel::from_str(&cli.opt_level).unwrap_or_else(|| {
        eprintln!("Warning: Invalid optimization level '{}', using O2", cli.opt_level);
        OptLevel::O2
    });

    // Determine output type
    let output_type = if cli.check {
        OutputType::Check
    } else {
        cli.emit.into()
    };

    // Create compiler configuration
    let config = CompilerConfig {
        input_files: cli.input.clone(),
        output_path: cli.output,
        output_type,
        opt_level,
        target_triple: cli.target,
        dump_ast: cli.dump_ast,
        dump_ir: cli.dump_ir,
        verbose: cli.verbose,
    };

    log::info!("Compilation configuration:");
    log::info!("  Optimization level: {:?}", config.opt_level);
    log::info!("  Output type: {:?}", config.output_type);

    // Run compilation
    match run_compilation(config) {
        Ok(()) => {
            log::info!("Compilation successful!");
            println!("Compilation successful!");
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_compilation(config: CompilerConfig) -> Result<()> {
    let mut compiler = Compiler::new(config);
    let results = compiler.compile()?;

    for result in results {
        println!("Compiled: {}", result.source_path.display());

        if let Some(output_file) = result.output_file {
            println!("  Output: {}", output_file.display());
        }

        if let Some(ref llvm_ir) = result.llvm_ir {
            log::debug!("Generated {} bytes of LLVM IR", llvm_ir.len());
        }
    }

    Ok(())
}
