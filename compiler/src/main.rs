//! VeZ Compiler Driver
//!
//! Main entry point for the VeZ compiler command-line tool.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use vezc::prelude::*;

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

    /// Emit type (exe, lib, obj, llvm-ir, asm)
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

    // TODO: Create compiler configuration
    log::info!("Compilation configuration:");
    log::info!("  Optimization level: {}", cli.opt_level);
    log::info!("  Emit type: {:?}", cli.emit);
    log::info!("  Check only: {}", cli.check);

    // TODO: Initialize compiler
    // let mut compiler = Compiler::new(config);

    // TODO: Run compilation
    // match compiler.compile() {
    //     Ok(_) => {
    //         log::info!("Compilation successful!");
    //     }
    //     Err(e) => {
    //         eprintln!("Compilation failed: {}", e);
    //         std::process::exit(1);
    //     }
    // }

    println!("VeZ compiler initialized successfully!");
    println!("Note: Compiler implementation is in progress (Phase 1)");
    println!("Input files: {:?}", cli.input);
    println!("Output: {:?}", cli.output.unwrap_or_else(|| PathBuf::from("a.out")));
}
