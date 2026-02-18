//! VeZ Benchmark Suite
//!
//! Comprehensive benchmarks for compiler and runtime performance

use anyhow::Result;
use colored::Colorize;
use std::time::{Duration, Instant};

mod micro;
mod macro_bench;
mod comparison;

pub use micro::{LexerBench, ParserBench, TypeCheckerBench};
pub use macro_bench::{CompileTimeBench, RuntimeBench, MemoryBench};
pub use comparison::{compare_with_rust, compare_with_go, compare_with_c};

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u64,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub throughput: Option<f64>,
}

impl BenchmarkResult {
    pub fn new(name: &str) -> Self {
        BenchmarkResult {
            name: name.to_string(),
            iterations: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            throughput: None,
        }
    }

    pub fn add_iteration(&mut self, time: Duration) {
        self.iterations += 1;
        self.total_time += time;
        self.min_time = self.min_time.min(time);
        self.max_time = self.max_time.max(time);
        self.avg_time = self.total_time / self.iterations as u32;
    }

    pub fn set_throughput(&mut self, items_per_second: f64) {
        self.throughput = Some(items_per_second);
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} iterations, avg {:?}, min {:?}, max {:?}",
            self.name.bright_cyan(),
            self.iterations,
            format_duration(self.avg_time),
            format_duration(self.min_time),
            format_duration(self.max_time)
        )?;
        if let Some(t) = self.throughput {
            write!(f, " ({:.2} items/s)", t)?;
        }
        Ok(())
    }
}

fn format_duration(d: Duration) -> String {
    if d < Duration::from_micros(1) {
        format!("{:?}ns", d.as_nanos())
    } else if d < Duration::from_millis(1) {
        format!("{:.2}µs", d.as_secs_f64() * 1_000_000.0)
    } else if d < Duration::from_secs(1) {
        format!("{:.2}ms", d.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.2}s", d.as_secs_f64())
    }
}

pub struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        BenchmarkSuite {
            results: Vec::new(),
        }
    }

    pub fn run<F>(&mut self, name: &str, iterations: u64, mut bench_fn: F)
    where
        F: FnMut(),
    {
        println!("Running {}...", name.bright_yellow());
        
        let mut result = BenchmarkResult::new(name);
        
        for _ in 0..iterations {
            let start = Instant::now();
            bench_fn();
            let elapsed = start.elapsed();
            result.add_iteration(elapsed);
        }
        
        println!("  {}", result);
        self.results.push(result);
    }

    pub fn run_with_setup<F, S>(&mut self, name: &str, iterations: u64, mut setup: S, mut bench_fn: F)
    where
        S: FnMut() -> (),
        F: FnMut(&()),
    {
        println!("Running {}...", name.bright_yellow());
        
        let mut result = BenchmarkResult::new(name);
        
        for _ in 0..iterations {
            let data = setup();
            let start = Instant::now();
            bench_fn(&data);
            let elapsed = start.elapsed();
            result.add_iteration(elapsed);
        }
        
        println!("  {}", result);
        self.results.push(result);
    }

    pub fn report(&self) {
        println!("\n{}", "=== BENCHMARK REPORT ===".bright_green());
        println!("{:<30} {:>12} {:>12} {:>12}", "Benchmark", "Avg", "Min", "Max");
        println!("{}", "-".repeat(70));
        
        for result in &self.results {
            println!(
                "{:<30} {:>12} {:>12} {:>12}",
                result.name.bright_cyan(),
                format_duration(result.avg_time),
                format_duration(result.min_time),
                format_duration(result.max_time)
            );
        }
        
        println!("{}", "-".repeat(70));
        
        let total_time: Duration = self.results.iter().map(|r| r.total_time).sum();
        println!("Total benchmark time: {}", format_duration(total_time).bright_yellow());
    }

    pub fn export_json(&self) -> String {
        let results: Vec<String> = self.results.iter().map(|r| {
            format!(
                r#"{{"name":"{}","iterations":{},"avg_ns":{},"min_ns":{},"max_ns":{}}}"#,
                r.name,
                r.iterations,
                r.avg_time.as_nanos(),
                r.min_time.as_nanos(),
                r.max_time.as_nanos()
            )
        }).collect();
        
        format!("[{}]", results.join(","))
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

fn main() -> Result<()> {
    println!("{}", "╔═══════════════════════════════════════════════════════════╗".bright_cyan());
    println!("{}", "║           VeZ Benchmark Suite v1.0.0                      ║".bright_cyan());
    println!("{}", "╚═══════════════════════════════════════════════════════════╝".bright_cyan());
    println!();

    let mut suite = BenchmarkSuite::new();

    // Micro-benchmarks
    println!("\n{}", "--- MICRO BENCHMARKS ---".bright_green());
    micro::run_all(&mut suite)?;

    // Macro-benchmarks
    println!("\n{}", "--- MACRO BENCHMARKS ---".bright_green());
    macro_bench::run_all(&mut suite)?;

    // Comparison benchmarks
    println!("\n{}", "--- COMPARISON BENCHMARKS ---".bright_green());
    comparison::run_all(&mut suite)?;

    // Final report
    suite.report();

    // Export results
    let json = suite.export_json();
    std::fs::write("benchmark_results.json", json)?;
    println!("\nResults exported to {}", "benchmark_results.json".bright_yellow());

    Ok(())
}
