//! Micro-benchmarks for individual compiler components

use crate::{BenchmarkSuite, BenchmarkResult};
use anyhow::Result;

pub fn run_all(suite: &mut BenchmarkSuite) -> Result<()> {
    bench_lexer(suite)?;
    bench_parser(suite)?;
    bench_type_checker(suite)?;
    Ok(())
}

pub fn bench_lexer(suite: &mut BenchmarkSuite) -> Result<()> {
    let small_code = "let x = 5";
    let medium_code = generate_code(100);
    let large_code = generate_code(10000);

    suite.run("lexer_small", 1000, || {
        let _ = simulate_lex(small_code);
    });

    suite.run("lexer_medium", 100, || {
        let _ = simulate_lex(&medium_code);
    });

    suite.run("lexer_large", 10, || {
        let _ = simulate_lex(&large_code);
    });

    Ok(())
}

pub fn bench_parser(suite: &mut BenchmarkSuite) -> Result<()> {
    let small_tokens = generate_tokens(10);
    let medium_tokens = generate_tokens(100);
    let large_tokens = generate_tokens(1000);

    suite.run("parser_small", 1000, || {
        let _ = simulate_parse(&small_tokens);
    });

    suite.run("parser_medium", 100, || {
        let _ = simulate_parse(&medium_tokens);
    });

    suite.run("parser_large", 10, || {
        let _ = simulate_parse(&large_tokens);
    });

    Ok(())
}

pub fn bench_type_checker(suite: &mut BenchmarkSuite) -> Result<()> {
    suite.run("type_check_simple", 1000, || {
        let _ = simulate_type_check(10);
    });

    suite.run("type_check_generic", 100, || {
        let _ = simulate_type_check_with_generics(10);
    });

    suite.run("type_check_inference", 100, || {
        let _ = simulate_type_inference(50);
    });

    Ok(())
}

fn generate_code(lines: usize) -> String {
    (0..lines)
        .map(|i| format!("let x{} = {} + {}", i, i, i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_tokens(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("TOKEN_{}", i % 10))
        .collect()
}

fn simulate_lex(code: &str) -> usize {
    let mut count = 0;
    for c in code.chars() {
        if c.is_whitespace() || c.is_alphanumeric() {
            count += 1;
        }
    }
    count
}

fn simulate_parse(tokens: &[String]) -> usize {
    tokens.len()
}

fn simulate_type_check(depth: usize) -> bool {
    if depth == 0 {
        return true;
    }
    simulate_type_check(depth - 1)
}

fn simulate_type_check_with_generics(depth: usize) -> bool {
    if depth == 0 {
        return true;
    }
    simulate_type_check_with_generics(depth - 1)
}

fn simulate_type_inference(exprs: usize) -> usize {
    (0..exprs).map(|i| i * 2).sum()
}

pub struct LexerBench {
    pub name: String,
    pub source: String,
}

impl LexerBench {
    pub fn new(name: &str, source: &str) -> Self {
        LexerBench {
            name: name.to_string(),
            source: source.to_string(),
        }
    }

    pub fn run(&self) -> BenchmarkResult {
        let mut result = BenchmarkResult::new(&self.name);
        for _ in 0..100 {
            let start = std::time::Instant::now();
            simulate_lex(&self.source);
            result.add_iteration(start.elapsed());
        }
        result
    }
}

pub struct ParserBench {
    pub name: String,
    pub tokens: Vec<String>,
}

impl ParserBench {
    pub fn new(name: &str, tokens: Vec<String>) -> Self {
        ParserBench {
            name: name.to_string(),
            tokens,
        }
    }

    pub fn run(&self) -> BenchmarkResult {
        let mut result = BenchmarkResult::new(&self.name);
        for _ in 0..100 {
            let start = std::time::Instant::now();
            simulate_parse(&self.tokens);
            result.add_iteration(start.elapsed());
        }
        result
    }
}

pub struct TypeCheckerBench {
    pub name: String,
    pub depth: usize,
}

impl TypeCheckerBench {
    pub fn new(name: &str, depth: usize) -> Self {
        TypeCheckerBench {
            name: name.to_string(),
            depth,
        }
    }

    pub fn run(&self) -> BenchmarkResult {
        let mut result = BenchmarkResult::new(&self.name);
        for _ in 0..100 {
            let start = std::time::Instant::now();
            simulate_type_check(self.depth);
            result.add_iteration(start.elapsed());
        }
        result
    }
}
