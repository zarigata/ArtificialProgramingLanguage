//! Macro-benchmarks for end-to-end performance

use crate::{BenchmarkSuite, BenchmarkResult};
use anyhow::Result;
use std::time::Duration;

pub fn run_all(suite: &mut BenchmarkSuite) -> Result<()> {
    bench_compile_time(suite)?;
    bench_runtime_performance(suite)?;
    bench_memory(suite)?;
    Ok(())
}

pub fn bench_compile_time(suite: &mut BenchmarkSuite) -> Result<()> {
    suite.run("compile_hello_world", 100, || {
        let _ = simulate_compile(1);
    });

    suite.run("compile_100_loc", 50, || {
        let _ = simulate_compile(100);
    });

    suite.run("compile_1000_loc", 10, || {
        let _ = simulate_compile(1000);
    });

    suite.run("compile_10000_loc", 3, || {
        let _ = simulate_compile(10000);
    });

    Ok(())
}

pub fn bench_runtime_performance(suite: &mut BenchmarkSuite) -> Result<()> {
    suite.run("runtime_fibonacci_30", 1000, || {
        let _ = simulate_fibonacci(30);
    });

    suite.run("runtime_fibonacci_35", 100, || {
        let _ = simulate_fibonacci(35);
    });

    suite.run("runtime_sort_1000", 100, || {
        let _ = simulate_sort(1000);
    });

    suite.run("runtime_sort_10000", 10, || {
        let _ = simulate_sort(10000);
    });

    suite.run("runtime_matrix_100x100", 50, || {
        let _ = simulate_matrix_multiply(100);
    });

    Ok(())
}

pub fn bench_memory(suite: &mut BenchmarkSuite) -> Result<()> {
    suite.run("memory_alloc_1000", 1000, || {
        let _ = simulate_allocations(1000);
    });

    suite.run("memory_alloc_10000", 100, || {
        let _ = simulate_allocations(10000);
    });

    suite.run("memory_vec_push_1000", 1000, || {
        let _ = simulate_vec_push(1000);
    });

    suite.run("memory_hashmap_1000", 100, || {
        let _ = simulate_hashmap_insert(1000);
    });

    Ok(())
}

fn simulate_compile(lines: usize) -> Duration {
    let start = std::time::Instant::now();
    let mut result = 0usize;
    for i in 0..lines {
        result = result.wrapping_add(i);
    }
    start.elapsed()
}

fn simulate_fibonacci(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

fn simulate_sort(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).rev().collect();
    v.sort();
    v
}

fn simulate_matrix_multiply(n: usize) -> Vec<Vec<f64>> {
    let a: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| (i * n + j) as f64).collect()).collect();
    let b: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| (i + j) as f64).collect()).collect();
    
    let mut c = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn simulate_allocations(count: usize) -> usize {
    let mut v: Vec<Box<usize>> = Vec::with_capacity(count);
    for i in 0..count {
        v.push(Box::new(i));
    }
    v.len()
}

fn simulate_vec_push(count: usize) -> usize {
    let mut v = Vec::new();
    for i in 0..count {
        v.push(i);
    }
    v.len()
}

fn simulate_hashmap_insert(count: usize) -> usize {
    use std::collections::HashMap;
    let mut m: HashMap<usize, usize> = HashMap::new();
    for i in 0..count {
        m.insert(i, i * 2);
    }
    m.len()
}

pub struct CompileTimeBench {
    pub lines_of_code: usize,
}

impl CompileTimeBench {
    pub fn new(loc: usize) -> Self {
        CompileTimeBench { lines_of_code: loc }
    }

    pub fn run(&self) -> BenchmarkResult {
        let name = format!("compile_{}_loc", self.lines_of_code);
        let mut result = BenchmarkResult::new(&name);
        
        for _ in 0..10 {
            let start = std::time::Instant::now();
            simulate_compile(self.lines_of_code);
            result.add_iteration(start.elapsed());
        }
        
        let throughput = (self.lines_of_code as f64) / result.avg_time.as_secs_f64();
        result.set_throughput(throughput);
        
        result
    }
}

pub struct RuntimeBench {
    pub name: String,
    pub iterations: usize,
}

impl RuntimeBench {
    pub fn new(name: &str, iterations: usize) -> Self {
        RuntimeBench {
            name: name.to_string(),
            iterations,
        }
    }
}

pub struct MemoryBench {
    pub name: String,
    pub allocations: usize,
}

impl MemoryBench {
    pub fn new(name: &str, allocations: usize) -> Self {
        MemoryBench {
            name: name.to_string(),
            allocations,
        }
    }
}
