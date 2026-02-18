//! Comparison benchmarks against other languages

use crate::BenchmarkSuite;
use anyhow::Result;

pub fn run_all(suite: &mut BenchmarkSuite) -> Result<()> {
    compare_with_rust(suite)?;
    compare_with_go(suite)?;
    compare_with_c(suite)?;
    Ok(())
}

pub fn compare_with_rust(suite: &mut BenchmarkSuite) -> Result<()> {
    suite.run("vs_rust_fibonacci", 100, || {
        let _ = fib_iterative(35);
    });

    suite.run("vs_rust_sort", 100, || {
        let _ = sort_test(10000);
    });

    suite.run("vs_rust_hashmap", 100, || {
        let _ = hashmap_test(10000);
    });

    Ok(())
}

pub fn compare_with_go(suite: &mut BenchmarkSuite) -> Result<()> {
    suite.run("vs_go_goroutine_sim", 50, || {
        let _ = simulate_concurrency(100);
    });

    suite.run("vs_go_json_parse", 100, || {
        let _ = simulate_json_parse(1000);
    });

    suite.run("vs_go_http_sim", 50, || {
        let _ = simulate_http_handler();
    });

    Ok(())
}

pub fn compare_with_c(suite: &mut BenchmarkSuite) -> Result<()> {
    suite.run("vs_c_pointer_ops", 1000, || {
        let _ = pointer_arithmetic_test(1000);
    });

    suite.run("vs_c_memcpy_sim", 100, || {
        let _ = memcpy_simulate(1024 * 1024);
    });

    suite.run("vs_c_bitwise", 1000, || {
        let _ = bitwise_ops(10000);
    });

    Ok(())
}

fn fib_iterative(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 2..=n {
        let temp = a.wrapping_add(b);
        a = b;
        b = temp;
    }
    b
}

fn sort_test(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).rev().collect();
    v.sort_unstable();
    v
}

fn hashmap_test(n: usize) -> usize {
    use std::collections::HashMap;
    let mut m: HashMap<usize, usize> = HashMap::with_capacity(n);
    for i in 0..n {
        m.insert(i, i.wrapping_mul(2));
    }
    m.len()
}

fn simulate_concurrency(tasks: usize) -> usize {
    use std::thread;
    let handles: Vec<_> = (0..tasks.min(8))
        .map(|i| {
            thread::spawn(move || {
                let mut sum = 0usize;
                for j in 0..1000 {
                    sum = sum.wrapping_add(j);
                }
                sum
            })
        })
        .collect();
    
    handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
}

fn simulate_json_parse(size: usize) -> usize {
    let json: String = (0..size)
        .map(|i| format!(r#"{{"key{}":"value{}"}}"#, i % 100, i % 100))
        .collect::<Vec<_>>()
        .join(",");
    
    json.len()
}

fn simulate_http_handler() -> usize {
    let request = "GET /api/users HTTP/1.1\r\nHost: example.com\r\n\r\n";
    let response = format!("HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, World!");
    request.len() + response.len()
}

fn pointer_arithmetic_test(n: usize) -> usize {
    let mut v: Vec<usize> = (0..n).collect();
    let ptr = v.as_mut_ptr();
    let mut sum = 0usize;
    
    for i in 0..n {
        unsafe {
            sum = sum.wrapping_add(*ptr.add(i));
        }
    }
    
    sum
}

fn memcpy_simulate(size: usize) -> usize {
    let src: Vec<u8> = vec![42; size];
    let mut dst: Vec<u8> = vec![0; size];
    dst.copy_from_slice(&src);
    dst.len()
}

fn bitwise_ops(n: usize) -> u64 {
    let mut result: u64 = 0;
    for i in 0..n {
        result ^= (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
        result = result.rotate_left(5);
        result = result.wrapping_add(i as u64);
    }
    result
}

pub struct ComparisonResult {
    pub vez_time_ns: u64,
    pub other_time_ns: u64,
    pub ratio: f64,
    pub winner: String,
}

impl ComparisonResult {
    pub fn new(vez: u64, other: u64, other_name: &str) -> Self {
        let ratio = vez as f64 / other as f64;
        let winner = if ratio < 1.0 { "VeZ" } else { other_name };
        
        ComparisonResult {
            vez_time_ns: vez,
            other_time_ns: other,
            ratio,
            winner: winner.to_string(),
        }
    }
}

impl std::fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ratio_str = if self.ratio < 1.0 {
            format!("{:.2}x faster", 1.0 / self.ratio)
        } else {
            format!("{:.2}x slower", self.ratio)
        };
        write!(
            f,
            "VeZ: {}ns vs Other: {}ns ({})",
            self.vez_time_ns, self.other_time_ns, ratio_str
        )
    }
}
