// VeZ Testing Framework
// Comprehensive unit, integration, and property-based testing

use std::collections::HashMap;
use std::time::{Duration, Instant};

// Test result
#[derive(Debug, Clone, PartialEq)]
pub enum TestResult {
    Pass,
    Fail(String),
    Skip(String),
    Timeout,
}

// Test case
pub struct TestCase {
    pub name: String,
    pub test_fn: Box<dyn Fn() -> TestResult>,
    pub timeout: Option<Duration>,
    pub tags: Vec<String>,
}

impl TestCase {
    pub fn new(name: String, test_fn: Box<dyn Fn() -> TestResult>) -> Self {
        TestCase {
            name,
            test_fn,
            timeout: Some(Duration::from_secs(30)),
            tags: Vec::new(),
        }
    }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn run(&self) -> TestResult {
        let start = Instant::now();
        
        // Run test with timeout
        let result = (self.test_fn)();
        
        if let Some(timeout) = self.timeout {
            if start.elapsed() > timeout {
                return TestResult::Timeout;
            }
        }
        
        result
    }
}

// Test suite
pub struct TestSuite {
    pub name: String,
    pub tests: Vec<TestCase>,
    pub setup: Option<Box<dyn Fn()>>,
    pub teardown: Option<Box<dyn Fn()>>,
}

impl TestSuite {
    pub fn new(name: String) -> Self {
        TestSuite {
            name,
            tests: Vec::new(),
            setup: None,
            teardown: None,
        }
    }
    
    pub fn add_test(&mut self, test: TestCase) {
        self.tests.push(test);
    }
    
    pub fn set_setup(&mut self, setup: Box<dyn Fn()>) {
        self.setup = Some(setup);
    }
    
    pub fn set_teardown(&mut self, teardown: Box<dyn Fn()>) {
        self.teardown = Some(teardown);
    }
    
    pub fn run(&self) -> TestReport {
        let mut report = TestReport::new(self.name.clone());
        
        // Run setup
        if let Some(setup) = &self.setup {
            setup();
        }
        
        // Run each test
        for test in &self.tests {
            let start = Instant::now();
            let result = test.run();
            let duration = start.elapsed();
            
            report.add_result(test.name.clone(), result, duration);
        }
        
        // Run teardown
        if let Some(teardown) = &self.teardown {
            teardown();
        }
        
        report
    }
}

// Test report
#[derive(Debug)]
pub struct TestReport {
    pub suite_name: String,
    pub results: Vec<(String, TestResult, Duration)>,
    pub total_time: Duration,
}

impl TestReport {
    pub fn new(suite_name: String) -> Self {
        TestReport {
            suite_name,
            results: Vec::new(),
            total_time: Duration::from_secs(0),
        }
    }
    
    pub fn add_result(&mut self, test_name: String, result: TestResult, duration: Duration) {
        self.results.push((test_name, result, duration));
        self.total_time += duration;
    }
    
    pub fn passed(&self) -> usize {
        self.results.iter()
            .filter(|(_, r, _)| matches!(r, TestResult::Pass))
            .count()
    }
    
    pub fn failed(&self) -> usize {
        self.results.iter()
            .filter(|(_, r, _)| matches!(r, TestResult::Fail(_)))
            .count()
    }
    
    pub fn skipped(&self) -> usize {
        self.results.iter()
            .filter(|(_, r, _)| matches!(r, TestResult::Skip(_)))
            .count()
    }
    
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Test Suite: {}", self.suite_name);
        println!("{}", "=".repeat(60));
        
        for (name, result, duration) in &self.results {
            let status = match result {
                TestResult::Pass => "✓ PASS",
                TestResult::Fail(_) => "✗ FAIL",
                TestResult::Skip(_) => "⊘ SKIP",
                TestResult::Timeout => "⏱ TIMEOUT",
            };
            
            println!("{:50} {} ({:?})", name, status, duration);
            
            if let TestResult::Fail(msg) = result {
                println!("    Error: {}", msg);
            }
        }
        
        println!("{}", "-".repeat(60));
        println!("Total: {} | Passed: {} | Failed: {} | Skipped: {}",
                 self.results.len(), self.passed(), self.failed(), self.skipped());
        println!("Time: {:?}", self.total_time);
        println!("{}", "=".repeat(60));
    }
}

// Property-based testing
pub struct PropertyTest<T> {
    pub name: String,
    pub generator: Box<dyn Fn() -> T>,
    pub property: Box<dyn Fn(&T) -> bool>,
    pub num_tests: usize,
}

impl<T> PropertyTest<T> {
    pub fn new(
        name: String,
        generator: Box<dyn Fn() -> T>,
        property: Box<dyn Fn(&T) -> bool>,
    ) -> Self {
        PropertyTest {
            name,
            generator,
            property,
            num_tests: 100,
        }
    }
    
    pub fn with_num_tests(mut self, num_tests: usize) -> Self {
        self.num_tests = num_tests;
        self
    }
    
    pub fn run(&self) -> TestResult {
        for i in 0..self.num_tests {
            let value = (self.generator)();
            
            if !(self.property)(&value) {
                return TestResult::Fail(format!(
                    "Property failed on test case {} of {}",
                    i + 1, self.num_tests
                ));
            }
        }
        
        TestResult::Pass
    }
}

// Assertion macros
#[macro_export]
macro_rules! assert_test {
    ($cond:expr) => {
        if !$cond {
            return TestResult::Fail(format!(
                "Assertion failed: {} at {}:{}",
                stringify!($cond),
                file!(),
                line!()
            ));
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return TestResult::Fail($msg.to_string());
        }
    };
}

#[macro_export]
macro_rules! assert_eq_test {
    ($left:expr, $right:expr) => {
        let left_val = $left;
        let right_val = $right;
        if left_val != right_val {
            return TestResult::Fail(format!(
                "Assertion failed: left != right\n  left: {:?}\n right: {:?}",
                left_val, right_val
            ));
        }
    };
}

// Benchmark framework
pub struct Benchmark {
    pub name: String,
    pub bench_fn: Box<dyn Fn()>,
    pub iterations: usize,
}

impl Benchmark {
    pub fn new(name: String, bench_fn: Box<dyn Fn()>) -> Self {
        Benchmark {
            name,
            bench_fn,
            iterations: 1000,
        }
    }
    
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
    
    pub fn run(&self) -> BenchmarkResult {
        let mut durations = Vec::new();
        
        // Warmup
        for _ in 0..10 {
            (self.bench_fn)();
        }
        
        // Actual benchmark
        for _ in 0..self.iterations {
            let start = Instant::now();
            (self.bench_fn)();
            durations.push(start.elapsed());
        }
        
        BenchmarkResult::new(self.name.clone(), durations)
    }
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub median_time: Duration,
}

impl BenchmarkResult {
    pub fn new(name: String, mut durations: Vec<Duration>) -> Self {
        let iterations = durations.len();
        let total_time: Duration = durations.iter().sum();
        let avg_time = total_time / iterations as u32;
        
        durations.sort();
        let min_time = durations[0];
        let max_time = durations[iterations - 1];
        let median_time = durations[iterations / 2];
        
        BenchmarkResult {
            name,
            iterations,
            total_time,
            avg_time,
            min_time,
            max_time,
            median_time,
        }
    }
    
    pub fn print_summary(&self) {
        println!("\nBenchmark: {}", self.name);
        println!("  Iterations: {}", self.iterations);
        println!("  Total time: {:?}", self.total_time);
        println!("  Average:    {:?}", self.avg_time);
        println!("  Median:     {:?}", self.median_time);
        println!("  Min:        {:?}", self.min_time);
        println!("  Max:        {:?}", self.max_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_test_case() {
        let test = TestCase::new(
            "sample_test".to_string(),
            Box::new(|| TestResult::Pass),
        );
        
        let result = test.run();
        assert_eq!(result, TestResult::Pass);
    }
    
    #[test]
    fn test_test_suite() {
        let mut suite = TestSuite::new("sample_suite".to_string());
        
        suite.add_test(TestCase::new(
            "test1".to_string(),
            Box::new(|| TestResult::Pass),
        ));
        
        suite.add_test(TestCase::new(
            "test2".to_string(),
            Box::new(|| TestResult::Pass),
        ));
        
        let report = suite.run();
        assert_eq!(report.passed(), 2);
        assert_eq!(report.failed(), 0);
    }
}
