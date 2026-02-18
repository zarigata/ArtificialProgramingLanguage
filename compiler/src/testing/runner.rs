//! Test runner for parallel test execution

use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use crossbeam_channel::{Sender, Receiver, bounded};

use super::{TestCase, TestStatus, TestStats, TestFailure};
use super::fixture::TestContext;

#[derive(Debug, Clone)]
pub struct TestConfig {
    pub parallel: bool,
    pub max_threads: usize,
    pub fail_fast: bool,
    pub timeout: Duration,
    pub filter: Option<String>,
    pub tags: Vec<String>,
    pub exclude_tags: Vec<String>,
    pub verbose: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        TestConfig {
            parallel: true,
            max_threads: num_cpus::get(),
            fail_fast: false,
            timeout: Duration::from_secs(60),
            filter: None,
            tags: Vec::new(),
            exclude_tags: Vec::new(),
            verbose: false,
        }
    }
}

impl TestConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }

    pub fn parallel(mut self, max_threads: usize) -> Self {
        self.parallel = true;
        self.max_threads = max_threads;
        self
    }

    pub fn fail_fast(mut self) -> Self {
        self.fail_fast = true;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn filter(mut self, pattern: &str) -> Self {
        self.filter = Some(pattern.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn exclude_tag(mut self, tag: &str) -> Self {
        self.exclude_tags.push(tag.to_string());
        self
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    fn should_run(&self, test: &TestCase) -> bool {
        if test.ignore {
            return false;
        }

        if let Some(ref filter) = self.filter {
            if !test.name.contains(filter) && !test.module.contains(filter) {
                return false;
            }
        }

        if !self.tags.is_empty() {
            let has_tag = self.tags.iter().any(|t| test.tags.contains(t));
            if !has_tag {
                return false;
            }
        }

        if !self.exclude_tags.is_empty() {
            let has_excluded = self.exclude_tags.iter().any(|t| test.tags.contains(t));
            if has_excluded {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub test: TestCase,
    pub status: TestStatus,
    pub duration: Duration,
    pub failure: Option<TestFailure>,
    pub output: Option<String>,
}

impl TestResult {
    pub fn passed(test: TestCase, duration: Duration) -> Self {
        TestResult {
            test,
            status: TestStatus::Passed,
            duration,
            failure: None,
            output: None,
        }
    }

    pub fn failed(test: TestCase, duration: Duration, failure: TestFailure) -> Self {
        TestResult {
            test,
            status: TestStatus::Failed,
            duration,
            failure: Some(failure),
            output: None,
        }
    }

    pub fn skipped(test: TestCase) -> Self {
        TestResult {
            test,
            status: TestStatus::Skipped,
            duration: Duration::ZERO,
            failure: None,
            output: None,
        }
    }

    pub fn panicked(test: TestCase, duration: Duration, message: &str) -> Self {
        let test_name = test.full_name();
        TestResult {
            test,
            status: TestStatus::Panicked,
            duration,
            failure: Some(TestFailure::new(&test_name, message)),
            output: None,
        }
    }

    pub fn timed_out(test: TestCase) -> Self {
        let test_name = test.full_name();
        TestResult {
            test,
            status: TestStatus::TimedOut,
            duration: Duration::ZERO,
            failure: Some(TestFailure::new(&test_name, "Test timed out")),
            output: None,
        }
    }
}

pub type BoxedTest = Box<dyn FnOnce(&mut TestContext) -> Result<(), String> + Send>;

pub struct TestRunner {
    config: TestConfig,
    tests: Vec<(TestCase, BoxedTest)>,
}

impl TestRunner {
    pub fn new() -> Self {
        TestRunner {
            config: TestConfig::default(),
            tests: Vec::new(),
        }
    }

    pub fn with_config(config: TestConfig) -> Self {
        TestRunner {
            config,
            tests: Vec::new(),
        }
    }

    pub fn add_test<F>(&mut self, test: TestCase, f: F)
    where
        F: FnOnce(&mut TestContext) -> Result<(), String> + Send + 'static,
    {
        self.tests.push((test, Box::new(f)));
    }

    pub fn run(self) -> (TestStats, Vec<TestResult>) {
        let total_tests: usize = self.tests.iter()
            .filter(|(test, _)| self.config.should_run(test))
            .count();

        self.run_sequential(total_tests)
    }

    fn run_sequential(self, total: usize) -> (TestStats, Vec<TestResult>) {
        let mut stats = TestStats::new();
        let mut results = Vec::new();
        let start_time = Instant::now();

        stats.total = total;

        for (test, f) in self.tests {
            if !self.config.should_run(&test) {
                results.push(TestResult::skipped(test));
                continue;
            }

            let result = Self::run_single_test(test, f, &self.config);
            
            match result.status {
                TestStatus::Passed => stats.passed += 1,
                TestStatus::Failed => stats.failed += 1,
                TestStatus::Skipped => stats.skipped += 1,
                TestStatus::Panicked => stats.panicked += 1,
                TestStatus::TimedOut => stats.timed_out += 1,
            }

            let should_stop = self.config.fail_fast && !result.status.is_ok();
            results.push(result);

            if should_stop {
                break;
            }
        }

        stats.duration = start_time.elapsed();
        (stats, results)
    }

    fn run_single_test(test: TestCase, f: BoxedTest, config: &TestConfig) -> TestResult {
        let timeout = test.timeout.unwrap_or(config.timeout);
        let test_name = test.full_name();
        let should_panic = test.should_panic;
        let test_name_clone = test_name.clone();

        let start = Instant::now();

        let (tx, rx) = std::sync::mpsc::channel();
        
        thread::spawn(move || {
            let mut ctx = TestContext::new(&test_name_clone);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f(&mut ctx)
            }));
            let _ = tx.send(result);
        });

        match rx.recv_timeout(timeout) {
            Ok(result) => {
                let duration = start.elapsed();
                
                match result {
                    Ok(Ok(())) => {
                        if should_panic {
                            TestResult::failed(
                                test,
                                duration,
                                TestFailure::new(&test_name, "Expected panic but test passed"),
                            )
                        } else {
                            TestResult::passed(test, duration)
                        }
                    }
                    Ok(Err(msg)) => {
                        if should_panic {
                            TestResult::passed(test, duration)
                        } else {
                            TestResult::failed(test, duration, TestFailure::new(&test_name, &msg))
                        }
                    }
                    Err(_) => {
                        if should_panic {
                            TestResult::passed(test, duration)
                        } else {
                            TestResult::panicked(test, duration, "Test panicked")
                        }
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                TestResult::timed_out(test)
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                TestResult::panicked(test, start.elapsed(), "Test thread disconnected")
            }
        }
    }
}

impl Default for TestRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_filter() {
        let config = TestConfig::new().filter("my_test");
        
        let matching = TestCase::new("my_test", "module");
        let non_matching = TestCase::new("other_test", "module");
        
        assert!(config.should_run(&matching));
        assert!(!config.should_run(&non_matching));
    }

    #[test]
    fn test_config_tags() {
        let config = TestConfig::new()
            .with_tag("unit")
            .exclude_tag("slow");
        
        let matching = TestCase::new("test", "module").with_tag("unit");
        let excluded = TestCase::new("test", "module").with_tag("unit").with_tag("slow");
        let no_tag = TestCase::new("test", "module");
        
        assert!(config.should_run(&matching));
        assert!(!config.should_run(&excluded));
        assert!(!config.should_run(&no_tag));
    }

    #[test]
    fn test_runner_basic() {
        let mut runner = TestRunner::new();
        
        runner.add_test(
            TestCase::new("passing_test", "tests"),
            |_| Ok(())
        );
        
        runner.add_test(
            TestCase::new("failing_test", "tests"),
            |_| Err("expected failure".to_string())
        );
        
        let (stats, results) = runner.run();
        
        assert_eq!(stats.total, 2);
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.failed, 1);
    }
}
