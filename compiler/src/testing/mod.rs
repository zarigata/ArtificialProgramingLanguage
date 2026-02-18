//! Professional Testing Framework for VeZ
//!
//! Provides a comprehensive testing infrastructure with:
//! - Test discovery and organization
//! - Rich assertions and matchers
//! - Test fixtures and lifecycle hooks
//! - Parallel test execution
//! - Rich reporting and output

pub mod assert;
pub mod runner;
pub mod report;
pub mod fixture;
pub mod mock;
pub mod bench;

pub use assert::{Assertion, AssertResult};
pub use runner::{TestRunner, TestConfig, TestResult};
pub use report::{TestReport, ReportFormat};
pub use fixture::{Fixture, TestContext};
pub use mock::{Mock, MockFn};
pub use bench::{Benchmark, BenchResult};

use std::time::Duration;

pub type TestFn = Box<dyn FnOnce(&mut TestContext) -> TestResult + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TestId(pub u64);

#[derive(Debug, Clone)]
pub struct TestCase {
    pub id: TestId,
    pub name: String,
    pub module: String,
    pub tags: Vec<String>,
    pub should_panic: bool,
    pub ignore: bool,
    pub timeout: Option<Duration>,
}

impl TestCase {
    pub fn new(name: &str, module: &str) -> Self {
        TestCase {
            id: TestId(0),
            name: name.to_string(),
            module: module.to_string(),
            tags: Vec::new(),
            should_panic: false,
            ignore: false,
            timeout: None,
        }
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn should_panic(mut self) -> Self {
        self.should_panic = true;
        self
    }

    pub fn ignore(mut self) -> Self {
        self.ignore = true;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn full_name(&self) -> String {
        if self.module.is_empty() {
            self.name.clone()
        } else {
            format!("{}::{}", self.module, self.name)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Panicked,
    TimedOut,
}

impl TestStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, TestStatus::Passed)
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            TestStatus::Passed => "✓",
            TestStatus::Failed => "✗",
            TestStatus::Skipped => "○",
            TestStatus::Panicked => "!",
            TestStatus::TimedOut => "⏱",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            TestStatus::Passed => "\x1b[32m",
            TestStatus::Failed => "\x1b[31m",
            TestStatus::Skipped => "\x1b[33m",
            TestStatus::Panicked => "\x1b[35m",
            TestStatus::TimedOut => "\x1b[36m",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TestStats {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub panicked: usize,
    pub timed_out: usize,
    pub duration: Duration,
}

impl TestStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64 * 100.0
        }
    }

    pub fn failures(&self) -> usize {
        self.failed + self.panicked + self.timed_out
    }

    pub fn all_passed(&self) -> bool {
        self.failures() == 0 && self.skipped == 0
    }
}

#[derive(Debug, Clone)]
pub struct TestFailure {
    pub test_name: String,
    pub message: String,
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub location: Option<String>,
    pub backtrace: Option<String>,
}

impl TestFailure {
    pub fn new(test_name: &str, message: &str) -> Self {
        TestFailure {
            test_name: test_name.to_string(),
            message: message.to_string(),
            expected: None,
            actual: None,
            location: None,
            backtrace: None,
        }
    }

    pub fn with_expected(mut self, expected: &str) -> Self {
        self.expected = Some(expected.to_string());
        self
    }

    pub fn with_actual(mut self, actual: &str) -> Self {
        self.actual = Some(actual.to_string());
        self
    }

    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }
}

#[macro_export]
macro_rules! test_case {
    ($name:ident, $body:block) => {
        #[test]
        fn $name() {
            let mut ctx = $crate::testing::TestContext::new(stringify!($name));
            $body
        }
    };
}

#[macro_export]
macro_rules! describe {
    ($name:expr, { $($test:item)* }) => {
        mod $name {
            use super::*;
            $($test)*
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_creation() {
        let tc = TestCase::new("my_test", "mymodule")
            .with_tag("unit")
            .with_timeout(Duration::from_secs(5));
        
        assert_eq!(tc.name, "my_test");
        assert_eq!(tc.full_name(), "mymodule::my_test");
        assert!(tc.tags.contains(&"unit".to_string()));
    }

    #[test]
    fn test_stats() {
        let mut stats = TestStats::new();
        stats.total = 10;
        stats.passed = 8;
        stats.failed = 2;
        
        assert!((stats.success_rate() - 80.0).abs() < 0.01);
        assert!(!stats.all_passed());
    }

    #[test]
    fn test_status() {
        assert!(TestStatus::Passed.is_ok());
        assert!(!TestStatus::Failed.is_ok());
        assert_eq!(TestStatus::Passed.symbol(), "✓");
    }
}
