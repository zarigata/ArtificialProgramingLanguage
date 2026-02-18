//! Test reporting and output formatting

use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use super::{TestResult, TestStats, TestStatus, TestFailure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Plain,
    Color,
    Json,
    Junit,
    Tap,
    Html,
}

pub struct TestReport {
    format: ReportFormat,
    verbose: bool,
}

impl TestReport {
    pub fn new(format: ReportFormat) -> Self {
        TestReport {
            format,
            verbose: false,
        }
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    pub fn render(&self, stats: &TestStats, results: &[TestResult]) -> String {
        match self.format {
            ReportFormat::Plain => self.render_plain(stats, results),
            ReportFormat::Color => self.render_color(stats, results),
            ReportFormat::Json => self.render_json(stats, results),
            ReportFormat::Junit => self.render_junit(stats, results),
            ReportFormat::Tap => self.render_tap(stats, results),
            ReportFormat::Html => self.render_html(stats, results),
        }
    }

    fn render_plain(&self, stats: &TestStats, results: &[TestResult]) -> String {
        let mut output = String::new();

        for result in results {
            let status = match result.status {
                TestStatus::Passed => "PASS",
                TestStatus::Failed => "FAIL",
                TestStatus::Skipped => "SKIP",
                TestStatus::Panicked => "PANIC",
                TestStatus::TimedOut => "TIMEOUT",
            };

            output.push_str(&format!(
                "{} {} ({:?})\n",
                status,
                result.test.full_name(),
                result.duration
            ));

            if let Some(ref failure) = result.failure {
                output.push_str(&format!("  Error: {}\n", failure.message));
                if let Some(ref expected) = failure.expected {
                    output.push_str(&format!("  Expected: {}\n", expected));
                }
                if let Some(ref actual) = failure.actual {
                    output.push_str(&format!("  Actual: {}\n", actual));
                }
            }
        }

        output.push_str("\n");
        output.push_str(&self.summary_plain(stats));

        output
    }

    fn render_color(&self, stats: &TestStats, results: &[TestResult]) -> String {
        let mut output = String::new();

        for result in results {
            let color = result.status.color();
            let status = result.status.symbol();
            let reset = "\x1b[0m";

            output.push_str(&format!(
                "{}{}{} {} ({:?})\n",
                color,
                status,
                reset,
                result.test.full_name(),
                result.duration
            ));

            if let Some(ref failure) = result.failure {
                output.push_str(&format!("\x1b[31m  Error: {}\x1b[0m\n", failure.message));
                if let Some(ref expected) = failure.expected {
                    output.push_str(&format!("  \x1b[32mExpected:\x1b[0m {}\n", expected));
                }
                if let Some(ref actual) = failure.actual {
                    output.push_str(&format!("  \x1b[31mActual:\x1b[0m {}\n", actual));
                }
            }
        }

        output.push_str("\n");
        output.push_str(&self.summary_color(stats));

        output
    }

    fn render_json(&self, stats: &TestStats, results: &[TestResult]) -> String {
        let mut json = String::from("{\n");
        
        json.push_str("  \"summary\": {\n");
        json.push_str(&format!("    \"total\": {},\n", stats.total));
        json.push_str(&format!("    \"passed\": {},\n", stats.passed));
        json.push_str(&format!("    \"failed\": {},\n", stats.failed));
        json.push_str(&format!("    \"skipped\": {},\n", stats.skipped));
        json.push_str(&format!("    \"duration_ms\": {}\n", stats.duration.as_millis()));
        json.push_str("  },\n");
        
        json.push_str("  \"tests\": [\n");
        
        for (i, result) in results.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"name\": \"{}\",\n", result.test.full_name()));
            json.push_str(&format!("      \"status\": \"{:?}\",\n", result.status));
            json.push_str(&format!("      \"duration_ms\": {}", result.duration.as_millis()));
            
            if let Some(ref failure) = result.failure {
                json.push_str(",\n      \"failure\": {\n");
                json.push_str(&format!("        \"message\": \"{}\"", escape_json(&failure.message)));
                if let Some(ref expected) = failure.expected {
                    json.push_str(&format!(",\n        \"expected\": \"{}\"", escape_json(expected)));
                }
                if let Some(ref actual) = failure.actual {
                    json.push_str(&format!(",\n        \"actual\": \"{}\"", escape_json(actual)));
                }
                json.push_str("\n      }");
            }
            
            json.push_str("\n    }");
            if i < results.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        
        json.push_str("  ]\n");
        json.push_str("}\n");
        
        json
    }

    fn render_junit(&self, stats: &TestStats, results: &[TestResult]) -> String {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<testsuites tests=\"{}\" failures=\"{}\" time=\"{:.3}\">\n",
            stats.total,
            stats.failures(),
            stats.duration.as_secs_f64()
        ));
        xml.push_str(&format!(
            "  <testsuite name=\"vez-tests\" tests=\"{}\" failures=\"{}\">\n",
            stats.total,
            stats.failures()
        ));

        for result in results {
            xml.push_str(&format!(
                "    <testcase name=\"{}\" classname=\"{}\" time=\"{:.3}\">\n",
                escape_xml(&result.test.name),
                escape_xml(&result.test.module),
                result.duration.as_secs_f64()
            ));

            match result.status {
                TestStatus::Skipped => {
                    xml.push_str("      <skipped/>\n");
                }
                TestStatus::Failed | TestStatus::Panicked | TestStatus::TimedOut => {
                    if let Some(ref failure) = result.failure {
                        xml.push_str(&format!(
                            "      <failure message=\"{}\">{}</failure>\n",
                            escape_xml(&failure.message),
                            escape_xml(&failure.message)
                        ));
                    }
                }
                _ => {}
            }

            xml.push_str("    </testcase>\n");
        }

        xml.push_str("  </testsuite>\n");
        xml.push_str("</testsuites>\n");

        xml
    }

    fn render_tap(&self, stats: &TestStats, results: &[TestResult]) -> String {
        let mut tap = String::new();
        tap.push_str(&format!("1..{}\n", stats.total));

        for (i, result) in results.iter().enumerate() {
            let test_num = i + 1;
            match result.status {
                TestStatus::Passed => {
                    tap.push_str(&format!("ok {} - {}\n", test_num, result.test.full_name()));
                }
                TestStatus::Skipped => {
                    tap.push_str(&format!(
                        "ok {} - {} # SKIP\n",
                        test_num,
                        result.test.full_name()
                    ));
                }
                _ => {
                    tap.push_str(&format!(
                        "not ok {} - {}\n",
                        test_num,
                        result.test.full_name()
                    ));
                    if let Some(ref failure) = result.failure {
                        tap.push_str(&format!("  ---\n  message: {}\n  ---\n", failure.message));
                    }
                }
            }
        }

        tap
    }

    fn render_html(&self, stats: &TestStats, results: &[TestResult]) -> String {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>VeZ Test Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(REPORT_STYLES_CSS);
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str("<h1>Test Report</h1>\n");
        html.push_str("<div class=\"summary\">\n");
        html.push_str(&format!(
            "<span class=\"total\">Total: {}</span>\n",
            stats.total
        ));
        html.push_str(&format!(
            "<span class=\"passed\">Passed: {}</span>\n",
            stats.passed
        ));
        html.push_str(&format!(
            "<span class=\"failed\">Failed: {}</span>\n",
            stats.failures()
        ));
        html.push_str(&format!(
            "<span class=\"duration\">Duration: {:?}</span>\n",
            stats.duration
        ));
        html.push_str("</div>\n");

        html.push_str("<table>\n");
        html.push_str("<tr><th>Status</th><th>Test</th><th>Duration</th></tr>\n");

        for result in results {
            let status_class = match result.status {
                TestStatus::Passed => "passed",
                TestStatus::Failed => "failed",
                TestStatus::Skipped => "skipped",
                TestStatus::Panicked => "panicked",
                TestStatus::TimedOut => "timedout",
            };

            html.push_str(&format!(
                "<tr class=\"{}\">\n",
                status_class
            ));
            html.push_str(&format!(
                "  <td>{}</td>\n",
                result.status.symbol()
            ));
            html.push_str(&format!(
                "  <td>{}</td>\n",
                escape_html(&result.test.full_name())
            ));
            html.push_str(&format!(
                "  <td>{:?}</td>\n",
                result.duration
            ));
            html.push_str("</tr>\n");

            if let Some(ref failure) = result.failure {
                html.push_str("<tr class=\"failure-details\">\n");
                html.push_str(&format!(
                    "  <td colspan=\"3\"><pre>{}</pre></td>\n",
                    escape_html(&failure.message)
                ));
                html.push_str("</tr>\n");
            }
        }

        html.push_str("</table>\n");
        html.push_str("</body>\n</html>\n");

        html
    }

    fn summary_plain(&self, stats: &TestStats) -> String {
        format!(
            "Test Results: {} passed, {} failed, {} skipped, {} panicked, {} timed out\n\
             Duration: {:?}",
            stats.passed,
            stats.failed,
            stats.skipped,
            stats.panicked,
            stats.timed_out,
            stats.duration
        )
    }

    fn summary_color(&self, stats: &TestStats) -> String {
        let reset = "\x1b[0m";
        let green = "\x1b[32m";
        let red = "\x1b[31m";
        let yellow = "\x1b[33m";
        let cyan = "\x1b[36m";

        format!(
            "{}Test Results:{} {}{} passed{}, {}{} failed{}, {}{} skipped{}, {}{} panicked{}, {}{} timed out{}\n\
             Duration: {:?}",
            cyan, reset,
            green, stats.passed, reset,
            red, stats.failed, reset,
            yellow, stats.skipped, reset,
            red, stats.panicked, reset,
            yellow, stats.timed_out, reset,
            stats.duration
        )
    }

    pub fn print(&self, stats: &TestStats, results: &[TestResult]) -> io::Result<()> {
        let output = self.render(stats, results);
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        handle.write_all(output.as_bytes())
    }

    pub fn write_to_file(&self, stats: &TestStats, results: &[TestResult], path: &Path) -> io::Result<()> {
        let output = self.render(stats, results);
        std::fs::write(path, output)
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub const REPORT_STYLES_CSS: &str = r#"
body { font-family: sans-serif; margin: 20px; }
.summary { margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
.summary span { margin-right: 20px; }
.passed { color: green; }
.failed { color: red; }
table { width: 100%; border-collapse: collapse; }
tr { border-bottom: 1px solid #ddd; }
td, th { padding: 10px; text-align: left; }
tr.passed { background: #e8f5e9; }
tr.failed { background: #ffebee; }
tr.skipped { background: #fff8e1; }
tr.panicked { background: #fce4ec; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestCase;

    fn make_test_results() -> (TestStats, Vec<TestResult>) {
        let mut stats = TestStats::new();
        stats.total = 3;
        stats.passed = 2;
        stats.failed = 1;

        let results = vec![
            TestResult::passed(TestCase::new("test1", "mod"), Duration::from_millis(10)),
            TestResult::passed(TestCase::new("test2", "mod"), Duration::from_millis(15)),
            TestResult::failed(
                TestCase::new("test3", "mod"),
                Duration::from_millis(5),
                TestFailure::new("mod::test3", "assertion failed"),
            ),
        ];

        (stats, results)
    }

    #[test]
    fn test_plain_report() {
        let (stats, results) = make_test_results();
        let report = TestReport::new(ReportFormat::Plain);
        let output = report.render(&stats, &results);
        
        assert!(output.contains("PASS"));
        assert!(output.contains("FAIL"));
    }

    #[test]
    fn test_json_report() {
        let (stats, results) = make_test_results();
        let report = TestReport::new(ReportFormat::Json);
        let output = report.render(&stats, &results);
        
        assert!(output.contains("\"total\": 3"));
        assert!(output.contains("\"passed\": 2"));
    }

    #[test]
    fn test_tap_report() {
        let (stats, results) = make_test_results();
        let report = TestReport::new(ReportFormat::Tap);
        let output = report.render(&stats, &results);
        
        assert!(output.contains("1..3"));
        assert!(output.starts_with("1.."));
    }
}
