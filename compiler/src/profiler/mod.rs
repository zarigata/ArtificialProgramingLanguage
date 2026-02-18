//! Profiler Integration
//!
//! Provides integration with various profiling tools and performance analysis.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Profiler type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilerType {
    /// CPU sampling profiler
    CpuSample,
    /// CPU tracing profiler (full call graph)
    CpuTrace,
    /// Memory allocation profiler
    Memory,
    /// Heap profiler
    Heap,
    /// Lock contention profiler
    Lock,
    /// I/O profiler
    Io,
    /// GPU profiler
    Gpu,
    /// Cache profiler
    Cache,
    /// Branch prediction profiler
    Branch,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Profiler type
    pub profiler_type: ProfilerType,
    /// Sampling frequency (Hz)
    pub sample_frequency: u32,
    /// Output file path
    pub output_path: String,
    /// Maximum stack depth
    pub max_stack_depth: u32,
    /// Include kernel frames
    pub include_kernel: bool,
    /// Enable thread profiling
    pub profile_threads: bool,
    /// Duration limit (None = unlimited)
    pub duration_limit: Option<Duration>,
    /// Memory limit in bytes (None = unlimited)
    pub memory_limit: Option<usize>,
    /// Filter by function name pattern
    pub function_filter: Option<String>,
    /// Enable debug info
    pub debug_info: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        ProfilerConfig {
            profiler_type: ProfilerType::CpuSample,
            sample_frequency: 99,
            output_path: "profile.data".to_string(),
            max_stack_depth: 128,
            include_kernel: false,
            profile_threads: true,
            duration_limit: None,
            memory_limit: None,
            function_filter: None,
            debug_info: true,
        }
    }
}

/// Profile sample
#[derive(Debug, Clone)]
pub struct ProfileSample {
    /// Timestamp
    pub timestamp: u64,
    /// Thread ID
    pub thread_id: u64,
    /// Call stack
    pub stack: Vec<StackFrame>,
    /// Sample weight
    pub weight: u64,
    /// CPU ID
    pub cpu_id: u32,
}

/// Stack frame
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Function name
    pub function: String,
    /// Source file
    pub file: Option<String>,
    /// Line number
    pub line: Option<u32>,
    /// Module/library
    pub module: Option<String>,
    /// Address
    pub address: u64,
    /// Offset from function start
    pub offset: u64,
}

/// Profile statistics
#[derive(Debug, Clone, Default)]
pub struct ProfileStats {
    /// Total samples
    pub total_samples: u64,
    /// Samples per function
    pub function_samples: HashMap<String, u64>,
    /// Samples per source location
    pub location_samples: HashMap<(String, u32), u64>,
    /// Hot paths (most frequent call stacks)
    pub hot_paths: Vec<HotPath>,
    /// Function call counts
    pub call_counts: HashMap<String, u64>,
    /// Total execution time per function
    pub function_time: HashMap<String, Duration>,
    /// Memory allocated per function
    pub memory_allocated: HashMap<String, u64>,
    /// Lock wait time per lock
    pub lock_wait_time: HashMap<String, Duration>,
}

/// Hot execution path
#[derive(Debug, Clone)]
pub struct HotPath {
    /// Call stack
    pub stack: Vec<String>,
    /// Sample count
    pub count: u64,
    /// Percentage of total samples
    pub percentage: f64,
}

/// Memory allocation record
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Size in bytes
    pub size: usize,
    /// Address
    pub address: u64,
    /// Stack trace
    pub stack: Vec<StackFrame>,
    /// Timestamp
    pub timestamp: u64,
    /// Thread ID
    pub thread_id: u64,
}

/// Profiler session
pub struct ProfilerSession {
    config: ProfilerConfig,
    samples: Vec<ProfileSample>,
    stats: ProfileStats,
    start_time: Option<Instant>,
    allocations: Vec<AllocationRecord>,
    active: bool,
}

impl ProfilerSession {
    pub fn new(config: ProfilerConfig) -> Self {
        ProfilerSession {
            config,
            samples: Vec::new(),
            stats: ProfileStats::default(),
            start_time: None,
            allocations: Vec::new(),
            active: false,
        }
    }
    
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.active = true;
    }
    
    pub fn stop(&mut self) {
        self.active = false;
    }
    
    pub fn record_sample(&mut self, sample: ProfileSample) {
        if !self.active {
            return;
        }
        
        self.stats.total_samples += 1;
        
        if let Some(frame) = sample.stack.first() {
            *self.stats.function_samples
                .entry(frame.function.clone())
                .or_insert(0) += sample.weight;
            
            if let (Some(file), Some(line)) = (&frame.file, frame.line) {
                *self.stats.location_samples
                    .entry((file.clone(), line))
                    .or_insert(0) += sample.weight;
            }
        }
        
        self.samples.push(sample);
    }
    
    pub fn record_allocation(&mut self, record: AllocationRecord) {
        if !self.active {
            return;
        }
        
        if let Some(frame) = record.stack.first() {
            *self.stats.memory_allocated
                .entry(frame.function.clone())
                .or_insert(0) += record.size as u64;
        }
        
        self.allocations.push(record);
    }
    
    pub fn get_stats(&self) -> &ProfileStats {
        &self.stats
    }
    
    pub fn analyze(&mut self) -> ProfileAnalysis {
        let total = self.stats.total_samples.max(1);
        
        let hot_functions: Vec<_> = self.stats.function_samples.iter()
            .map(|(name, count)| {
                let percentage = (*count as f64 / total as f64) * 100.0;
                HotFunction { name: name.clone(), count: *count, percentage }
            })
            .filter(|hf| hf.percentage > 0.1)
            .collect();
        
        let recommendations = self.generate_recommendations();
        
        ProfileAnalysis {
            total_samples: self.stats.total_samples,
            duration: self.start_time.map(|t| t.elapsed()),
            hot_functions,
            hot_paths: self.stats.hot_paths.clone(),
            memory_analysis: self.analyze_memory(),
            recommendations,
        }
    }
    
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        for (func, time) in &self.stats.function_time {
            if time.as_millis() > 100 {
                recommendations.push(format!(
                    "Function '{}' takes {}ms - consider optimization",
                    func,
                    time.as_millis()
                ));
            }
        }
        
        for (lock, time) in &self.stats.lock_wait_time {
            if time.as_millis() > 10 {
                recommendations.push(format!(
                    "Lock '{}' has {}ms wait time - consider reducing contention",
                    lock,
                    time.as_millis()
                ));
            }
        }
        
        let mut func_samples: Vec<_> = self.stats.function_samples.iter().collect();
        func_samples.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        
        if let Some((func, count)) = func_samples.first() {
            let count_val = **count;
            let percentage = (count_val as f64 / self.stats.total_samples.max(1) as f64) * 100.0;
            if percentage > 50.0 {
                recommendations.push(format!(
                    "Function '{}' dominates execution ({}%) - optimize hot path",
                    func,
                    percentage as i32
                ));
            }
        }
        
        recommendations
    }
    
    fn analyze_memory(&self) -> MemoryAnalysis {
        let total_allocated: u64 = self.allocations.iter()
            .map(|a| a.size as u64)
            .sum();
        
        let allocation_count = self.allocations.len();
        
        let mut size_histogram: HashMap<usize, u64> = HashMap::new();
        for alloc in &self.allocations {
            let bucket = match alloc.size {
                0..=16 => 16,
                17..=32 => 32,
                33..=64 => 64,
                65..=128 => 128,
                129..=256 => 256,
                257..=512 => 512,
                513..=1024 => 1024,
                1025..=4096 => 4096,
                4097..=16384 => 16384,
                _ => alloc.size.next_power_of_two(),
            };
            *size_histogram.entry(bucket).or_insert(0) += 1;
        }
        
        let peak_memory = self.allocations.iter()
            .scan(0u64, |acc, a| {
                *acc += a.size as u64;
                Some(*acc)
            })
            .max()
            .unwrap_or(0);
        
        MemoryAnalysis {
            total_allocated,
            allocation_count,
            peak_memory,
            size_histogram,
        }
    }
}

/// Hot function information
#[derive(Debug, Clone)]
pub struct HotFunction {
    pub name: String,
    pub count: u64,
    pub percentage: f64,
}

/// Memory analysis results
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    pub total_allocated: u64,
    pub allocation_count: usize,
    pub peak_memory: u64,
    pub size_histogram: HashMap<usize, u64>,
}

/// Profile analysis results
#[derive(Debug, Clone)]
pub struct ProfileAnalysis {
    pub total_samples: u64,
    pub duration: Option<Duration>,
    pub hot_functions: Vec<HotFunction>,
    pub hot_paths: Vec<HotPath>,
    pub memory_analysis: MemoryAnalysis,
    pub recommendations: Vec<String>,
}

/// Profiler report generator
pub struct ReportGenerator;

impl ReportGenerator {
    pub fn generate_text(analysis: &ProfileAnalysis) -> String {
        let mut report = String::new();
        
        report.push_str("=== Profile Analysis Report ===\n\n");
        
        report.push_str(&format!("Total samples: {}\n", analysis.total_samples));
        if let Some(duration) = analysis.duration {
            report.push_str(&format!("Duration: {:?}\n", duration));
        }
        report.push_str("\n");
        
        report.push_str("--- Hot Functions ---\n");
        for hf in &analysis.hot_functions {
            report.push_str(&format!(
                "  {} - {} samples ({:.1}%)\n",
                hf.name, hf.count, hf.percentage
            ));
        }
        report.push_str("\n");
        
        report.push_str("--- Memory Analysis ---\n");
        report.push_str(&format!(
            "  Total allocated: {} bytes\n",
            analysis.memory_analysis.total_allocated
        ));
        report.push_str(&format!(
            "  Peak memory: {} bytes\n",
            analysis.memory_analysis.peak_memory
        ));
        report.push_str(&format!(
            "  Allocation count: {}\n",
            analysis.memory_analysis.allocation_count
        ));
        report.push_str("\n");
        
        if !analysis.recommendations.is_empty() {
            report.push_str("--- Recommendations ---\n");
            for rec in &analysis.recommendations {
                report.push_str(&format!("  * {}\n", rec));
            }
        }
        
        report
    }
    
    pub fn generate_json(analysis: &ProfileAnalysis) -> String {
        serde_json::to_string_pretty(&serde_json::json!({
            "total_samples": analysis.total_samples,
            "duration_ms": analysis.duration.map(|d| d.as_millis() as u64),
            "hot_functions": analysis.hot_functions.iter().map(|hf| serde_json::json!({
                "name": hf.name,
                "count": hf.count,
                "percentage": hf.percentage
            })).collect::<Vec<_>>(),
            "memory": {
                "total_allocated": analysis.memory_analysis.total_allocated,
                "peak_memory": analysis.memory_analysis.peak_memory,
                "allocation_count": analysis.memory_analysis.allocation_count,
            },
            "recommendations": analysis.recommendations,
        })).unwrap_or_default()
    }
    
    pub fn generate_html(analysis: &ProfileAnalysis) -> String {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Profile Report</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str("h1 { color: #333; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #4CAF50; color: white; }\n");
        html.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
        html.push_str(".hot { color: red; font-weight: bold; }\n");
        html.push_str(".warm { color: orange; }\n");
        html.push_str(".bar { height: 20px; background: #4CAF50; }\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        
        html.push_str("<h1>Profile Analysis Report</h1>\n");
        
        html.push_str(&format!(
            "<p><strong>Total samples:</strong> {}</p>\n",
            analysis.total_samples
        ));
        
        html.push_str("<h2>Hot Functions</h2>\n");
        html.push_str("<table>\n<tr><th>Function</th><th>Samples</th><th>%</th></tr>\n");
        
        let max_count = analysis.hot_functions.first()
            .map(|hf| hf.count)
            .unwrap_or(1);
        
        for hf in &analysis.hot_functions {
            let bar_width = (hf.count as f64 / max_count as f64 * 100.0) as i32;
            let class = if hf.percentage > 20.0 { "hot" } else if hf.percentage > 5.0 { "warm" } else { "" };
            
            html.push_str(&format!(
                "<tr class=\"{}\"><td>{}</td><td>{}</td><td>{:.1}%</td></tr>\n",
                class, hf.name, hf.count, hf.percentage
            ));
        }
        html.push_str("</table>\n");
        
        html.push_str("<h2>Memory Analysis</h2>\n");
        html.push_str("<ul>\n");
        html.push_str(&format!(
            "<li>Total allocated: {} bytes</li>\n",
            analysis.memory_analysis.total_allocated
        ));
        html.push_str(&format!(
            "<li>Peak memory: {} bytes</li>\n",
            analysis.memory_analysis.peak_memory
        ));
        html.push_str(&format!(
            "<li>Allocation count: {}</li>\n",
            analysis.memory_analysis.allocation_count
        ));
        html.push_str("</ul>\n");
        
        if !analysis.recommendations.is_empty() {
            html.push_str("<h2>Recommendations</h2>\n<ul>\n");
            for rec in &analysis.recommendations {
                html.push_str(&format!("<li>{}</li>\n", rec));
            }
            html.push_str("</ul>\n");
        }
        
        html.push_str("</body>\n</html>\n");
        html
    }
}

/// Instrumentation helpers for manual profiling
pub struct Instrument;

impl Instrument {
    pub fn time<F, R>(label: &str, f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
    
    pub fn time_iter<T, F, R>(items: &[T], mut f: F) -> Vec<(R, Duration)>
    where
        F: FnMut(&T) -> R,
    {
        items.iter()
            .map(|item| {
                let start = Instant::now();
                let result = f(item);
                (result, start.elapsed())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profiler_session() {
        let config = ProfilerConfig::default();
        let mut session = ProfilerSession::new(config);
        
        session.start();
        
        session.record_sample(ProfileSample {
            timestamp: 0,
            thread_id: 1,
            stack: vec![StackFrame {
                function: "main".to_string(),
                file: Some("test.zari".to_string()),
                line: Some(10),
                module: None,
                address: 0,
                offset: 0,
            }],
            weight: 1,
            cpu_id: 0,
        });
        
        let stats = session.get_stats();
        assert_eq!(stats.total_samples, 1);
        assert!(stats.function_samples.contains_key("main"));
    }
    
    #[test]
    fn test_hot_function() {
        let mut session = ProfilerSession::new(ProfilerConfig::default());
        session.start();
        
        for _ in 0..100 {
            session.record_sample(ProfileSample {
                timestamp: 0,
                thread_id: 1,
                stack: vec![StackFrame {
                    function: "hot_func".to_string(),
                    file: None,
                    line: None,
                    module: None,
                    address: 0,
                    offset: 0,
                }],
                weight: 1,
                cpu_id: 0,
            });
        }
        
        let analysis = session.analyze();
        assert!(!analysis.hot_functions.is_empty());
        assert!(analysis.hot_functions[0].name == "hot_func");
    }
    
    #[test]
    fn test_report_generation() {
        let mut session = ProfilerSession::new(ProfilerConfig::default());
        session.start();
        
        session.record_sample(ProfileSample {
            timestamp: 0,
            thread_id: 1,
            stack: vec![StackFrame {
                function: "test".to_string(),
                file: None,
                line: None,
                module: None,
                address: 0,
                offset: 0,
            }],
            weight: 1,
            cpu_id: 0,
        });
        
        let analysis = session.analyze();
        
        let text_report = ReportGenerator::generate_text(&analysis);
        assert!(text_report.contains("Profile Analysis Report"));
        
        let json_report = ReportGenerator::generate_json(&analysis);
        assert!(json_report.contains("total_samples"));
        
        let html_report = ReportGenerator::generate_html(&analysis);
        assert!(html_report.contains("<!DOCTYPE html>"));
    }
}
