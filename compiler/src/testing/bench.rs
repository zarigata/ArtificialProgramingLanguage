//! Benchmarking support for performance testing

use std::time::{Duration, Instant};
use std::collections::BTreeMap;

pub struct Benchmark {
    name: String,
    samples: Vec<Duration>,
    warmup_iterations: usize,
    measurement_iterations: usize,
}

impl Benchmark {
    pub fn new(name: &str) -> Self {
        Benchmark {
            name: name.to_string(),
            samples: Vec::new(),
            warmup_iterations: 10,
            measurement_iterations: 100,
        }
    }

    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    pub fn run<F>(&mut self, mut f: F)
    where
        F: FnMut(),
    {
        for _ in 0..self.warmup_iterations {
            f();
        }

        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            f();
            self.samples.push(start.elapsed());
        }
    }

    pub fn run_with_setup<F, S>(&mut self, mut setup: S, mut f: F)
    where
        F: FnMut(),
        S: FnMut(),
    {
        for _ in 0..self.warmup_iterations {
            setup();
            f();
        }

        for _ in 0..self.measurement_iterations {
            setup();
            let start = Instant::now();
            f();
            self.samples.push(start.elapsed());
        }
    }

    pub fn finish(self) -> BenchResult {
        BenchResult::new(self.name, self.samples)
    }
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub samples: Vec<Duration>,
    pub mean: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub percentiles: BTreeMap<u8, Duration>,
}

impl BenchResult {
    fn new(name: String, samples: Vec<Duration>) -> Self {
        if samples.is_empty() {
            return BenchResult {
                name,
                samples,
                mean: Duration::ZERO,
                median: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                std_dev: Duration::ZERO,
                percentiles: BTreeMap::new(),
            };
        }

        let mut sorted: Vec<_> = samples.iter().collect();
        sorted.sort();

        let n = samples.len();
        let sum: Duration = samples.iter().sum();
        let mean = sum / n as u32;

        let median = if n % 2 == 0 {
            (*sorted[n / 2 - 1] + *sorted[n / 2]) / 2
        } else {
            *sorted[n / 2]
        };

        let min = **sorted.first().unwrap();
        let max = **sorted.last().unwrap();

        let variance: f64 = samples.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>() / n as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let percentiles = Self::calculate_percentiles(&sorted);

        BenchResult {
            name,
            samples,
            mean,
            median,
            min,
            max,
            std_dev,
            percentiles,
        }
    }

    fn calculate_percentiles(sorted: &[&Duration]) -> BTreeMap<u8, Duration> {
        let mut percentiles = BTreeMap::new();
        let n = sorted.len();

        for p in [10, 25, 50, 75, 90, 95, 99].iter() {
            let idx = ((n as f64 * *p as f64 / 100.0).ceil() as usize).min(n) - 1;
            percentiles.insert(*p, *sorted[idx]);
        }

        percentiles
    }

    pub fn iterations_per_second(&self) -> f64 {
        if self.mean.is_zero() {
            0.0
        } else {
            1_000_000_000.0 / self.mean.as_nanos() as f64
        }
    }

    pub fn ops_per_sec(&self) -> u64 {
        self.iterations_per_second() as u64
    }

    pub fn compare(&self, other: &BenchResult) -> BenchComparison {
        let ratio = self.mean.as_nanos() as f64 / other.mean.as_nanos() as f64;
        
        let faster = if ratio < 1.0 {
            Some(1.0 / ratio)
        } else {
            None
        };

        let slower = if ratio > 1.0 {
            Some(ratio)
        } else {
            None
        };

        BenchComparison {
            base: self.name.clone(),
            other: other.name.clone(),
            ratio,
            faster,
            slower,
            significant: self.is_significantly_different(other),
        }
    }

    fn is_significantly_different(&self, other: &BenchResult) -> bool {
        let combined_std = (self.std_dev.as_nanos() as f64 + other.std_dev.as_nanos() as f64) / 2.0;
        let diff = (self.mean.as_nanos() as f64 - other.mean.as_nanos() as f64).abs();
        diff > 2.0 * combined_std
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Benchmark: {}", self.name)?;
        writeln!(f, "  Mean: {:?}", self.mean)?;
        writeln!(f, "  Median: {:?}", self.median)?;
        writeln!(f, "  Min: {:?}", self.min)?;
        writeln!(f, "  Max: {:?}", self.max)?;
        writeln!(f, "  Std Dev: {:?}", self.std_dev)?;
        writeln!(f, "  Iterations/sec: {:.2}", self.iterations_per_second())?;
        writeln!(f, "  Percentiles:")?;
        for (p, d) in &self.percentiles {
            writeln!(f, "    p{}: {:?}", p, d)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BenchComparison {
    pub base: String,
    pub other: String,
    pub ratio: f64,
    pub faster: Option<f64>,
    pub slower: Option<f64>,
    pub significant: bool,
}

impl std::fmt::Display for BenchComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(faster) = self.faster {
            write!(f, "{} is {:.2}x faster than {}", self.base, faster, self.other)?;
        } else if let Some(slower) = self.slower {
            write!(f, "{} is {:.2}x slower than {}", self.base, slower, self.other)?;
        } else {
            write!(f, "{} and {} have similar performance", self.base, self.other)?;
        }
        
        if self.significant {
            write!(f, " (statistically significant)")?;
        }
        
        Ok(())
    }
}

pub struct BenchGroup {
    name: String,
    results: Vec<BenchResult>,
}

impl BenchGroup {
    pub fn new(name: &str) -> Self {
        BenchGroup {
            name: name.to_string(),
            results: Vec::new(),
        }
    }

    pub fn bench<F>(&mut self, name: &str, f: F) -> &mut Self
    where
        F: FnMut(),
    {
        let mut bench = Benchmark::new(name);
        bench.run(f);
        self.results.push(bench.finish());
        self
    }

    pub fn bench_with_setup<F, S>(&mut self, name: &str, setup: S, f: F) -> &mut Self
    where
        F: FnMut(),
        S: FnMut(),
    {
        let mut bench = Benchmark::new(name);
        bench.run_with_setup(setup, f);
        self.results.push(bench.finish());
        self
    }

    pub fn results(&self) -> &[BenchResult] {
        &self.results
    }

    pub fn summary(&self) -> BenchSummary {
        let total_iterations: u64 = self.results.iter()
            .map(|r| r.samples.len() as u64)
            .sum();

        let total_duration: Duration = self.results.iter()
            .map(|r| r.samples.iter().copied().sum::<Duration>())
            .sum();

        BenchSummary {
            group_name: self.name.clone(),
            benchmark_count: self.results.len(),
            total_iterations,
            total_duration,
            fastest: self.results.iter()
                .min_by_key(|r| r.mean)
                .map(|r| r.name.clone()),
            slowest: self.results.iter()
                .max_by_key(|r| r.mean)
                .map(|r| r.name.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchSummary {
    pub group_name: String,
    pub benchmark_count: usize,
    pub total_iterations: u64,
    pub total_duration: Duration,
    pub fastest: Option<String>,
    pub slowest: Option<String>,
}

#[macro_export]
macro_rules! bench {
    ($name:expr, $iterations:expr, $body:block) => {
        {
            let mut bench = $crate::testing::bench::Benchmark::new($name)
                .with_iterations($iterations);
            bench.run(|| $body);
            bench.finish()
        }
    };
}

#[macro_export]
macro_rules! bench_group {
    ($name:expr, { $($bench_name:expr => $body:block),* $(,)? }) => {
        {
            let mut group = $crate::testing::bench::BenchGroup::new($name);
            $(
                group.bench($bench_name, || $body);
            )*
            group
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_basic() {
        let mut bench = Benchmark::new("test_bench")
            .with_warmup(2)
            .with_iterations(10);
        
        bench.run(|| {
            let mut sum = 0;
            for i in 0..100 {
                sum += i;
            }
        });
        
        let result = bench.finish();
        
        assert_eq!(result.name, "test_bench");
        assert_eq!(result.samples.len(), 10);
        assert!(result.mean > Duration::ZERO);
    }

    #[test]
    fn test_bench_result_stats() {
        let mut bench = Benchmark::new("stats_test")
            .with_warmup(0)
            .with_iterations(100);
        
        bench.run(|| {});
        
        let result = bench.finish();
        
        assert!(result.percentiles.contains_key(&50));
        assert!(result.percentiles.contains_key(&95));
        assert!(result.iterations_per_second() > 0.0);
    }

    #[test]
    fn test_bench_group() {
        let mut group = BenchGroup::new("test_group");
        
        group
            .bench("fast", || {})
            .bench("slow", || std::thread::sleep(Duration::from_micros(10)));
        
        assert_eq!(group.results().len(), 2);
        
        let summary = group.summary();
        assert_eq!(summary.benchmark_count, 2);
        assert!(summary.fastest.is_some());
        assert!(summary.slowest.is_some());
    }

    #[test]
    fn test_bench_comparison() {
        let mut bench1 = Benchmark::new("fast").with_iterations(50);
        bench1.run(|| {});
        let result1 = bench1.finish();

        let mut bench2 = Benchmark::new("slow").with_iterations(50);
        bench2.run(|| std::thread::sleep(Duration::from_micros(100)));
        let result2 = bench2.finish();

        let comparison = result2.compare(&result1);
        assert!(comparison.slower.is_some());
    }
}
