//! Profile-Guided Optimization (PGO) Support
//!
//! This module provides infrastructure for collecting and applying
//! runtime profile data to improve compiler optimizations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Profile data collected during instrumented execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileData {
    /// Function execution counts
    pub function_counts: HashMap<String, u64>,
    /// Basic block execution counts
    pub block_counts: HashMap<String, u64>,
    /// Edge execution counts (for branch prediction)
    pub edge_counts: HashMap<(String, String), u64>,
    /// Call site frequencies
    pub call_counts: HashMap<String, u64>,
    /// Loop iteration counts
    pub loop_counts: HashMap<String, LoopProfile>,
    /// Value profiles (for indirect calls, etc.)
    pub value_profiles: HashMap<String, ValueProfile>,
    /// Memory access patterns
    pub memory_patterns: HashMap<String, MemoryPattern>,
    /// Total samples collected
    pub total_samples: u64,
}

/// Loop execution profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopProfile {
    /// Loop identifier
    pub id: String,
    /// Total iterations across all invocations
    pub total_iterations: u64,
    /// Number of times the loop was entered
    pub invocations: u64,
    /// Average iterations per invocation
    pub avg_iterations: f64,
    /// Maximum iterations observed
    pub max_iterations: u64,
    /// Minimum iterations observed
    pub min_iterations: u64,
}

/// Value profile for indirect targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueProfile {
    /// Target identifier
    pub id: String,
    /// Value -> count mapping
    pub values: HashMap<String, u64>,
    /// Total samples
    pub total: u64,
}

impl ValueProfile {
    pub fn new(id: String) -> Self {
        ValueProfile {
            id,
            values: HashMap::new(),
            total: 0,
        }
    }
    
    pub fn record(&mut self, value: String) {
        *self.values.entry(value).or_insert(0) += 1;
        self.total += 1;
    }
    
    pub fn get_top_values(&self, n: usize) -> Vec<(&String, u64)> {
        let mut entries: Vec<_> = self.values.iter().map(|(k, v)| (k, *v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.into_iter().take(n).collect()
    }
}

/// Memory access pattern profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    /// Memory location identifier
    pub id: String,
    /// Access count
    pub accesses: u64,
    /// Cache misses (if available)
    pub cache_misses: Option<u64>,
    /// Sequential access ratio
    pub sequential_ratio: f64,
    /// Average stride between accesses
    pub avg_stride: f64,
}

/// PGO instrumentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentationConfig {
    /// Enable function instrumentation
    pub instrument_functions: bool,
    /// Enable basic block instrumentation
    pub instrument_blocks: bool,
    /// Enable edge instrumentation
    pub instrument_edges: bool,
    /// Enable value profiling
    pub profile_values: bool,
    /// Enable memory access profiling
    pub profile_memory: bool,
    /// Sampling frequency (0 = instrumentation, >0 = sample-based)
    pub sample_frequency: u32,
    /// Output file for profile data
    pub output_path: PathBuf,
}

impl Default for InstrumentationConfig {
    fn default() -> Self {
        InstrumentationConfig {
            instrument_functions: true,
            instrument_blocks: true,
            instrument_edges: true,
            profile_values: true,
            profile_memory: false,
            sample_frequency: 0,
            output_path: PathBuf::from("default.profdata"),
        }
    }
}

/// PGO optimizer that uses profile data
#[derive(Debug)]
pub struct PgoOptimizer {
    /// Loaded profile data
    profile: ProfileData,
    /// Optimization thresholds
    thresholds: PgoThresholds,
    /// Hot function threshold (percentage of total samples)
    hot_function_threshold: f64,
    /// Hot block threshold (percentage of function samples)
    hot_block_threshold: f64,
}

/// Thresholds for PGO decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PgoThresholds {
    /// Minimum samples to consider data reliable
    pub min_samples: u64,
    /// Hot code threshold (percentage)
    pub hot_threshold: f64,
    /// Cold code threshold (percentage)
    pub cold_threshold: f64,
    /// Inlining benefit multiplier for hot callsites
    pub inline_hot_multiplier: f64,
    /// Loop unroll multiplier for hot loops
    pub unroll_hot_multiplier: f64,
}

impl Default for PgoThresholds {
    fn default() -> Self {
        PgoThresholds {
            min_samples: 1000,
            hot_threshold: 90.0,
            cold_threshold: 1.0,
            inline_hot_multiplier: 2.0,
            unroll_hot_multiplier: 1.5,
        }
    }
}

impl PgoOptimizer {
    pub fn new(profile: ProfileData) -> Self {
        PgoOptimizer {
            profile,
            thresholds: PgoThresholds::default(),
            hot_function_threshold: 1.0,
            hot_block_threshold: 10.0,
        }
    }
    
    pub fn with_thresholds(mut self, thresholds: PgoThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }
    
    /// Check if function is considered "hot"
    pub fn is_hot_function(&self, name: &str) -> bool {
        let count = self.profile.function_counts.get(name).copied().unwrap_or(0);
        if self.profile.total_samples == 0 {
            return false;
        }
        let percentage = (count as f64 / self.profile.total_samples as f64) * 100.0;
        percentage >= self.hot_function_threshold
    }
    
    /// Check if basic block is considered "hot"
    pub fn is_hot_block(&self, block_id: &str) -> bool {
        let count = self.profile.block_counts.get(block_id).copied().unwrap_or(0);
        if self.profile.total_samples == 0 {
            return false;
        }
        let percentage = (count as f64 / self.profile.total_samples as f64) * 100.0;
        percentage >= self.hot_block_threshold
    }
    
    /// Get execution count for a function
    pub fn get_function_count(&self, name: &str) -> u64 {
        self.profile.function_counts.get(name).copied().unwrap_or(0)
    }
    
    /// Get execution count for a basic block
    pub fn get_block_count(&self, block_id: &str) -> u64 {
        self.profile.block_counts.get(block_id).copied().unwrap_or(0)
    }
    
    /// Get branch probability
    pub fn get_branch_probability(&self, from_block: &str, to_block: &str) -> f64 {
        let edge_count = self.profile.edge_counts
            .get(&(from_block.to_string(), to_block.to_string()))
            .copied()
            .unwrap_or(0);
        
        let total_outgoing: u64 = self.profile.edge_counts
            .keys()
            .filter(|(from, _)| from == from_block)
            .map(|(_, to)| {
                self.profile.edge_counts
                    .get(&(from_block.to_string(), to.clone()))
                    .copied()
                    .unwrap_or(0)
            })
            .sum();
        
        if total_outgoing == 0 {
            0.5 // Default probability
        } else {
            edge_count as f64 / total_outgoing as f64
        }
    }
    
    /// Get most likely call target for indirect call
    pub fn get_likely_call_target(&self, callsite: &str) -> Option<String> {
        self.profile.value_profiles
            .get(callsite)
            .and_then(|vp| {
                vp.get_top_values(1)
                    .first()
                    .map(|(target, _)| (*target).clone())
            })
    }
    
    /// Get loop iteration profile
    pub fn get_loop_profile(&self, loop_id: &str) -> Option<&LoopProfile> {
        self.profile.loop_counts.get(loop_id)
    }
    
    /// Should inline this callsite based on profile
    pub fn should_inline(&self, caller: &str, callee: &str) -> bool {
        let callsite_key = format!("{}->{}", caller, callee);
        let call_count = self.profile.call_counts.get(&callsite_key).copied().unwrap_or(0);
        
        // Hot callsites benefit more from inlining
        if self.is_hot_function(callee) && call_count > 0 {
            return true;
        }
        
        // Callee is small and called frequently
        let callee_count = self.get_function_count(callee);
        callee_count > self.thresholds.min_samples
    }
    
    /// Get recommended unroll factor for loop
    pub fn get_unroll_factor(&self, loop_id: &str) -> u32 {
        if let Some(loop_profile) = self.get_loop_profile(loop_id) {
            if loop_profile.avg_iterations > 0.0 {
                // Hot loops with predictable iteration counts benefit from unrolling
                let variance = loop_profile.max_iterations as f64 - loop_profile.min_iterations as f64;
                let avg = loop_profile.avg_iterations;
                
                // Low variance means predictable, good for unrolling
                if variance / avg < 0.1 {
                    let base_factor = if avg > 16.0 { 4 } else if avg > 8.0 { 2 } else { 1 };
                    
                    // Multiply if loop is hot
                    let hotness = loop_profile.invocations as f64 / self.profile.total_samples.max(1) as f64;
                    if hotness > self.thresholds.hot_threshold / 100.0 {
                        (base_factor as f64 * self.thresholds.unroll_hot_multiplier) as u32
                    } else {
                        base_factor
                    }
                } else {
                    1 // High variance, don't unroll
                }
            } else {
                1
            }
        } else {
            1 // No profile data
        }
    }
    
    /// Get optimization decisions based on profile
    pub fn get_optimization_decisions(&self) -> OptimizationDecisions {
        let mut decisions = OptimizationDecisions::default();
        
        // Identify hot functions
        for (name, count) in &self.profile.function_counts {
            if self.is_hot_function(name) {
                decisions.hot_functions.push(HotFunction {
                    name: name.clone(),
                    count: *count,
                    priority: (*count as f64 / self.profile.total_samples.max(1) as f64) * 100.0,
                });
            }
        }
        
        decisions.hot_functions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        
        // Identify cold functions
        for (name, count) in &self.profile.function_counts {
            if *count == 0 || (*count as f64 / self.profile.total_samples.max(1) as f64) * 100.0 < self.thresholds.cold_threshold {
                decisions.cold_functions.push(name.clone());
            }
        }
        
        // Recommend inlining
        for (callsite, count) in &self.profile.call_counts {
            let parts: Vec<&str> = callsite.split("->").collect();
            if parts.len() == 2 {
                let caller = parts[0];
                let callee = parts[1];
                if self.should_inline(caller, callee) {
                    decisions.inline_recommendations.push(InlineRecommendation {
                        caller: caller.to_string(),
                        callee: callee.to_string(),
                        call_count: *count,
                        benefit: self.thresholds.inline_hot_multiplier,
                    });
                }
            }
        }
        
        // Recommend loop optimizations
        for (loop_id, loop_profile) in &self.profile.loop_counts {
            let unroll_factor = self.get_unroll_factor(loop_id);
            if unroll_factor > 1 {
                decisions.loop_optimizations.push(LoopOptimization {
                    loop_id: loop_id.clone(),
                    unroll_factor,
                    avg_iterations: loop_profile.avg_iterations,
                    hot: self.is_hot_block(loop_id),
                });
            }
        }
        
        decisions
    }
}

/// Optimization decisions derived from profile data
#[derive(Debug, Default)]
pub struct OptimizationDecisions {
    /// Hot functions to prioritize
    pub hot_functions: Vec<HotFunction>,
    /// Cold functions for function layout
    pub cold_functions: Vec<String>,
    /// Inlining recommendations
    pub inline_recommendations: Vec<InlineRecommendation>,
    /// Loop optimization recommendations
    pub loop_optimizations: Vec<LoopOptimization>,
}

/// Hot function information
#[derive(Debug, Clone)]
pub struct HotFunction {
    pub name: String,
    pub count: u64,
    pub priority: f64,
}

/// Inlining recommendation
#[derive(Debug, Clone)]
pub struct InlineRecommendation {
    pub caller: String,
    pub callee: String,
    pub call_count: u64,
    pub benefit: f64,
}

/// Loop optimization recommendation
#[derive(Debug, Clone)]
pub struct LoopOptimization {
    pub loop_id: String,
    pub unroll_factor: u32,
    pub avg_iterations: f64,
    pub hot: bool,
}

/// Profile data file format
#[derive(Debug, Clone)]
pub enum ProfileFormat {
    /// Binary format (compact)
    Binary,
    /// Text format (human-readable)
    Text,
    /// JSON format
    Json,
}

impl ProfileFormat {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "profdata" | "prof" | "bin" => ProfileFormat::Binary,
            "json" => ProfileFormat::Json,
            _ => ProfileFormat::Text,
        }
    }
}

/// Profile data reader/writer
pub struct ProfileIO;

impl ProfileIO {
    /// Write profile data to file
    pub fn write(profile: &ProfileData, path: &Path, format: ProfileFormat) -> std::io::Result<()> {
        let content = match format {
            ProfileFormat::Json => {
                serde_json::to_string_pretty(profile).unwrap_or_default()
            }
            ProfileFormat::Text => {
                Self::to_text(profile)
            }
            ProfileFormat::Binary => {
                Self::to_binary(profile)
            }
        };
        
        std::fs::write(path, content)
    }
    
    /// Read profile data from file
    pub fn read(path: &Path) -> std::io::Result<ProfileData> {
        let content = std::fs::read_to_string(path)?;
        
        // Try to detect format
        let trimmed = content.trim();
        if trimmed.starts_with('{') {
            // JSON format
            serde_json::from_str(&content)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        } else if trimmed.starts_with("PGO") {
            // Binary/text format
            Self::from_text(&content)
        } else {
            // Default to text
            Self::from_text(&content)
        }
    }
    
    fn to_text(profile: &ProfileData) -> String {
        let mut output = String::new();
        output.push_str("PGO Profile Data v1.0\n\n");
        
        output.push_str("[Functions]\n");
        for (name, count) in &profile.function_counts {
            output.push_str(&format!("{}: {}\n", name, count));
        }
        
        output.push_str("\n[Blocks]\n");
        for (id, count) in &profile.block_counts {
            output.push_str(&format!("{}: {}\n", id, count));
        }
        
        output.push_str("\n[Edges]\n");
        for ((from, to), count) in &profile.edge_counts {
            output.push_str(&format!("{} -> {}: {}\n", from, to, count));
        }
        
        output.push_str("\n[Loops]\n");
        for (id, lp) in &profile.loop_counts {
            output.push_str(&format!("{}: avg={:.1}, max={}, min={}, invocations={}\n",
                id, lp.avg_iterations, lp.max_iterations, lp.min_iterations, lp.invocations));
        }
        
        output.push_str(&format!("\n[Summary]\nTotal samples: {}\n", profile.total_samples));
        
        output
    }
    
    fn from_text(content: &str) -> std::io::Result<ProfileData> {
        let mut profile = ProfileData::default();
        let mut section = "";
        
        for line in content.lines() {
            let line = line.trim();
            
            if line.starts_with('[') && line.ends_with(']') {
                section = &line[1..line.len()-1];
                continue;
            }
            
            if line.is_empty() || line.starts_with("PGO") {
                continue;
            }
            
            match section {
                "Functions" => {
                    if let Some((name, count)) = line.split_once(':') {
                        if let Ok(c) = count.trim().parse::<u64>() {
                            profile.function_counts.insert(name.trim().to_string(), c);
                        }
                    }
                }
                "Blocks" => {
                    if let Some((id, count)) = line.split_once(':') {
                        if let Ok(c) = count.trim().parse::<u64>() {
                            profile.block_counts.insert(id.trim().to_string(), c);
                        }
                    }
                }
                "Edges" => {
                    if let Some((edge, count)) = line.split_once(':') {
                        if let Some((from, to)) = edge.split_once("->") {
                            if let Ok(c) = count.trim().parse::<u64>() {
                                profile.edge_counts.insert(
                                    (from.trim().to_string(), to.trim().to_string()),
                                    c
                                );
                            }
                        }
                    }
                }
                "Loops" => {
                    if let Some((id, rest)) = line.split_once(':') {
                        let mut lp = LoopProfile {
                            id: id.trim().to_string(),
                            total_iterations: 0,
                            invocations: 0,
                            avg_iterations: 0.0,
                            max_iterations: 0,
                            min_iterations: 0,
                        };
                        
                        for part in rest.split(',') {
                            let part = part.trim();
                            if let Some((key, val)) = part.split_once('=') {
                                match key.trim() {
                                    "avg" => lp.avg_iterations = val.trim().parse().unwrap_or(0.0),
                                    "max" => lp.max_iterations = val.trim().parse().unwrap_or(0),
                                    "min" => lp.min_iterations = val.trim().parse().unwrap_or(0),
                                    "invocations" => lp.invocations = val.trim().parse().unwrap_or(0),
                                    _ => {}
                                }
                            }
                        }
                        
                        profile.loop_counts.insert(id.trim().to_string(), lp);
                    }
                }
                "Summary" => {
                    if line.starts_with("Total samples:") {
                        if let Ok(c) = line.split(':').nth(1).unwrap_or("0").trim().parse::<u64>() {
                            profile.total_samples = c;
                        }
                    }
                }
                _ => {}
            }
        }
        
        // Calculate total samples if not present
        if profile.total_samples == 0 {
            profile.total_samples = profile.function_counts.values().sum();
        }
        
        Ok(profile)
    }
    
    fn to_binary(_profile: &ProfileData) -> String {
        // Simplified binary-like format (actually just text for now)
        // A real implementation would use byte serialization
        Self::to_text(_profile)
    }
    
    /// Merge multiple profile data files
    pub fn merge(profiles: &[ProfileData]) -> ProfileData {
        let mut merged = ProfileData::default();
        
        for profile in profiles {
            for (name, count) in &profile.function_counts {
                *merged.function_counts.entry(name.clone()).or_insert(0) += count;
            }
            
            for (id, count) in &profile.block_counts {
                *merged.block_counts.entry(id.clone()).or_insert(0) += count;
            }
            
            for (edge, count) in &profile.edge_counts {
                *merged.edge_counts.entry(edge.clone()).or_insert(0) += count;
            }
            
            for (id, count) in &profile.call_counts {
                *merged.call_counts.entry(id.clone()).or_insert(0) += count;
            }
            
            merged.total_samples += profile.total_samples;
        }
        
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pgo_optimizer_hot_function() {
        let mut profile = ProfileData::default();
        profile.function_counts.insert("hot_func".to_string(), 10000);
        profile.function_counts.insert("cold_func".to_string(), 10);
        profile.total_samples = 10010;
        
        let optimizer = PgoOptimizer::new(profile);
        
        assert!(optimizer.is_hot_function("hot_func"));
        assert!(!optimizer.is_hot_function("cold_func"));
    }
    
    #[test]
    fn test_value_profile() {
        let mut vp = ValueProfile::new("test".to_string());
        vp.record("foo".to_string());
        vp.record("foo".to_string());
        vp.record("bar".to_string());
        
        let top = vp.get_top_values(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, &"foo".to_string());
        assert_eq!(top[0].1, 2);
    }
    
    #[test]
    fn test_branch_probability() {
        let mut profile = ProfileData::default();
        profile.edge_counts.insert(("A".to_string(), "B".to_string()), 80);
        profile.edge_counts.insert(("A".to_string(), "C".to_string()), 20);
        
        let optimizer = PgoOptimizer::new(profile);
        
        let prob_b = optimizer.get_branch_probability("A", "B");
        let prob_c = optimizer.get_branch_probability("A", "C");
        
        assert!((prob_b - 0.8).abs() < 0.01);
        assert!((prob_c - 0.2).abs() < 0.01);
    }
    
    #[test]
    fn test_profile_io_text() {
        let mut profile = ProfileData::default();
        profile.function_counts.insert("main".to_string(), 100);
        profile.total_samples = 100;
        
        let text = ProfileIO::to_text(&profile);
        let parsed = ProfileIO::from_text(&text).unwrap();
        
        assert_eq!(parsed.function_counts.get("main"), Some(&100));
    }
    
    #[test]
    fn test_merge_profiles() {
        let mut p1 = ProfileData::default();
        p1.function_counts.insert("f".to_string(), 100);
        p1.total_samples = 100;
        
        let mut p2 = ProfileData::default();
        p2.function_counts.insert("f".to_string(), 200);
        p2.total_samples = 200;
        
        let merged = ProfileIO::merge(&[p1, p2]);
        
        assert_eq!(merged.function_counts.get("f"), Some(&300));
        assert_eq!(merged.total_samples, 300);
    }
}
