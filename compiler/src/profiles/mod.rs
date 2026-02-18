//! Build Profiles System
//!
//! Provides configurable build profiles for different development scenarios.

use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// A build profile with all configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProfile {
    /// Profile name
    pub name: String,
    /// Optimization level
    pub opt_level: OptLevel,
    /// Debug information
    pub debug_info: DebugInfo,
    /// Link-time optimization
    pub lto: LtoMode,
    /// Code generation units
    pub codegen_units: u32,
    /// Strip symbols
    pub strip: StripMode,
    /// Panic strategy
    pub panic: PanicStrategy,
    /// Overflow checks
    pub overflow_checks: bool,
    /// Debug assertions
    pub debug_assertions: bool,
    /// Target architecture
    pub target: Option<String>,
    /// CPU features to enable
    pub cpu_features: Vec<String>,
    /// Linker to use
    pub linker: Option<String>,
    /// Additional linker arguments
    pub linker_args: Vec<String>,
    /// Rpath settings
    pub rpath: bool,
    /// Profile-guided optimization data
    pub pgo_profile: Option<PathBuf>,
    /// Custom environment variables
    pub env: HashMap<String, String>,
    /// Feature flags
    pub features: Vec<String>,
    /// Default feature flags
    pub default_features: bool,
    /// Build dependencies in release mode
    pub release_deps: bool,
    /// Incremental compilation
    pub incremental: bool,
    /// Parallel jobs (0 = auto)
    pub jobs: u32,
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptLevel {
    /// No optimization
    #[serde(rename = "0")]
    O0,
    /// Basic optimization
    #[serde(rename = "1")]
    O1,
    /// Standard optimization
    #[serde(rename = "2")]
    O2,
    /// Aggressive optimization
    #[serde(rename = "3")]
    O3,
    /// Optimize for size
    #[serde(rename = "s")]
    Os,
    /// Optimize for size aggressively
    #[serde(rename = "z")]
    Oz,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::O0
    }
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::O0 => write!(f, "0"),
            OptLevel::O1 => write!(f, "1"),
            OptLevel::O2 => write!(f, "2"),
            OptLevel::O3 => write!(f, "3"),
            OptLevel::Os => write!(f, "s"),
            OptLevel::Oz => write!(f, "z"),
        }
    }
}

/// Debug information level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugInfo {
    /// No debug info
    None,
    /// Line tables only
    LineTables,
    /// Full debug info
    Full,
    /// Maximum debug info
    Maximum,
}

impl Default for DebugInfo {
    fn default() -> Self {
        DebugInfo::None
    }
}

impl std::fmt::Display for DebugInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DebugInfo::None => write!(f, "none"),
            DebugInfo::LineTables => write!(f, "line-tables-only"),
            DebugInfo::Full => write!(f, "full"),
            DebugInfo::Maximum => write!(f, "maximum"),
        }
    }
}

/// Link-time optimization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LtoMode {
    /// No LTO
    None,
    /// Thin LTO (faster)
    Thin,
    /// Full LTO (slower, better optimization)
    Full,
    /// Thin LTO for local crates only
    ThinLocal,
}

impl Default for LtoMode {
    fn default() -> Self {
        LtoMode::None
    }
}

impl std::fmt::Display for LtoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LtoMode::None => write!(f, "off"),
            LtoMode::Thin => write!(f, "thin"),
            LtoMode::Full => write!(f, "fat"),
            LtoMode::ThinLocal => write!(f, "thin-local"),
        }
    }
}

/// Symbol stripping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StripMode {
    /// No stripping
    None,
    /// Strip debug symbols
    Debug,
    /// Strip all symbols
    Symbols,
}

impl Default for StripMode {
    fn default() -> Self {
        StripMode::None
    }
}

/// Panic handling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanicStrategy {
    /// Unwind the stack
    Unwind,
    /// Abort immediately
    Abort,
}

impl Default for PanicStrategy {
    fn default() -> Self {
        PanicStrategy::Unwind
    }
}

impl std::fmt::Display for PanicStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PanicStrategy::Unwind => write!(f, "unwind"),
            PanicStrategy::Abort => write!(f, "abort"),
        }
    }
}

impl BuildProfile {
    /// Create a new profile with the given name
    pub fn new(name: impl Into<String>) -> Self {
        BuildProfile {
            name: name.into(),
            opt_level: OptLevel::O0,
            debug_info: DebugInfo::None,
            lto: LtoMode::None,
            codegen_units: 256,
            strip: StripMode::None,
            panic: PanicStrategy::Unwind,
            overflow_checks: false,
            debug_assertions: false,
            target: None,
            cpu_features: Vec::new(),
            linker: None,
            linker_args: Vec::new(),
            rpath: false,
            pgo_profile: None,
            env: HashMap::new(),
            features: Vec::new(),
            default_features: true,
            release_deps: false,
            incremental: false,
            jobs: 0,
        }
    }
    
    /// Create the default debug profile
    pub fn debug() -> Self {
        let mut profile = BuildProfile::new("debug");
        profile.opt_level = OptLevel::O0;
        profile.debug_info = DebugInfo::Full;
        profile.debug_assertions = true;
        profile.overflow_checks = true;
        profile.incremental = true;
        profile.codegen_units = 256;
        profile
    }
    
    /// Create the default release profile
    pub fn release() -> Self {
        let mut profile = BuildProfile::new("release");
        profile.opt_level = OptLevel::O3;
        profile.debug_info = DebugInfo::None;
        profile.lto = LtoMode::Full;
        profile.codegen_units = 1;
        profile.strip = StripMode::Symbols;
        profile
    }
    
    /// Create a profile optimized for profiling
    pub fn profiling() -> Self {
        let mut profile = BuildProfile::new("profiling");
        profile.opt_level = OptLevel::O3;
        profile.debug_info = DebugInfo::Full;
        profile.debug_assertions = false;
        profile.overflow_checks = false;
        profile.lto = LtoMode::Thin;
        profile.codegen_units = 4;
        profile
    }
    
    /// Create a profile for benchmarking
    pub fn bench() -> Self {
        let mut profile = BuildProfile::new("bench");
        profile.opt_level = OptLevel::O3;
        profile.debug_info = DebugInfo::LineTables;
        profile.lto = LtoMode::Full;
        profile.codegen_units = 1;
        profile.strip = StripMode::Debug;
        profile
    }
    
    /// Create a profile for size optimization
    pub fn min_size() -> Self {
        let mut profile = BuildProfile::new("min-size");
        profile.opt_level = OptLevel::Oz;
        profile.debug_info = DebugInfo::None;
        profile.lto = LtoMode::Full;
        profile.codegen_units = 1;
        profile.strip = StripMode::Symbols;
        profile.panic = PanicStrategy::Abort;
        profile
    }
    
    /// Create a profile for fast compilation
    pub fn fast_build() -> Self {
        let mut profile = BuildProfile::new("fast-build");
        profile.opt_level = OptLevel::O1;
        profile.debug_info = DebugInfo::LineTables;
        profile.lto = LtoMode::None;
        profile.codegen_units = 256;
        profile.incremental = true;
        profile
    }
    
    /// Create a profile for development with warnings
    pub fn dev() -> Self {
        let mut profile = BuildProfile::debug();
        profile.name = "dev".to_string();
        profile
    }
    
    /// Create a profile for testing
    pub fn test() -> Self {
        let mut profile = BuildProfile::debug();
        profile.name = "test".to_string();
        profile
    }
    
    /// Set optimization level
    pub fn with_opt_level(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }
    
    /// Set debug info level
    pub fn with_debug_info(mut self, info: DebugInfo) -> Self {
        self.debug_info = info;
        self
    }
    
    /// Set LTO mode
    pub fn with_lto(mut self, lto: LtoMode) -> Self {
        self.lto = lto;
        self
    }
    
    /// Set codegen units
    pub fn with_codegen_units(mut self, units: u32) -> Self {
        self.codegen_units = units;
        self
    }
    
    /// Enable/disable stripping
    pub fn with_strip(mut self, strip: StripMode) -> Self {
        self.strip = strip;
        self
    }
    
    /// Set panic strategy
    pub fn with_panic(mut self, panic: PanicStrategy) -> Self {
        self.panic = panic;
        self
    }
    
    /// Set target
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }
    
    /// Add CPU feature
    pub fn with_cpu_feature(mut self, feature: impl Into<String>) -> Self {
        self.cpu_features.push(feature.into());
        self
    }
    
    /// Add feature flag
    pub fn with_feature(mut self, feature: impl Into<String>) -> Self {
        self.features.push(feature.into());
        self
    }
    
    /// Set PGO profile
    pub fn with_pgo(mut self, profile: PathBuf) -> Self {
        self.pgo_profile = Some(profile);
        self
    }
    
    /// Set environment variable
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
    
    /// Check if this is a release-like profile
    pub fn is_release(&self) -> bool {
        matches!(self.opt_level, OptLevel::O2 | OptLevel::O3 | OptLevel::Os | OptLevel::Oz)
    }
    
    /// Check if debug info is enabled
    pub fn has_debug_info(&self) -> bool {
        !matches!(self.debug_info, DebugInfo::None)
    }
    
    /// Convert to command-line arguments
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        
        args.push(format!("--opt-level={}", self.opt_level));
        args.push(format!("--debug-info={}", self.debug_info));
        
        if self.lto != LtoMode::None {
            args.push(format!("--lto={}", self.lto));
        }
        
        args.push(format!("--codegen-units={}", self.codegen_units));
        
        if self.strip != StripMode::None {
            args.push(format!("--strip={}", match self.strip {
                StripMode::None => "none",
                StripMode::Debug => "debug",
                StripMode::Symbols => "symbols",
            }));
        }
        
        if self.panic == PanicStrategy::Abort {
            args.push("--panic=abort".to_string());
        }
        
        if let Some(target) = &self.target {
            args.push(format!("--target={}", target));
        }
        
        for feature in &self.cpu_features {
            args.push(format!("-C target-cpu={}", feature));
        }
        
        if let Some(pgo) = &self.pgo_profile {
            args.push(format!("--profile-use={}", pgo.display()));
        }
        
        for feature in &self.features {
            args.push(format!("--features={}", feature));
        }
        
        if !self.default_features {
            args.push("--no-default-features".to_string());
        }
        
        args
    }
}

/// Build profile manager
#[derive(Debug, Clone, Default)]
pub struct ProfileManager {
    profiles: HashMap<String, BuildProfile>,
    active_profile: Option<String>,
}

impl ProfileManager {
    pub fn new() -> Self {
        let mut manager = ProfileManager {
            profiles: HashMap::new(),
            active_profile: None,
        };
        
        manager.register_defaults();
        manager
    }
    
    fn register_defaults(&mut self) {
        self.register(BuildProfile::debug());
        self.register(BuildProfile::release());
        self.register(BuildProfile::profiling());
        self.register(BuildProfile::bench());
        self.register(BuildProfile::min_size());
        self.register(BuildProfile::fast_build());
        self.register(BuildProfile::dev());
        self.register(BuildProfile::test());
    }
    
    /// Register a new profile
    pub fn register(&mut self, profile: BuildProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }
    
    /// Get a profile by name
    pub fn get(&self, name: &str) -> Option<&BuildProfile> {
        self.profiles.get(name)
    }
    
    /// Get a mutable reference to a profile
    pub fn get_mut(&mut self, name: &str) -> Option<&mut BuildProfile> {
        self.profiles.get_mut(name)
    }
    
    /// Set the active profile
    pub fn set_active(&mut self, name: &str) -> bool {
        if self.profiles.contains_key(name) {
            self.active_profile = Some(name.to_string());
            true
        } else {
            false
        }
    }
    
    /// Get the active profile
    pub fn active(&self) -> Option<&BuildProfile> {
        self.active_profile.as_ref().and_then(|name| self.profiles.get(name))
    }
    
    /// List all profile names
    pub fn list_profiles(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }
    
    /// Remove a profile
    pub fn remove(&mut self, name: &str) -> Option<BuildProfile> {
        self.profiles.remove(name)
    }
    
    /// Create a derived profile
    pub fn derive(&self, base: &str, new_name: &str) -> Option<BuildProfile> {
        self.profiles.get(base).map(|p| {
            let mut derived = p.clone();
            derived.name = new_name.to_string();
            derived
        })
    }
    
    /// Load profiles from a config file
    pub fn load_from_file(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let content = std::fs::read_to_string(path)?;
        
        let config: ProfileConfig = toml::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        for (name, profile) in config.profiles {
            self.profiles.insert(name, profile);
        }
        
        Ok(())
    }
    
    /// Save profiles to a config file
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let config = ProfileConfig {
            profiles: self.profiles.clone(),
        };
        
        let content = toml::to_string_pretty(&config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        std::fs::write(path, content)
    }
}

/// Profile configuration file format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileConfig {
    profiles: HashMap<String, BuildProfile>,
}

/// Build configuration combining profile and other settings
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Build profile
    pub profile: BuildProfile,
    /// Source directory
    pub src_dir: PathBuf,
    /// Output directory
    pub out_dir: PathBuf,
    /// Output file name
    pub output_name: String,
    /// Binary type
    pub bin_type: BinaryType,
    /// Verbose output
    pub verbose: bool,
    /// Quiet output
    pub quiet: bool,
    /// Dry run
    pub dry_run: bool,
    /// Watch for changes
    pub watch: bool,
}

/// Binary output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryType {
    /// Executable
    Executable,
    /// Static library
    StaticLib,
    /// Dynamic library
    DynamicLib,
    /// Object file
    Object,
    /// LLVM IR
    LlvmIr,
    /// Assembly
    Assembly,
}

impl Default for BinaryType {
    fn default() -> Self {
        BinaryType::Executable
    }
}

impl BuildConfig {
    pub fn new(profile: BuildProfile) -> Self {
        BuildConfig {
            profile,
            src_dir: PathBuf::from("src"),
            out_dir: PathBuf::from("target"),
            output_name: "a.out".to_string(),
            bin_type: BinaryType::Executable,
            verbose: false,
            quiet: false,
            dry_run: false,
            watch: false,
        }
    }
    
    pub fn with_src_dir(mut self, dir: PathBuf) -> Self {
        self.src_dir = dir;
        self
    }
    
    pub fn with_out_dir(mut self, dir: PathBuf) -> Self {
        self.out_dir = dir;
        self
    }
    
    pub fn with_output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }
    
    pub fn with_bin_type(mut self, bin_type: BinaryType) -> Self {
        self.bin_type = bin_type;
        self
    }
    
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
    
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_profiles() {
        let manager = ProfileManager::new();
        
        assert!(manager.get("debug").is_some());
        assert!(manager.get("release").is_some());
        assert!(manager.get("profiling").is_some());
    }
    
    #[test]
    fn test_debug_profile() {
        let profile = BuildProfile::debug();
        
        assert_eq!(profile.opt_level, OptLevel::O0);
        assert_eq!(profile.debug_info, DebugInfo::Full);
        assert!(profile.debug_assertions);
        assert!(profile.overflow_checks);
    }
    
    #[test]
    fn test_release_profile() {
        let profile = BuildProfile::release();
        
        assert_eq!(profile.opt_level, OptLevel::O3);
        assert_eq!(profile.debug_info, DebugInfo::None);
        assert_eq!(profile.lto, LtoMode::Full);
        assert_eq!(profile.codegen_units, 1);
    }
    
    #[test]
    fn test_profile_builder() {
        let profile = BuildProfile::new("custom")
            .with_opt_level(OptLevel::O2)
            .with_debug_info(DebugInfo::LineTables)
            .with_lto(LtoMode::Thin)
            .with_feature("simd");
        
        assert_eq!(profile.opt_level, OptLevel::O2);
        assert_eq!(profile.debug_info, DebugInfo::LineTables);
        assert!(profile.features.contains(&"simd".to_string()));
    }
    
    #[test]
    fn test_profile_args() {
        let profile = BuildProfile::release();
        let args = profile.to_args();
        
        assert!(args.iter().any(|a| a.contains("--opt-level=3")));
        assert!(args.iter().any(|a| a.contains("--lto=")));
    }
    
    #[test]
    fn test_profile_derive() {
        let manager = ProfileManager::new();
        let derived = manager.derive("release", "my-release");
        
        assert!(derived.is_some());
        let derived = derived.unwrap();
        assert_eq!(derived.name, "my-release");
        assert_eq!(derived.opt_level, OptLevel::O3);
    }
}
