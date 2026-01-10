//! Target machine configuration

use crate::error::{Result, CompilerError};

/// Target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    AArch64,
    ARM,
    RISCV64,
}

/// Target operating system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OS {
    Linux,
    MacOS,
    Windows,
    FreeBSD,
}

/// Optimization level for code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenOptLevel {
    None,
    Less,
    Default,
    Aggressive,
}

/// Relocation model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocMode {
    Static,
    PIC,
    DynamicNoPic,
}

/// Code model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeModel {
    Small,
    Medium,
    Large,
}

/// Target machine configuration
pub struct TargetMachine {
    pub arch: Arch,
    pub os: OS,
    pub triple: String,
    pub cpu: String,
    pub features: Vec<String>,
    pub opt_level: CodegenOptLevel,
    pub reloc_mode: RelocMode,
    pub code_model: CodeModel,
}

impl TargetMachine {
    /// Create a target machine for the host
    pub fn host() -> Self {
        let (arch, os, triple) = Self::detect_host();
        
        TargetMachine {
            arch,
            os,
            triple,
            cpu: "generic".to_string(),
            features: Vec::new(),
            opt_level: CodegenOptLevel::Default,
            reloc_mode: RelocMode::PIC,
            code_model: CodeModel::Small,
        }
    }
    
    /// Create a target machine with custom configuration
    pub fn new(triple: String) -> Result<Self> {
        let (arch, os) = Self::parse_triple(&triple)?;
        
        Ok(TargetMachine {
            arch,
            os,
            triple,
            cpu: "generic".to_string(),
            features: Vec::new(),
            opt_level: CodegenOptLevel::Default,
            reloc_mode: RelocMode::PIC,
            code_model: CodeModel::Small,
        })
    }
    
    /// Detect the host target
    fn detect_host() -> (Arch, OS, String) {
        #[cfg(target_arch = "x86_64")]
        let arch = Arch::X86_64;
        
        #[cfg(target_arch = "aarch64")]
        let arch = Arch::AArch64;
        
        #[cfg(target_arch = "arm")]
        let arch = Arch::ARM;
        
        #[cfg(target_arch = "riscv64")]
        let arch = Arch::RISCV64;
        
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "riscv64"
        )))]
        let arch = Arch::X86_64; // Default
        
        #[cfg(target_os = "linux")]
        let os = OS::Linux;
        
        #[cfg(target_os = "macos")]
        let os = OS::MacOS;
        
        #[cfg(target_os = "windows")]
        let os = OS::Windows;
        
        #[cfg(target_os = "freebsd")]
        let os = OS::FreeBSD;
        
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows",
            target_os = "freebsd"
        )))]
        let os = OS::Linux; // Default
        
        let triple = Self::make_triple(arch, os);
        
        (arch, os, triple)
    }
    
    /// Parse a target triple
    fn parse_triple(triple: &str) -> Result<(Arch, OS)> {
        let parts: Vec<&str> = triple.split('-').collect();
        
        let arch = match parts.get(0) {
            Some(&"x86_64") => Arch::X86_64,
            Some(&"aarch64") => Arch::AArch64,
            Some(&"arm") => Arch::ARM,
            Some(&"riscv64") => Arch::RISCV64,
            _ => return Err(CompilerError::CodegenError(
                format!("Unknown architecture in triple: {}", triple)
            )),
        };
        
        let os = if triple.contains("linux") {
            OS::Linux
        } else if triple.contains("darwin") || triple.contains("macos") {
            OS::MacOS
        } else if triple.contains("windows") {
            OS::Windows
        } else if triple.contains("freebsd") {
            OS::FreeBSD
        } else {
            return Err(CompilerError::CodegenError(
                format!("Unknown OS in triple: {}", triple)
            ));
        };
        
        Ok((arch, os))
    }
    
    /// Make a target triple from arch and OS
    fn make_triple(arch: Arch, os: OS) -> String {
        let arch_str = match arch {
            Arch::X86_64 => "x86_64",
            Arch::AArch64 => "aarch64",
            Arch::ARM => "arm",
            Arch::RISCV64 => "riscv64",
        };
        
        match os {
            OS::Linux => format!("{}-unknown-linux-gnu", arch_str),
            OS::MacOS => format!("{}-apple-darwin", arch_str),
            OS::Windows => format!("{}-pc-windows-msvc", arch_str),
            OS::FreeBSD => format!("{}-unknown-freebsd", arch_str),
        }
    }
    
    /// Set the CPU type
    pub fn with_cpu(mut self, cpu: String) -> Self {
        self.cpu = cpu;
        self
    }
    
    /// Add CPU features
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }
    
    /// Set optimization level
    pub fn with_opt_level(mut self, level: CodegenOptLevel) -> Self {
        self.opt_level = level;
        self
    }
    
    /// Set relocation mode
    pub fn with_reloc_mode(mut self, mode: RelocMode) -> Self {
        self.reloc_mode = mode;
        self
    }
    
    /// Set code model
    pub fn with_code_model(mut self, model: CodeModel) -> Self {
        self.code_model = model;
        self
    }
    
    /// Get the object file extension for this target
    pub fn object_extension(&self) -> &str {
        match self.os {
            OS::Windows => "obj",
            _ => "o",
        }
    }
    
    /// Get the executable extension for this target
    pub fn executable_extension(&self) -> &str {
        match self.os {
            OS::Windows => "exe",
            _ => "",
        }
    }
    
    /// Get the dynamic library extension for this target
    pub fn dylib_extension(&self) -> &str {
        match self.os {
            OS::Linux | OS::FreeBSD => "so",
            OS::MacOS => "dylib",
            OS::Windows => "dll",
        }
    }
    
    /// Get the static library extension for this target
    pub fn staticlib_extension(&self) -> &str {
        match self.os {
            OS::Windows => "lib",
            _ => "a",
        }
    }
    
    /// Get the pointer size in bytes
    pub fn pointer_size(&self) -> usize {
        match self.arch {
            Arch::X86_64 | Arch::AArch64 | Arch::RISCV64 => 8,
            Arch::ARM => 4,
        }
    }
    
    /// Check if this is a 64-bit target
    pub fn is_64bit(&self) -> bool {
        self.pointer_size() == 8
    }
}

impl Default for TargetMachine {
    fn default() -> Self {
        Self::host()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_target() {
        let target = TargetMachine::host();
        assert!(!target.triple.is_empty());
        assert!(!target.cpu.is_empty());
    }

    #[test]
    fn test_parse_triple() {
        let result = TargetMachine::parse_triple("x86_64-unknown-linux-gnu");
        assert!(result.is_ok());
        let (arch, os) = result.unwrap();
        assert_eq!(arch, Arch::X86_64);
        assert_eq!(os, OS::Linux);
    }

    #[test]
    fn test_custom_target() {
        let target = TargetMachine::new("x86_64-unknown-linux-gnu".to_string());
        assert!(target.is_ok());
        let target = target.unwrap();
        assert_eq!(target.arch, Arch::X86_64);
        assert_eq!(target.os, OS::Linux);
    }

    #[test]
    fn test_with_cpu() {
        let target = TargetMachine::host()
            .with_cpu("native".to_string());
        assert_eq!(target.cpu, "native");
    }

    #[test]
    fn test_with_features() {
        let target = TargetMachine::host()
            .with_features(vec!["avx2".to_string(), "fma".to_string()]);
        assert_eq!(target.features.len(), 2);
    }

    #[test]
    fn test_extensions() {
        let linux_target = TargetMachine::new("x86_64-unknown-linux-gnu".to_string()).unwrap();
        assert_eq!(linux_target.object_extension(), "o");
        assert_eq!(linux_target.executable_extension(), "");
        assert_eq!(linux_target.dylib_extension(), "so");
        
        let windows_target = TargetMachine::new("x86_64-pc-windows-msvc".to_string()).unwrap();
        assert_eq!(windows_target.object_extension(), "obj");
        assert_eq!(windows_target.executable_extension(), "exe");
        assert_eq!(windows_target.dylib_extension(), "dll");
    }

    #[test]
    fn test_pointer_size() {
        let target_64 = TargetMachine::new("x86_64-unknown-linux-gnu".to_string()).unwrap();
        assert_eq!(target_64.pointer_size(), 8);
        assert!(target_64.is_64bit());
    }

    #[test]
    fn test_opt_level() {
        let target = TargetMachine::host()
            .with_opt_level(CodegenOptLevel::Aggressive);
        assert_eq!(target.opt_level, CodegenOptLevel::Aggressive);
    }
}
