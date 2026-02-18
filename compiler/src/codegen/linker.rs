//! Linker integration for creating executables

use crate::error::{Result, Error, ErrorKind};
use crate::codegen::target::TargetMachine;
use std::path::PathBuf;
use std::process::Command;

/// Output type for linking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Executable binary
    Executable,
    /// Static library
    StaticLib,
    /// Dynamic library
    DynamicLib,
    /// Object file (no linking)
    Object,
}

/// Linker configuration
pub struct Linker {
    target: TargetMachine,
    output_type: OutputType,
    output_path: PathBuf,
    object_files: Vec<PathBuf>,
    libraries: Vec<String>,
    library_paths: Vec<PathBuf>,
    link_args: Vec<String>,
}

impl Linker {
    /// Create a new linker
    pub fn new(target: TargetMachine, output_type: OutputType, output_path: PathBuf) -> Self {
        Linker {
            target,
            output_type,
            output_path,
            object_files: Vec::new(),
            libraries: Vec::new(),
            library_paths: Vec::new(),
            link_args: Vec::new(),
        }
    }
    
    /// Add an object file to link
    pub fn add_object(&mut self, path: PathBuf) {
        self.object_files.push(path);
    }
    
    /// Add a library to link against
    pub fn add_library(&mut self, name: String) {
        self.libraries.push(name);
    }
    
    /// Add a library search path
    pub fn add_library_path(&mut self, path: PathBuf) {
        self.library_paths.push(path);
    }
    
    /// Add a custom linker argument
    pub fn add_link_arg(&mut self, arg: String) {
        self.link_args.push(arg);
    }
    
    /// Get the linker command for this target
    fn get_linker_command(&self) -> String {
        use crate::codegen::target::OS;
        
        match self.target.os {
            OS::Linux | OS::FreeBSD => "ld".to_string(),
            OS::MacOS => "ld".to_string(),
            OS::Windows => "link.exe".to_string(),
        }
    }
    
    /// Build the linker command line
    fn build_command(&self) -> Command {
        use crate::codegen::target::OS;
        
        let linker = self.get_linker_command();
        let mut cmd = Command::new(&linker);
        
        match self.target.os {
            OS::Linux | OS::FreeBSD => {
                self.build_unix_command(&mut cmd);
            }
            OS::MacOS => {
                self.build_macos_command(&mut cmd);
            }
            OS::Windows => {
                self.build_windows_command(&mut cmd);
            }
        }
        
        cmd
    }
    
    /// Build command for Unix-like systems
    fn build_unix_command(&self, cmd: &mut Command) {
        // Output file
        cmd.arg("-o");
        cmd.arg(&self.output_path);
        
        // Output type specific flags
        match self.output_type {
            OutputType::Executable => {
                // Default is executable
            }
            OutputType::StaticLib => {
                cmd.arg("-r");
            }
            OutputType::DynamicLib => {
                cmd.arg("-shared");
            }
            OutputType::Object => {
                // Single object, no linking
                return;
            }
        }
        
        // Object files
        for obj in &self.object_files {
            cmd.arg(obj);
        }
        
        // Library paths
        for path in &self.library_paths {
            cmd.arg(format!("-L{}", path.display()));
        }
        
        // Libraries
        for lib in &self.libraries {
            cmd.arg(format!("-l{}", lib));
        }
        
        // Custom arguments
        for arg in &self.link_args {
            cmd.arg(arg);
        }
        
        // Standard libraries
        if self.output_type == OutputType::Executable {
            cmd.arg("-lc"); // Link against libc
        }
    }
    
    /// Build command for macOS
    fn build_macos_command(&self, cmd: &mut Command) {
        // Output file
        cmd.arg("-o");
        cmd.arg(&self.output_path);
        
        // Output type specific flags
        match self.output_type {
            OutputType::Executable => {
                cmd.arg("-macosx_version_min");
                cmd.arg("10.15");
            }
            OutputType::StaticLib => {
                cmd.arg("-r");
            }
            OutputType::DynamicLib => {
                cmd.arg("-dylib");
            }
            OutputType::Object => {
                return;
            }
        }
        
        // Object files
        for obj in &self.object_files {
            cmd.arg(obj);
        }
        
        // Library paths
        for path in &self.library_paths {
            cmd.arg(format!("-L{}", path.display()));
        }
        
        // Libraries
        for lib in &self.libraries {
            cmd.arg(format!("-l{}", lib));
        }
        
        // Custom arguments
        for arg in &self.link_args {
            cmd.arg(arg);
        }
        
        // System libraries
        if self.output_type == OutputType::Executable {
            cmd.arg("-lSystem");
        }
    }
    
    /// Build command for Windows
    fn build_windows_command(&self, cmd: &mut Command) {
        // Output file
        cmd.arg(format!("/OUT:{}", self.output_path.display()));
        
        // Output type specific flags
        match self.output_type {
            OutputType::Executable => {
                cmd.arg("/SUBSYSTEM:CONSOLE");
            }
            OutputType::StaticLib => {
                cmd.arg("/LIB");
            }
            OutputType::DynamicLib => {
                cmd.arg("/DLL");
            }
            OutputType::Object => {
                return;
            }
        }
        
        // Object files
        for obj in &self.object_files {
            cmd.arg(obj);
        }
        
        // Library paths
        for path in &self.library_paths {
            cmd.arg(format!("/LIBPATH:{}", path.display()));
        }
        
        // Libraries
        for lib in &self.libraries {
            cmd.arg(format!("{}.lib", lib));
        }
        
        // Custom arguments
        for arg in &self.link_args {
            cmd.arg(arg);
        }
    }
    
    /// Run the linker
    pub fn link(&self) -> Result<()> {
        if self.output_type == OutputType::Object {
            // No linking needed for object files
            return Ok(());
        }
        
        if self.object_files.is_empty() {
            return Err(Error::new(ErrorKind::InternalError,
                "No object files to link".to_string()
            ));
        }
        
        let mut cmd = self.build_command();
        
        println!("Linking: {:?}", cmd);
        
        let output = cmd.output()
            .map_err(|e| Error::new(ErrorKind::InternalError,
                format!("Failed to run linker: {}", e)
            ))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::new(ErrorKind::InternalError,
                format!("Linker failed: {}", stderr)
            ));
        }
        
        println!("Successfully created: {}", self.output_path.display());
        Ok(())
    }
    
    /// Create an executable from object files
    pub fn link_executable(
        target: TargetMachine,
        object_files: Vec<PathBuf>,
        output_path: PathBuf,
    ) -> Result<()> {
        let mut linker = Linker::new(target, OutputType::Executable, output_path);
        
        for obj in object_files {
            linker.add_object(obj);
        }
        
        linker.link()
    }
    
    /// Create a static library from object files
    pub fn link_static_lib(
        target: TargetMachine,
        object_files: Vec<PathBuf>,
        output_path: PathBuf,
    ) -> Result<()> {
        let mut linker = Linker::new(target, OutputType::StaticLib, output_path);
        
        for obj in object_files {
            linker.add_object(obj);
        }
        
        linker.link()
    }
    
    /// Create a dynamic library from object files
    pub fn link_dynamic_lib(
        target: TargetMachine,
        object_files: Vec<PathBuf>,
        output_path: PathBuf,
    ) -> Result<()> {
        let mut linker = Linker::new(target, OutputType::DynamicLib, output_path);
        
        for obj in object_files {
            linker.add_object(obj);
        }
        
        linker.link()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linker_creation() {
        let target = TargetMachine::host();
        let output = PathBuf::from("test.out");
        let linker = Linker::new(target, OutputType::Executable, output.clone());
        
        assert_eq!(linker.output_type, OutputType::Executable);
        assert_eq!(linker.output_path, output);
        assert_eq!(linker.object_files.len(), 0);
    }

    #[test]
    fn test_add_object() {
        let target = TargetMachine::host();
        let output = PathBuf::from("test.out");
        let mut linker = Linker::new(target, OutputType::Executable, output);
        
        linker.add_object(PathBuf::from("test.o"));
        assert_eq!(linker.object_files.len(), 1);
    }

    #[test]
    fn test_add_library() {
        let target = TargetMachine::host();
        let output = PathBuf::from("test.out");
        let mut linker = Linker::new(target, OutputType::Executable, output);
        
        linker.add_library("m".to_string());
        assert_eq!(linker.libraries.len(), 1);
    }

    #[test]
    fn test_add_library_path() {
        let target = TargetMachine::host();
        let output = PathBuf::from("test.out");
        let mut linker = Linker::new(target, OutputType::Executable, output);
        
        linker.add_library_path(PathBuf::from("/usr/lib"));
        assert_eq!(linker.library_paths.len(), 1);
    }

    #[test]
    fn test_output_types() {
        assert_eq!(OutputType::Executable, OutputType::Executable);
        assert_ne!(OutputType::Executable, OutputType::StaticLib);
    }

    #[test]
    fn test_link_no_objects() {
        let target = TargetMachine::host();
        let output = PathBuf::from("test.out");
        let linker = Linker::new(target, OutputType::Executable, output);
        
        let result = linker.link();
        assert!(result.is_err());
    }

    #[test]
    fn test_object_output_no_link() {
        let target = TargetMachine::host();
        let output = PathBuf::from("test.o");
        let linker = Linker::new(target, OutputType::Object, output);
        
        // Object output should succeed without linking
        let result = linker.link();
        assert!(result.is_ok());
    }
}
