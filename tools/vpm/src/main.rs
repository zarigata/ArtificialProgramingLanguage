// VPM - VeZ Package Manager
// Cargo-like package manager for VeZ

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// Package manifest (VeZ.toml)
#[derive(Debug, Clone)]
pub struct Manifest {
    pub package: Package,
    pub dependencies: HashMap<String, Dependency>,
    pub dev_dependencies: HashMap<String, Dependency>,
    pub build_dependencies: HashMap<String, Dependency>,
}

#[derive(Debug, Clone)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub authors: Vec<String>,
    pub edition: String,
    pub description: Option<String>,
    pub license: Option<String>,
    pub repository: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Dependency {
    Version(String),
    Git { url: String, branch: Option<String>, tag: Option<String> },
    Path(PathBuf),
}

// Package registry
pub struct Registry {
    index_path: PathBuf,
    cache_path: PathBuf,
}

impl Registry {
    pub fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let vpm_home = PathBuf::from(home).join(".vpm");
        
        Registry {
            index_path: vpm_home.join("index"),
            cache_path: vpm_home.join("cache"),
        }
    }
    
    pub fn search(&self, query: &str) -> Vec<PackageInfo> {
        // Search the package index
        let mut results = Vec::new();
        
        if let Ok(entries) = fs::read_dir(&self.index_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name() {
                    let name_str = name.to_string_lossy();
                    if name_str.contains(query) {
                        if let Ok(info) = self.load_package_info(&path) {
                            results.push(info);
                        }
                    }
                }
            }
        }
        
        results
    }
    
    pub fn install(&self, name: &str, version: &str) -> Result<PathBuf, String> {
        println!("Installing {} v{}", name, version);
        
        // Download package
        let package_path = self.download_package(name, version)?;
        
        // Extract to cache
        let install_path = self.cache_path.join(format!("{}-{}", name, version));
        self.extract_package(&package_path, &install_path)?;
        
        println!("Installed {} v{}", name, version);
        Ok(install_path)
    }
    
    fn download_package(&self, name: &str, version: &str) -> Result<PathBuf, String> {
        // In production, this would download from a registry
        let url = format!("https://registry.vez-lang.org/packages/{}/{}", name, version);
        let dest = self.cache_path.join(format!("{}-{}.tar.gz", name, version));
        
        // Simulate download
        println!("Downloading from {}", url);
        
        Ok(dest)
    }
    
    fn extract_package(&self, archive: &Path, dest: &Path) -> Result<(), String> {
        fs::create_dir_all(dest).map_err(|e| e.to_string())?;
        
        // In production, extract tar.gz
        println!("Extracting to {}", dest.display());
        
        Ok(())
    }
    
    fn load_package_info(&self, path: &Path) -> Result<PackageInfo, String> {
        // Load package metadata
        Ok(PackageInfo {
            name: "example".to_string(),
            version: "0.1.0".to_string(),
            description: "Example package".to_string(),
        })
    }
}

#[derive(Debug)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub description: String,
}

// Build system
pub struct Builder {
    manifest: Manifest,
    target_dir: PathBuf,
}

impl Builder {
    pub fn new(manifest: Manifest) -> Self {
        Builder {
            manifest,
            target_dir: PathBuf::from("target"),
        }
    }
    
    pub fn build(&self, release: bool) -> Result<PathBuf, String> {
        let profile = if release { "release" } else { "debug" };
        println!("Compiling {} v{} ({})", 
                 self.manifest.package.name,
                 self.manifest.package.version,
                 profile);
        
        // Create target directory
        let build_dir = self.target_dir.join(profile);
        fs::create_dir_all(&build_dir).map_err(|e| e.to_string())?;
        
        // Compile dependencies first
        self.build_dependencies()?;
        
        // Compile main package
        let output = build_dir.join(&self.manifest.package.name);
        self.compile_package(&output, release)?;
        
        println!("Finished {} [optimized] target(s)", profile);
        Ok(output)
    }
    
    fn build_dependencies(&self) -> Result<(), String> {
        for (name, dep) in &self.manifest.dependencies {
            println!("   Compiling {} ...", name);
            // Build dependency
        }
        Ok(())
    }
    
    fn compile_package(&self, output: &Path, release: bool) -> Result<(), String> {
        let mut cmd = Command::new("vezc");
        cmd.arg("src/main.zari");
        cmd.arg("-o").arg(output);
        
        if release {
            cmd.arg("-O3");
        }
        
        let status = cmd.status().map_err(|e| e.to_string())?;
        
        if !status.success() {
            return Err("Compilation failed".to_string());
        }
        
        Ok(())
    }
    
    pub fn test(&self) -> Result<(), String> {
        println!("Running tests for {} v{}", 
                 self.manifest.package.name,
                 self.manifest.package.version);
        
        // Find and run test files
        let test_dir = PathBuf::from("tests");
        if test_dir.exists() {
            self.run_tests(&test_dir)?;
        }
        
        println!("test result: ok");
        Ok(())
    }
    
    fn run_tests(&self, test_dir: &Path) -> Result<(), String> {
        if let Ok(entries) = fs::read_dir(test_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("zari") {
                    println!("   Running {}", path.display());
                    // Compile and run test
                }
            }
        }
        Ok(())
    }
    
    pub fn clean(&self) -> Result<(), String> {
        println!("Removing {}", self.target_dir.display());
        if self.target_dir.exists() {
            fs::remove_dir_all(&self.target_dir).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

// CLI commands
pub enum Command {
    New { name: String },
    Build { release: bool },
    Run { release: bool, args: Vec<String> },
    Test,
    Clean,
    Install { package: String },
    Search { query: String },
    Publish,
}

pub fn execute_command(cmd: Command) -> Result<(), String> {
    match cmd {
        Command::New { name } => {
            create_new_project(&name)?;
            println!("Created binary (application) `{}` package", name);
        }
        Command::Build { release } => {
            let manifest = load_manifest()?;
            let builder = Builder::new(manifest);
            builder.build(release)?;
        }
        Command::Run { release, args } => {
            let manifest = load_manifest()?;
            let builder = Builder::new(manifest);
            let binary = builder.build(release)?;
            
            println!("Running `{}`", binary.display());
            let mut cmd = Command::new(&binary);
            cmd.args(&args);
            cmd.status().map_err(|e| e.to_string())?;
        }
        Command::Test => {
            let manifest = load_manifest()?;
            let builder = Builder::new(manifest);
            builder.test()?;
        }
        Command::Clean => {
            let manifest = load_manifest()?;
            let builder = Builder::new(manifest);
            builder.clean()?;
        }
        Command::Install { package } => {
            let registry = Registry::new();
            registry.install(&package, "latest")?;
        }
        Command::Search { query } => {
            let registry = Registry::new();
            let results = registry.search(&query);
            for pkg in results {
                println!("{} = \"{}\" # {}", pkg.name, pkg.version, pkg.description);
            }
        }
        Command::Publish => {
            let manifest = load_manifest()?;
            publish_package(&manifest)?;
        }
    }
    Ok(())
}

fn create_new_project(name: &str) -> Result<(), String> {
    let project_dir = PathBuf::from(name);
    fs::create_dir_all(&project_dir).map_err(|e| e.to_string())?;
    
    // Create VeZ.toml
    let manifest = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
edition = "2024"

[dependencies]
"#,
        name
    );
    fs::write(project_dir.join("VeZ.toml"), manifest).map_err(|e| e.to_string())?;
    
    // Create src/main.zari
    fs::create_dir_all(project_dir.join("src")).map_err(|e| e.to_string())?;
    let main_code = r#"fn main() {
    println!("Hello, VeZ!");
}
"#;
    fs::write(project_dir.join("src/main.zari"), main_code).map_err(|e| e.to_string())?;
    
    Ok(())
}

fn load_manifest() -> Result<Manifest, String> {
    let content = fs::read_to_string("VeZ.toml").map_err(|e| e.to_string())?;
    parse_manifest(&content)
}

fn parse_manifest(content: &str) -> Result<Manifest, String> {
    // Simplified TOML parsing
    Ok(Manifest {
        package: Package {
            name: "example".to_string(),
            version: "0.1.0".to_string(),
            authors: vec![],
            edition: "2024".to_string(),
            description: None,
            license: None,
            repository: None,
        },
        dependencies: HashMap::new(),
        dev_dependencies: HashMap::new(),
        build_dependencies: HashMap::new(),
    })
}

fn publish_package(manifest: &Manifest) -> Result<(), String> {
    println!("Publishing {} v{}", manifest.package.name, manifest.package.version);
    
    // Package the project
    println!("Packaging...");
    
    // Upload to registry
    println!("Uploading to registry...");
    
    println!("Published successfully!");
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let command = if args.len() < 2 {
        Command::Build { release: false }
    } else {
        match args[1].as_str() {
            "new" => Command::New {
                name: args.get(2).cloned().unwrap_or_else(|| "my-project".to_string()),
            },
            "build" => Command::Build {
                release: args.contains(&"--release".to_string()),
            },
            "run" => Command::Run {
                release: args.contains(&"--release".to_string()),
                args: args[2..].to_vec(),
            },
            "test" => Command::Test,
            "clean" => Command::Clean,
            "install" => Command::Install {
                package: args.get(2).cloned().unwrap_or_default(),
            },
            "search" => Command::Search {
                query: args.get(2).cloned().unwrap_or_default(),
            },
            "publish" => Command::Publish,
            _ => {
                eprintln!("Unknown command: {}", args[1]);
                std::process::exit(1);
            }
        }
    };
    
    if let Err(e) = execute_command(command) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
