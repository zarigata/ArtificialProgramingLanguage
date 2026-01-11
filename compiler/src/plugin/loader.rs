// Plugin Loader
// Dynamic loading of plugins from shared libraries

use super::*;
use std::path::{Path, PathBuf};
use std::fs;

// Plugin loader
pub struct PluginLoader {
    search_paths: Vec<PathBuf>,
    loaded_plugins: Vec<String>,
}

impl PluginLoader {
    pub fn new() -> Self {
        let mut search_paths = Vec::new();
        
        // Add default search paths
        if let Ok(home) = std::env::var("HOME") {
            search_paths.push(PathBuf::from(home).join(".vez/plugins"));
        }
        
        search_paths.push(PathBuf::from("/usr/local/lib/vez/plugins"));
        search_paths.push(PathBuf::from("./plugins"));
        
        PluginLoader {
            search_paths,
            loaded_plugins: Vec::new(),
        }
    }
    
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }
    
    pub fn discover_plugins(&self) -> Result<Vec<PluginDescriptor>> {
        let mut descriptors = Vec::new();
        
        for search_path in &self.search_paths {
            if !search_path.exists() {
                continue;
            }
            
            let entries = fs::read_dir(search_path)
                .map_err(|e| Error::new(format!("Failed to read directory: {}", e), Span::dummy()))?;
            
            for entry in entries.flatten() {
                let path = entry.path();
                
                // Look for plugin manifest files
                if path.is_dir() {
                    let manifest_path = path.join("plugin.toml");
                    if manifest_path.exists() {
                        if let Ok(descriptor) = self.load_descriptor(&manifest_path) {
                            descriptors.push(descriptor);
                        }
                    }
                }
            }
        }
        
        Ok(descriptors)
    }
    
    pub fn load_plugin(&mut self, descriptor: &PluginDescriptor) -> Result<Box<dyn Plugin>> {
        // Check if already loaded
        if self.loaded_plugins.contains(&descriptor.name) {
            return Err(Error::new(
                format!("Plugin already loaded: {}", descriptor.name),
                Span::dummy(),
            ));
        }
        
        // Load the plugin library
        let plugin = self.load_plugin_library(&descriptor.path)?;
        
        self.loaded_plugins.push(descriptor.name.clone());
        
        Ok(plugin)
    }
    
    fn load_descriptor(&self, path: &Path) -> Result<PluginDescriptor> {
        let content = fs::read_to_string(path)
            .map_err(|e| Error::new(format!("Failed to read manifest: {}", e), Span::dummy()))?;
        
        // Parse TOML manifest
        self.parse_manifest(&content, path.parent().unwrap())
    }
    
    fn parse_manifest(&self, content: &str, plugin_dir: &Path) -> Result<PluginDescriptor> {
        // Simplified TOML parsing
        // In production, use a proper TOML parser
        
        let mut name = String::new();
        let mut version = String::new();
        let mut author = String::new();
        let mut description = String::new();
        
        for line in content.lines() {
            if let Some(value) = line.strip_prefix("name = ") {
                name = value.trim_matches('"').to_string();
            } else if let Some(value) = line.strip_prefix("version = ") {
                version = value.trim_matches('"').to_string();
            } else if let Some(value) = line.strip_prefix("author = ") {
                author = value.trim_matches('"').to_string();
            } else if let Some(value) = line.strip_prefix("description = ") {
                description = value.trim_matches('"').to_string();
            }
        }
        
        Ok(PluginDescriptor {
            name,
            version,
            author,
            description,
            path: plugin_dir.to_path_buf(),
            library: plugin_dir.join("lib").join("plugin.so"),
        })
    }
    
    fn load_plugin_library(&self, path: &Path) -> Result<Box<dyn Plugin>> {
        // In production, use libloading or similar to dynamically load shared libraries
        // For now, return a placeholder
        
        Err(Error::new(
            "Dynamic plugin loading not yet implemented".to_string(),
            Span::dummy(),
        ))
    }
}

// Plugin descriptor
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub path: PathBuf,
    pub library: PathBuf,
}

// Plugin builder for creating plugins programmatically
pub struct PluginBuilder {
    metadata: PluginMetadata,
}

impl PluginBuilder {
    pub fn new(name: String) -> Self {
        PluginBuilder {
            metadata: PluginMetadata {
                name,
                version: "0.1.0".to_string(),
                author: String::new(),
                description: String::new(),
                dependencies: Vec::new(),
                api_version: "1.0".to_string(),
                capabilities: Vec::new(),
            },
        }
    }
    
    pub fn version(mut self, version: String) -> Self {
        self.metadata.version = version;
        self
    }
    
    pub fn author(mut self, author: String) -> Self {
        self.metadata.author = author;
        self
    }
    
    pub fn description(mut self, description: String) -> Self {
        self.metadata.description = description;
        self
    }
    
    pub fn add_dependency(mut self, name: String, version: String) -> Self {
        self.metadata.dependencies.push(PluginDependency { name, version });
        self
    }
    
    pub fn add_capability(mut self, capability: PluginCapability) -> Self {
        self.metadata.capabilities.push(capability);
        self
    }
    
    pub fn build_metadata(self) -> PluginMetadata {
        self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_loader() {
        let loader = PluginLoader::new();
        assert!(!loader.search_paths.is_empty());
    }
    
    #[test]
    fn test_plugin_builder() {
        let metadata = PluginBuilder::new("test_plugin".to_string())
            .version("1.0.0".to_string())
            .author("Test Author".to_string())
            .description("A test plugin".to_string())
            .add_capability(PluginCapability::Analysis)
            .build_metadata();
        
        assert_eq!(metadata.name, "test_plugin");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.capabilities.len(), 1);
    }
}
