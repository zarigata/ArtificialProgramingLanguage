//! Package registry management for VPM

use std::collections::HashMap;
use std::path::PathBuf;

/// Package registry configuration
#[derive(Debug, Clone)]
pub struct Registry {
    pub name: String,
    pub url: String,
    pub registry_type: RegistryType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegistryType {
    /// Official VezHub registry
    Official,
    /// GitHub-based registry
    GitHub,
    /// Local filesystem registry
    Local(PathBuf),
    /// Custom HTTP registry
    Custom,
}

/// Registry manager handles multiple package registries
pub struct RegistryManager {
    registries: HashMap<String, Registry>,
    default_registry: String,
}

impl RegistryManager {
    /// Create a new registry manager with default registries
    pub fn new() -> Self {
        let mut registries = HashMap::new();
        
        // Add official registry
        registries.insert(
            "official".to_string(),
            Registry {
                name: "official".to_string(),
                url: "https://registry.vezhub.org".to_string(),
                registry_type: RegistryType::Official,
            },
        );
        
        // Add GitHub registry
        registries.insert(
            "github".to_string(),
            Registry {
                name: "github".to_string(),
                url: "https://github.com/vez-packages".to_string(),
                registry_type: RegistryType::GitHub,
            },
        );
        
        RegistryManager {
            registries,
            default_registry: "official".to_string(),
        }
    }
    
    /// Add a new registry
    pub fn add_registry(&mut self, registry: Registry) {
        self.registries.insert(registry.name.clone(), registry);
    }
    
    /// Remove a registry
    pub fn remove_registry(&mut self, name: &str) -> Option<Registry> {
        self.registries.remove(name)
    }
    
    /// Get a registry by name
    pub fn get_registry(&self, name: &str) -> Option<&Registry> {
        self.registries.get(name)
    }
    
    /// List all registries
    pub fn list_registries(&self) -> Vec<&Registry> {
        self.registries.values().collect()
    }
    
    /// Set default registry
    pub fn set_default(&mut self, name: String) {
        self.default_registry = name;
    }
    
    /// Get default registry
    pub fn get_default(&self) -> Option<&Registry> {
        self.registries.get(&self.default_registry)
    }
    
    /// Search for a package across all registries
    pub fn search_package(&self, query: &str) -> Vec<PackageSearchResult> {
        let mut results = Vec::new();
        
        for registry in self.registries.values() {
            // Simulate search (in real implementation, would query registry API)
            if let Some(result) = self.search_in_registry(registry, query) {
                results.push(result);
            }
        }
        
        results
    }
    
    fn search_in_registry(&self, registry: &Registry, query: &str) -> Option<PackageSearchResult> {
        // Placeholder for actual registry search
        // In real implementation, would make HTTP request to registry API
        Some(PackageSearchResult {
            name: query.to_string(),
            version: "1.0.0".to_string(),
            description: format!("Package {} from {}", query, registry.name),
            registry: registry.name.clone(),
            downloads: 0,
        })
    }
}

impl Default for RegistryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Package search result
#[derive(Debug, Clone)]
pub struct PackageSearchResult {
    pub name: String,
    pub version: String,
    pub description: String,
    pub registry: String,
    pub downloads: u64,
}

/// Package metadata
#[derive(Debug, Clone)]
pub struct PackageMetadata {
    pub name: String,
    pub version: String,
    pub authors: Vec<String>,
    pub description: String,
    pub license: String,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_manager_creation() {
        let manager = RegistryManager::new();
        assert_eq!(manager.registries.len(), 2);
        assert!(manager.get_registry("official").is_some());
        assert!(manager.get_registry("github").is_some());
    }

    #[test]
    fn test_add_registry() {
        let mut manager = RegistryManager::new();
        
        let custom_registry = Registry {
            name: "custom".to_string(),
            url: "https://custom.registry.com".to_string(),
            registry_type: RegistryType::Custom,
        };
        
        manager.add_registry(custom_registry);
        assert_eq!(manager.registries.len(), 3);
        assert!(manager.get_registry("custom").is_some());
    }

    #[test]
    fn test_remove_registry() {
        let mut manager = RegistryManager::new();
        
        let removed = manager.remove_registry("github");
        assert!(removed.is_some());
        assert_eq!(manager.registries.len(), 1);
        assert!(manager.get_registry("github").is_none());
    }

    #[test]
    fn test_default_registry() {
        let manager = RegistryManager::new();
        let default = manager.get_default();
        assert!(default.is_some());
        assert_eq!(default.unwrap().name, "official");
    }

    #[test]
    fn test_set_default_registry() {
        let mut manager = RegistryManager::new();
        manager.set_default("github".to_string());
        
        let default = manager.get_default();
        assert!(default.is_some());
        assert_eq!(default.unwrap().name, "github");
    }

    #[test]
    fn test_list_registries() {
        let manager = RegistryManager::new();
        let registries = manager.list_registries();
        assert_eq!(registries.len(), 2);
    }

    #[test]
    fn test_search_package() {
        let manager = RegistryManager::new();
        let results = manager.search_package("serde");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_local_registry() {
        let mut manager = RegistryManager::new();
        
        let local_registry = Registry {
            name: "local".to_string(),
            url: "file:///home/user/packages".to_string(),
            registry_type: RegistryType::Local(PathBuf::from("/home/user/packages")),
        };
        
        manager.add_registry(local_registry);
        
        let registry = manager.get_registry("local");
        assert!(registry.is_some());
        
        if let Some(reg) = registry {
            assert!(matches!(reg.registry_type, RegistryType::Local(_)));
        }
    }
}
