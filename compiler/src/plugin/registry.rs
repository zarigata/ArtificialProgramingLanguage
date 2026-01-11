// Plugin Registry
// Central registry for managing installed plugins

use super::*;

pub struct PluginRegistry {
    plugins: HashMap<String, PluginInfo>,
}

#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub path: PathBuf,
    pub enabled: bool,
}

impl PluginRegistry {
    pub fn new() -> Self {
        PluginRegistry {
            plugins: HashMap::new(),
        }
    }
    
    pub fn register(&mut self, info: PluginInfo) {
        self.plugins.insert(info.name.clone(), info);
    }
    
    pub fn get(&self, name: &str) -> Option<&PluginInfo> {
        self.plugins.get(name)
    }
    
    pub fn list(&self) -> Vec<&PluginInfo> {
        self.plugins.values().collect()
    }
}
