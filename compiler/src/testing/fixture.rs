//! Test fixtures and context management

use std::collections::HashMap;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::rc::Rc;

pub struct TestContext {
    pub name: String,
    fixtures: HashMap<TypeId, Box<dyn Any>>,
    setup_hooks: Vec<Box<dyn Fn()>>,
    teardown_hooks: Vec<Box<dyn Fn()>>,
    temp_paths: Vec<std::path::PathBuf>,
}

impl TestContext {
    pub fn new(name: &str) -> Self {
        TestContext {
            name: name.to_string(),
            fixtures: HashMap::new(),
            setup_hooks: Vec::new(),
            teardown_hooks: Vec::new(),
            temp_paths: Vec::new(),
        }
    }

    pub fn add_fixture<T: 'static>(&mut self, fixture: T) {
        self.fixtures.insert(TypeId::of::<T>(), Box::new(fixture));
    }

    pub fn get_fixture<T: 'static + Clone>(&self) -> Option<T> {
        self.fixtures
            .get(&TypeId::of::<T>())
            .and_then(|f| f.downcast_ref::<T>())
            .cloned()
    }

    pub fn get_fixture_ref<T: 'static>(&self) -> Option<&T> {
        self.fixtures
            .get(&TypeId::of::<T>())
            .and_then(|f| f.downcast_ref::<T>())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn create_temp_file(&mut self, content: &str) -> std::io::Result<std::path::PathBuf> {
        let temp_dir = std::env::temp_dir();
        let file_name = format!("vez_test_{}_{}.tmp", self.name, uuid::Uuid::new_v4());
        let path = temp_dir.join(file_name);
        
        std::fs::write(&path, content)?;
        self.temp_paths.push(path.clone());
        
        Ok(path)
    }

    pub fn create_temp_dir(&mut self) -> std::io::Result<std::path::PathBuf> {
        let temp_base = std::env::temp_dir();
        let dir_name = format!("vez_test_{}_{}", self.name, uuid::Uuid::new_v4());
        let path = temp_base.join(dir_name);
        
        std::fs::create_dir_all(&path)?;
        self.temp_paths.push(path.clone());
        
        Ok(path)
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        for path in &self.temp_paths {
            let _ = std::fs::remove_file(path);
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

pub trait Fixture: Clone {
    fn setup(&mut self) {}
    fn teardown(&mut self) {}
}

#[derive(Clone)]
pub struct DatabaseFixture {
    pub connection_string: String,
    pub tables: Vec<String>,
}

impl Fixture for DatabaseFixture {
    fn setup(&mut self) {
        // Initialize test database
    }

    fn teardown(&mut self) {
        // Clean up test database
    }
}

impl DatabaseFixture {
    pub fn new(connection_string: &str) -> Self {
        DatabaseFixture {
            connection_string: connection_string.to_string(),
            tables: Vec::new(),
        }
    }

    pub fn with_tables(mut self, tables: &[&str]) -> Self {
        self.tables = tables.iter().map(|s| s.to_string()).collect();
        self
    }
}

#[derive(Clone)]
pub struct HttpFixture {
    pub base_url: String,
    pub server: Option<String>,
}

impl Fixture for HttpFixture {
    fn setup(&mut self) {
        // Start mock HTTP server
    }

    fn teardown(&mut self) {
        // Stop mock HTTP server
    }
}

impl HttpFixture {
    pub fn new(base_url: &str) -> Self {
        HttpFixture {
            base_url: base_url.to_string(),
            server: None,
        }
    }
}

#[derive(Clone)]
pub struct FileFixture {
    pub root: std::path::PathBuf,
    pub files: HashMap<String, String>,
}

impl Fixture for FileFixture {
    fn setup(&mut self) {
        std::fs::create_dir_all(&self.root).ok();
        for (path, content) in &self.files {
            let full_path = self.root.join(path);
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&full_path, content).ok();
        }
    }

    fn teardown(&mut self) {
        std::fs::remove_dir_all(&self.root).ok();
    }
}

impl FileFixture {
    pub fn new(root: &std::path::Path) -> Self {
        FileFixture {
            root: root.to_path_buf(),
            files: HashMap::new(),
        }
    }

    pub fn with_file(mut self, path: &str, content: &str) -> Self {
        self.files.insert(path.to_string(), content.to_string());
        self
    }

    pub fn read_file(&self, path: &str) -> std::io::Result<String> {
        std::fs::read_to_string(self.root.join(path))
    }
}

pub struct FixtureScope<T: Fixture> {
    fixture: T,
}

impl<T: Fixture> FixtureScope<T> {
    pub fn new(mut fixture: T) -> Self {
        fixture.setup();
        FixtureScope { fixture }
    }

    pub fn get(&self) -> &T {
        &self.fixture
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.fixture
    }
}

impl<T: Fixture> Drop for FixtureScope<T> {
    fn drop(&mut self) {
        self.fixture.teardown();
    }
}

#[macro_export]
macro_rules! fixture {
    ($name:ident, $type:ty, $init:expr) => {
        let $name = $crate::testing::fixture::FixtureScope::new::<$type>($init);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_context_fixtures() {
        let mut ctx = TestContext::new("test");
        ctx.add_fixture(42i32);
        
        assert_eq!(ctx.get_fixture::<i32>(), Some(42));
        assert_eq!(ctx.get_fixture::<String>(), None);
    }

    #[test]
    fn test_database_fixture() {
        let mut fixture = DatabaseFixture::new("sqlite::memory:")
            .with_tables(&["users", "posts"]);
        
        fixture.setup();
        assert!(!fixture.tables.is_empty());
        fixture.teardown();
    }

    #[test]
    fn test_file_fixture() {
        let temp = std::env::temp_dir().join("vez_test_fixture");
        let mut fixture = FileFixture::new(&temp)
            .with_file("test.txt", "Hello, World!")
            .with_file("nested/data.json", "{}");
        
        fixture.setup();
        
        assert!(fixture.read_file("test.txt").is_ok());
        assert_eq!(fixture.read_file("test.txt").unwrap(), "Hello, World!");
        
        fixture.teardown();
    }
}
