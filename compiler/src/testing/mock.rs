//! Mocking support for testing

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct Mock {
    name: String,
    calls: Arc<Mutex<Vec<MockCall>>>,
    expectations: Arc<Mutex<Vec<MockExpectation>>>,
}

#[derive(Debug, Clone)]
pub struct MockCall {
    pub args: Vec<String>,
    pub result: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MockExpectation {
    pub args: Vec<String>,
    pub result: String,
    pub times: Option<usize>,
    pub times_called: usize,
}

impl Mock {
    pub fn new(name: &str) -> Self {
        Mock {
            name: name.to_string(),
            calls: Arc::new(Mutex::new(Vec::new())),
            expectations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn expect(&mut self, args: Vec<String>, result: &str) -> &mut Self {
        let mut expectations = self.expectations.lock().unwrap();
        expectations.push(MockExpectation {
            args,
            result: result.to_string(),
            times: None,
            times_called: 0,
        });
        drop(expectations);
        self
    }

    pub fn expect_times(&mut self, args: Vec<String>, result: &str, times: usize) -> &mut Self {
        let mut expectations = self.expectations.lock().unwrap();
        expectations.push(MockExpectation {
            args,
            result: result.to_string(),
            times: Some(times),
            times_called: 0,
        });
        drop(expectations);
        self
    }

    pub fn call(&self, args: Vec<String>) -> String {
        let result = {
            let mut expectations = self.expectations.lock().unwrap();
            let mut found = None;
            
            for exp in expectations.iter_mut() {
                if exp.args == args && (exp.times.is_none() || exp.times_called < exp.times.unwrap()) {
                    exp.times_called += 1;
                    found = Some(exp.result.clone());
                    break;
                }
            }
            
            let result = found.unwrap_or_else(|| format!("unexpected call with args: {:?}", args));
            drop(expectations);
            result
        };

        let mut calls = self.calls.lock().unwrap();
        calls.push(MockCall {
            args,
            result: Some(result.clone()),
        });

        result
    }

    pub fn verify(&self) -> Result<(), String> {
        let expectations = self.expectations.lock().unwrap();
        
        for exp in expectations.iter() {
            if let Some(times) = exp.times {
                if exp.times_called != times {
                    return Err(format!(
                        "Mock '{}': expected {:?} to be called {} times, but was called {} times",
                        self.name, exp.args, times, exp.times_called
                    ));
                }
            }
        }
        
        Ok(())
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    pub fn calls(&self) -> Vec<MockCall> {
        self.calls.lock().unwrap().clone()
    }

    pub fn reset(&mut self) {
        self.calls.lock().unwrap().clear();
        self.expectations.lock().unwrap().clear();
    }
}

pub struct MockFn {
    mock: Mock,
}

impl MockFn {
    pub fn new(name: &str) -> Self {
        MockFn {
            mock: Mock::new(name),
        }
    }

    pub fn returns(&mut self, value: &str) -> &mut Self {
        self.mock.expect(Vec::new(), value);
        self
    }

    pub fn with_args(&mut self, args: Vec<String>, value: &str) -> &mut Self {
        self.mock.expect(args, value);
        self
    }

    pub fn call(&self) -> String {
        self.mock.call(Vec::new())
    }

    pub fn call_with(&self, args: Vec<String>) -> String {
        self.mock.call(args)
    }

    pub fn verify(&self) -> Result<(), String> {
        self.mock.verify()
    }
}

pub struct MockRegistry {
    mocks: HashMap<String, Mock>,
}

impl MockRegistry {
    pub fn new() -> Self {
        MockRegistry {
            mocks: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str) -> &mut Mock {
        self.mocks.entry(name.to_string())
            .or_insert_with(|| Mock::new(name))
    }

    pub fn get(&self, name: &str) -> Option<&Mock> {
        self.mocks.get(name)
    }

    pub fn verify_all(&self) -> Result<(), Vec<String>> {
        let errors: Vec<String> = self.mocks.values()
            .filter_map(|m| m.verify().err())
            .collect();
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn reset_all(&mut self) {
        for mock in self.mocks.values_mut() {
            mock.reset();
        }
    }
}

impl Default for MockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! mock_fn {
    ($name:ident, $return_type:ty) => {
        pub fn $name() -> $return_type {
            unimplemented!("mock function")
        }
    };
}

#[macro_export]
macro_rules! when {
    ($mock:expr, $method:ident($($args:expr),*) => $result:expr) => {
        $mock.$method(vec![$(format!("{:?}", $args)),*], $result);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_basic() {
        let mut mock = Mock::new("test_fn");
        mock.expect(vec!["1".to_string()], "result1");
        mock.expect(vec!["2".to_string()], "result2");

        assert_eq!(mock.call(vec!["1".to_string()]), "result1");
        assert_eq!(mock.call(vec!["2".to_string()]), "result2");
        assert!(mock.verify().is_ok());
    }

    #[test]
    fn test_mock_times() {
        let mut mock = Mock::new("test_fn");
        mock.expect_times(vec!["x".to_string()], "ok", 2);

        mock.call(vec!["x".to_string()]);
        mock.call(vec!["x".to_string()]);
        
        assert!(mock.verify().is_ok());
    }

    #[test]
    fn test_mock_verify_failure() {
        let mut mock = Mock::new("test_fn");
        mock.expect_times(vec!["x".to_string()], "ok", 2);

        mock.call(vec!["x".to_string()]);
        
        assert!(mock.verify().is_err());
    }

    #[test]
    fn test_mock_registry() {
        let mut registry = MockRegistry::new();
        
        registry.register("fn1").expect(vec![], "a");
        registry.register("fn2").expect(vec![], "b");
        
        assert!(registry.verify_all().is_ok());
    }

    #[test]
    fn test_mock_fn() {
        let mut mock_fn = MockFn::new("get_value");
        mock_fn.returns("42");

        assert_eq!(mock_fn.call(), "42");
        assert!(mock_fn.verify().is_ok());
    }
}
