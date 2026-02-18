//! Assertion library for the testing framework

use std::fmt::Debug;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Assertion {
    pub message: String,
    pub passed: bool,
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub location: Option<String>,
}

pub type AssertResult = Result<(), Assertion>;

impl Assertion {
    pub fn fail(message: &str) -> Self {
        Assertion {
            message: message.to_string(),
            passed: false,
            expected: None,
            actual: None,
            location: None,
        }
    }

    pub fn pass(message: &str) -> Self {
        Assertion {
            message: message.to_string(),
            passed: true,
            expected: None,
            actual: None,
            location: None,
        }
    }

    pub fn with_expected(mut self, expected: &str) -> Self {
        self.expected = Some(expected.to_string());
        self
    }

    pub fn with_actual(mut self, actual: &str) -> Self {
        self.actual = Some(actual.to_string());
        self
    }

    pub fn at(mut self, file: &str, line: u32) -> Self {
        self.location = Some(format!("{}:{}", file, line));
        self
    }
}

impl std::fmt::Display for Assertion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let (Some(exp), Some(act)) = (&self.expected, &self.actual) {
            write!(f, "\n  Expected: {}", exp)?;
            write!(f, "\n  Actual:   {}", act)?;
        }
        Ok(())
    }
}

impl std::error::Error for Assertion {}

pub fn assert_true(condition: bool) -> AssertResult {
    if condition {
        Ok(())
    } else {
        Err(Assertion::fail("Expected true, got false")
            .with_expected("true")
            .with_actual("false"))
    }
}

pub fn assert_false(condition: bool) -> AssertResult {
    if !condition {
        Ok(())
    } else {
        Err(Assertion::fail("Expected false, got true")
            .with_expected("false")
            .with_actual("true"))
    }
}

pub fn assert_eq<T: Debug + PartialEq>(left: &T, right: &T) -> AssertResult {
    if left == right {
        Ok(())
    } else {
        Err(Assertion::fail("Values are not equal")
            .with_expected(&format!("{:?}", right))
            .with_actual(&format!("{:?}", left)))
    }
}

pub fn assert_ne<T: Debug + PartialEq>(left: &T, right: &T) -> AssertResult {
    if left != right {
        Ok(())
    } else {
        Err(Assertion::fail("Values are equal but should be different")
            .with_expected(&format!("not {:?}", right))
            .with_actual(&format!("{:?}", left)))
    }
}

pub fn assert_gt<T: Debug + PartialOrd>(left: &T, right: &T) -> AssertResult {
    if left > right {
        Ok(())
    } else {
        Err(Assertion::fail("Left value is not greater than right")
            .with_expected(&format!("> {:?}", right))
            .with_actual(&format!("{:?}", left)))
    }
}

pub fn assert_ge<T: Debug + PartialOrd>(left: &T, right: &T) -> AssertResult {
    if left >= right {
        Ok(())
    } else {
        Err(Assertion::fail("Left value is not greater than or equal to right")
            .with_expected(&format!(">= {:?}", right))
            .with_actual(&format!("{:?}", left)))
    }
}

pub fn assert_lt<T: Debug + PartialOrd>(left: &T, right: &T) -> AssertResult {
    if left < right {
        Ok(())
    } else {
        Err(Assertion::fail("Left value is not less than right")
            .with_expected(&format!("< {:?}", right))
            .with_actual(&format!("{:?}", left)))
    }
}

pub fn assert_le<T: Debug + PartialOrd>(left: &T, right: &T) -> AssertResult {
    if left <= right {
        Ok(())
    } else {
        Err(Assertion::fail("Left value is not less than or equal to right")
            .with_expected(&format!("<= {:?}", right))
            .with_actual(&format!("{:?}", left)))
    }
}

pub fn assert_near(left: f64, right: f64, epsilon: f64) -> AssertResult {
    if (left - right).abs() < epsilon {
        Ok(())
    } else {
        Err(Assertion::fail(&format!("Values differ by more than {}", epsilon))
            .with_expected(&format!("{:.6} ± {:.6}", right, epsilon))
            .with_actual(&format!("{:.6}", left)))
    }
}

pub fn assert_contains<T: Debug + PartialEq>(collection: &[T], item: &T) -> AssertResult {
    if collection.contains(item) {
        Ok(())
    } else {
        Err(Assertion::fail("Collection does not contain item")
            .with_expected(&format!("collection containing {:?}", item))
            .with_actual(&format!("{:?}", collection)))
    }
}

pub fn assert_not_contains<T: Debug + PartialEq>(collection: &[T], item: &T) -> AssertResult {
    if !collection.contains(item) {
        Ok(())
    } else {
        Err(Assertion::fail("Collection should not contain item")
            .with_expected(&format!("collection without {:?}", item))
            .with_actual(&format!("{:?}", collection)))
    }
}

pub fn assert_empty<T>(collection: &[T]) -> AssertResult {
    if collection.is_empty() {
        Ok(())
    } else {
        Err(Assertion::fail("Collection should be empty")
            .with_expected("empty collection")
            .with_actual(&format!("collection with {} items", collection.len())))
    }
}

pub fn assert_not_empty<T>(collection: &[T]) -> AssertResult {
    if !collection.is_empty() {
        Ok(())
    } else {
        Err(Assertion::fail("Collection should not be empty")
            .with_expected("non-empty collection")
            .with_actual("empty collection"))
    }
}

pub fn assert_len<T>(collection: &[T], expected_len: usize) -> AssertResult {
    if collection.len() == expected_len {
        Ok(())
    } else {
        Err(Assertion::fail("Collection has wrong length")
            .with_expected(&format!("length {}", expected_len))
            .with_actual(&format!("length {}", collection.len())))
    }
}

pub fn assert_starts_with(s: &str, prefix: &str) -> AssertResult {
    if s.starts_with(prefix) {
        Ok(())
    } else {
        Err(Assertion::fail("String does not start with prefix")
            .with_expected(&format!("string starting with {:?}", prefix))
            .with_actual(&format!("{:?}", s)))
    }
}

pub fn assert_ends_with(s: &str, suffix: &str) -> AssertResult {
    if s.ends_with(suffix) {
        Ok(())
    } else {
        Err(Assertion::fail("String does not end with suffix")
            .with_expected(&format!("string ending with {:?}", suffix))
            .with_actual(&format!("{:?}", s)))
    }
}

pub fn assert_matches(s: &str, pattern: &str) -> AssertResult {
    if s.contains(pattern) {
        Ok(())
    } else {
        Err(Assertion::fail("String does not match pattern")
            .with_expected(&format!("string matching {:?}", pattern))
            .with_actual(&format!("{:?}", s)))
    }
}

pub fn assert_ok<T: Debug, E: Debug>(result: &Result<T, E>) -> AssertResult {
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(Assertion::fail("Expected Ok, got Err")
            .with_expected("Ok(_)")
            .with_actual(&format!("Err({:?})", e))),
    }
}

pub fn assert_err<T: Debug, E: Debug>(result: &Result<T, E>) -> AssertResult {
    match result {
        Err(_) => Ok(()),
        Ok(v) => Err(Assertion::fail("Expected Err, got Ok")
            .with_expected("Err(_)")
            .with_actual(&format!("Ok({:?})", v))),
    }
}

pub fn assert_some<T: Debug>(option: &Option<T>) -> AssertResult {
    match option {
        Some(_) => Ok(()),
        None => Err(Assertion::fail("Expected Some, got None")
            .with_expected("Some(_)")
            .with_actual("None")),
    }
}

pub fn assert_none<T: Debug>(option: &Option<T>) -> AssertResult {
    match option {
        None => Ok(()),
        Some(v) => Err(Assertion::fail("Expected None, got Some")
            .with_expected("None")
            .with_actual(&format!("Some({:?})", v))),
    }
}

pub fn assert_type<T: 'static, U: 'static>(value: &T) -> AssertResult {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>() {
        Ok(())
    } else {
        Err(Assertion::fail("Value has wrong type")
            .with_expected(std::any::type_name::<U>())
            .with_actual(std::any::type_name::<T>()))
    }
}

pub fn assert_panics<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> AssertResult {
    let result = std::panic::catch_unwind(f);
    match result {
        Err(_) => Ok(()),
        Ok(()) => Err(Assertion::fail("Expected panic, but function completed successfully")
            .with_expected("panic")
            .with_actual("normal return")),
    }
}

pub fn assert_no_panic<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> AssertResult {
    let result = std::panic::catch_unwind(f);
    match result {
        Ok(()) => Ok(()),
        Err(_) => Err(Assertion::fail("Function panicked unexpectedly")
            .with_expected("no panic")
            .with_actual("panic occurred")),
    }
}

pub struct FluentAssert<T> {
    value: T,
    negated: bool,
}

impl<T> FluentAssert<T> {
    pub fn new(value: T) -> Self {
        FluentAssert { value, negated: false }
    }

    pub fn not(mut self) -> Self {
        self.negated = !self.negated;
        self
    }

    pub fn is(self) -> Self {
        self
    }
}

impl<T: Debug + PartialEq> FluentAssert<T> {
    pub fn equal_to(self, other: &T) -> AssertResult {
        if self.negated {
            assert_ne(&self.value, other)
        } else {
            assert_eq(&self.value, other)
        }
    }
}

impl<T: Debug + PartialOrd> FluentAssert<T> {
    pub fn greater_than(self, other: &T) -> AssertResult {
        if self.negated {
            assert_le(&self.value, other)
        } else {
            assert_gt(&self.value, other)
        }
    }

    pub fn less_than(self, other: &T) -> AssertResult {
        if self.negated {
            assert_ge(&self.value, other)
        } else {
            assert_lt(&self.value, other)
        }
    }
}

impl<T: Debug> FluentAssert<Option<T>> {
    pub fn some(self) -> AssertResult {
        if self.negated {
            assert_none(&self.value)
        } else {
            assert_some(&self.value)
        }
    }
}

impl<T: Debug, E: Debug> FluentAssert<Result<T, E>> {
    pub fn ok(self) -> AssertResult {
        if self.negated {
            assert_err(&self.value)
        } else {
            assert_ok(&self.value)
        }
    }
}

pub fn expect<T>(value: T) -> FluentAssert<T> {
    FluentAssert::new(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_eq() {
        assert!(assert_eq(&1, &1).is_ok());
        assert!(assert_eq(&1, &2).is_err());
    }

    #[test]
    fn test_assert_near() {
        assert!(assert_near(1.0, 1.001, 0.01).is_ok());
        assert!(assert_near(1.0, 1.1, 0.01).is_err());
    }

    #[test]
    fn test_assert_contains() {
        let arr = [1, 2, 3];
        assert!(assert_contains(&arr, &2).is_ok());
        assert!(assert_contains(&arr, &4).is_err());
    }

    #[test]
    fn test_assert_some() {
        assert!(assert_some(&Some(1)).is_ok());
        assert!(assert_some(&None::<i32>).is_err());
    }

    #[test]
    fn test_fluent_assert() {
        assert!(expect(5).equal_to(&5).is_ok());
        assert!(expect(5).greater_than(&3).is_ok());
        assert!(expect(Some(1)).some().is_ok());
    }
}
