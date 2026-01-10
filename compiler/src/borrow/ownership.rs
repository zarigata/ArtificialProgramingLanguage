//! Ownership and move tracking

use std::collections::{HashMap, HashSet};
use crate::error::{Error, ErrorKind, Result};
use crate::parser::Expr;

/// Value state in ownership system
#[derive(Debug, Clone, PartialEq)]
pub enum ValueState {
    /// Value is owned and available
    Owned,
    /// Value has been moved
    Moved,
    /// Value is borrowed immutably
    BorrowedShared,
    /// Value is borrowed mutably
    BorrowedMut,
    /// Value is partially moved (for structs/tuples)
    PartiallyMoved(Vec<String>),
}

/// Ownership tracker for variables
#[derive(Debug, Clone)]
pub struct OwnershipTracker {
    /// Variable states
    states: HashMap<String, ValueState>,
    /// Move history for error reporting
    move_history: HashMap<String, String>,
}

impl OwnershipTracker {
    pub fn new() -> Self {
        OwnershipTracker {
            states: HashMap::new(),
            move_history: HashMap::new(),
        }
    }
    
    /// Register a new variable as owned
    pub fn register(&mut self, name: String) {
        self.states.insert(name, ValueState::Owned);
    }
    
    /// Mark a variable as moved
    pub fn mark_moved(&mut self, name: String, location: String) -> Result<()> {
        match self.states.get(&name) {
            Some(ValueState::Owned) => {
                self.states.insert(name.clone(), ValueState::Moved);
                self.move_history.insert(name, location);
                Ok(())
            }
            Some(ValueState::Moved) => {
                let prev_location = self.move_history.get(&name)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Use of moved value '{}' (previously moved at {})", name, prev_location)
                ))
            }
            Some(ValueState::BorrowedShared) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot move '{}' while it is borrowed", name)
                ))
            }
            Some(ValueState::BorrowedMut) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot move '{}' while it is mutably borrowed", name)
                ))
            }
            Some(ValueState::PartiallyMoved(fields)) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot move '{}' because fields {:?} are already moved", name, fields)
                ))
            }
            None => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Unknown variable '{}'", name)
                ))
            }
        }
    }
    
    /// Mark a variable as borrowed immutably
    pub fn mark_borrowed_shared(&mut self, name: &str) -> Result<()> {
        match self.states.get(name) {
            Some(ValueState::Owned) | Some(ValueState::BorrowedShared) => {
                self.states.insert(name.to_string(), ValueState::BorrowedShared);
                Ok(())
            }
            Some(ValueState::Moved) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot borrow moved value '{}'", name)
                ))
            }
            Some(ValueState::BorrowedMut) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot borrow '{}' as shared because it is already mutably borrowed", name)
                ))
            }
            Some(ValueState::PartiallyMoved(_)) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot borrow partially moved value '{}'", name)
                ))
            }
            None => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Unknown variable '{}'", name)
                ))
            }
        }
    }
    
    /// Mark a variable as borrowed mutably
    pub fn mark_borrowed_mut(&mut self, name: &str) -> Result<()> {
        match self.states.get(name) {
            Some(ValueState::Owned) => {
                self.states.insert(name.to_string(), ValueState::BorrowedMut);
                Ok(())
            }
            Some(ValueState::Moved) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot borrow moved value '{}'", name)
                ))
            }
            Some(ValueState::BorrowedShared) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot borrow '{}' as mutable because it is already borrowed as shared", name)
                ))
            }
            Some(ValueState::BorrowedMut) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot borrow '{}' as mutable more than once", name)
                ))
            }
            Some(ValueState::PartiallyMoved(_)) => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Cannot borrow partially moved value '{}'", name)
                ))
            }
            None => {
                Err(Error::new(
                    ErrorKind::BorrowError,
                    format!("Unknown variable '{}'", name)
                ))
            }
        }
    }
    
    /// Release a borrow
    pub fn release_borrow(&mut self, name: &str) {
        if let Some(state) = self.states.get(name) {
            match state {
                ValueState::BorrowedShared | ValueState::BorrowedMut => {
                    self.states.insert(name.to_string(), ValueState::Owned);
                }
                _ => {}
            }
        }
    }
    
    /// Check if a variable is available (not moved)
    pub fn is_available(&self, name: &str) -> bool {
        matches!(
            self.states.get(name),
            Some(ValueState::Owned) | Some(ValueState::BorrowedShared) | Some(ValueState::BorrowedMut)
        )
    }
    
    /// Get the state of a variable
    pub fn get_state(&self, name: &str) -> Option<&ValueState> {
        self.states.get(name)
    }
}

impl Default for OwnershipTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Move checker for expressions
pub struct MoveChecker {
    /// Types that implement Copy trait
    copy_types: HashSet<String>,
}

impl MoveChecker {
    pub fn new() -> Self {
        let mut copy_types = HashSet::new();
        
        // Primitive types that implement Copy
        copy_types.insert("i8".to_string());
        copy_types.insert("i16".to_string());
        copy_types.insert("i32".to_string());
        copy_types.insert("i64".to_string());
        copy_types.insert("i128".to_string());
        copy_types.insert("u8".to_string());
        copy_types.insert("u16".to_string());
        copy_types.insert("u32".to_string());
        copy_types.insert("u64".to_string());
        copy_types.insert("u128".to_string());
        copy_types.insert("f32".to_string());
        copy_types.insert("f64".to_string());
        copy_types.insert("bool".to_string());
        copy_types.insert("char".to_string());
        
        MoveChecker { copy_types }
    }
    
    /// Check if a type implements Copy
    pub fn is_copy_type(&self, type_name: &str) -> bool {
        self.copy_types.contains(type_name)
    }
    
    /// Register a type as Copy
    pub fn register_copy_type(&mut self, type_name: String) {
        self.copy_types.insert(type_name);
    }
    
    /// Check if an expression causes a move
    pub fn is_move(&self, expr: &Expr, type_name: &str) -> bool {
        // References never move
        if type_name.starts_with('&') {
            return false;
        }
        
        // Copy types don't move
        if self.is_copy_type(type_name) {
            return false;
        }
        
        // Check expression kind
        match expr {
            Expr::Ident(_) => true,  // Using a variable moves it (unless Copy)
            Expr::Field(_, _) => true,  // Field access can move
            _ => false,  // Other expressions don't directly move
        }
    }
}

impl Default for MoveChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ownership_register() {
        let mut tracker = OwnershipTracker::new();
        tracker.register("x".to_string());
        assert!(tracker.is_available("x"));
    }

    #[test]
    fn test_move_tracking() {
        let mut tracker = OwnershipTracker::new();
        tracker.register("x".to_string());
        
        tracker.mark_moved("x".to_string(), "line 1".to_string()).unwrap();
        assert!(!tracker.is_available("x"));
        
        let result = tracker.mark_moved("x".to_string(), "line 2".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_borrow() {
        let mut tracker = OwnershipTracker::new();
        tracker.register("x".to_string());
        
        tracker.mark_borrowed_shared("x").unwrap();
        assert!(tracker.is_available("x"));
        
        // Can have multiple shared borrows
        tracker.mark_borrowed_shared("x").unwrap();
    }

    #[test]
    fn test_mutable_borrow() {
        let mut tracker = OwnershipTracker::new();
        tracker.register("x".to_string());
        
        tracker.mark_borrowed_mut("x").unwrap();
        assert!(tracker.is_available("x"));
        
        // Cannot have multiple mutable borrows
        let result = tracker.mark_borrowed_mut("x");
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_and_mutable_conflict() {
        let mut tracker = OwnershipTracker::new();
        tracker.register("x".to_string());
        
        tracker.mark_borrowed_shared("x").unwrap();
        
        // Cannot borrow mutably while shared borrow exists
        let result = tracker.mark_borrowed_mut("x");
        assert!(result.is_err());
    }

    #[test]
    fn test_borrow_release() {
        let mut tracker = OwnershipTracker::new();
        tracker.register("x".to_string());
        
        tracker.mark_borrowed_mut("x").unwrap();
        tracker.release_borrow("x");
        
        // Can borrow again after release
        tracker.mark_borrowed_mut("x").unwrap();
    }

    #[test]
    fn test_copy_types() {
        let checker = MoveChecker::new();
        assert!(checker.is_copy_type("i32"));
        assert!(checker.is_copy_type("bool"));
        assert!(!checker.is_copy_type("String"));
    }

    #[test]
    fn test_move_detection() {
        let checker = MoveChecker::new();
        let expr = Expr::Ident("x".to_string());
        
        // Non-Copy type moves
        assert!(checker.is_move(&expr, "String"));
        
        // Copy type doesn't move
        assert!(!checker.is_move(&expr, "i32"));
        
        // References don't move
        assert!(!checker.is_move(&expr, "&String"));
    }
}
