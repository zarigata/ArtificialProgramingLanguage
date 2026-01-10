//! Borrow checker for memory safety

pub mod lifetime;
pub mod ownership;
pub mod checker;

pub use lifetime::{Lifetime, LifetimeEnv};
pub use ownership::{OwnershipTracker, MoveChecker};
pub use checker::BorrowChecker;
