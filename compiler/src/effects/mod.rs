//! Effect system for tracking and inferring side effects

pub mod effect;
pub mod inference;
pub mod checker;

pub use effect::{Effect, EffectSet, EffectKind};
pub use inference::EffectInference;
pub use checker::{EffectChecker, EffectError};
