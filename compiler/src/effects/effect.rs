//! Effect definitions

use std::fmt;
use std::collections::{HashMap, HashSet};

/// Effect kind - categorizes what kind of side effect occurs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectKind {
    /// IO read effect
    IoRead,
    /// IO write effect
    IoWrite,
    /// State read effect
    StateRead,
    /// State write effect
    StateWrite,
    /// Memory allocation
    Allocate,
    /// Memory deallocation
    Deallocate,
    /// Async operation (may yield)
    AsyncYield,
    /// Async spawn
    AsyncSpawn,
    /// May throw exception
    Throw,
    /// Catches exceptions
    Catch,
    /// Pure (no side effects)
    Pure,
    /// Unknown effect
    Unknown,
    /// Custom named effect
    Custom(String),
}

impl EffectKind {
    pub fn is_pure(&self) -> bool {
        matches!(self, EffectKind::Pure)
    }
    
    pub fn is_io(&self) -> bool {
        matches!(self, EffectKind::IoRead | EffectKind::IoWrite)
    }
    
    pub fn is_state(&self) -> bool {
        matches!(self, EffectKind::StateRead | EffectKind::StateWrite)
    }
    
    pub fn is_async(&self) -> bool {
        matches!(self, EffectKind::AsyncYield | EffectKind::AsyncSpawn)
    }
    
    pub fn is_safe_to_reorder(&self) -> bool {
        matches!(
            self,
            EffectKind::Pure | 
            EffectKind::StateRead | 
            EffectKind::IoRead |
            EffectKind::Unknown
        )
    }
}

impl fmt::Display for EffectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectKind::IoRead => write!(f, "IO.read"),
            EffectKind::IoWrite => write!(f, "IO.write"),
            EffectKind::StateRead => write!(f, "State.read"),
            EffectKind::StateWrite => write!(f, "State.write"),
            EffectKind::Allocate => write!(f, "Memory.alloc"),
            EffectKind::Deallocate => write!(f, "Memory.free"),
            EffectKind::AsyncYield => write!(f, "Async.yield"),
            EffectKind::AsyncSpawn => write!(f, "Async.spawn"),
            EffectKind::Throw => write!(f, "Exception.throw"),
            EffectKind::Catch => write!(f, "Exception.catch"),
            EffectKind::Pure => write!(f, "Pure"),
            EffectKind::Unknown => write!(f, "Unknown"),
            EffectKind::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// A single effect with optional location and target information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect {
    pub kind: EffectKind,
    pub target: Option<String>,
}

impl Effect {
    pub fn new(kind: EffectKind) -> Self {
        Effect { kind, target: None }
    }
    
    pub fn with_target(kind: EffectKind, target: &str) -> Self {
        Effect {
            kind,
            target: Some(target.to_string()),
        }
    }
    
    pub fn io_read() -> Self {
        Effect::new(EffectKind::IoRead)
    }
    
    pub fn io_write() -> Self {
        Effect::new(EffectKind::IoWrite)
    }
    
    pub fn state_read() -> Self {
        Effect::new(EffectKind::StateRead)
    }
    
    pub fn state_write() -> Self {
        Effect::new(EffectKind::StateWrite)
    }
    
    pub fn pure() -> Self {
        Effect::new(EffectKind::Pure)
    }
    
    pub fn async_yield() -> Self {
        Effect::new(EffectKind::AsyncYield)
    }
    
    pub fn throws() -> Self {
        Effect::new(EffectKind::Throw)
    }
    
    pub fn is_pure(&self) -> bool {
        self.kind.is_pure()
    }
    
    pub fn is_io(&self) -> bool {
        self.kind.is_io()
    }
    
    pub fn is_state(&self) -> bool {
        self.kind.is_state()
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.target {
            Some(target) => write!(f, "{}[{}]", self.kind, target),
            None => write!(f, "{}", self.kind),
        }
    }
}

impl From<EffectKind> for Effect {
    fn from(kind: EffectKind) -> Self {
        Effect::new(kind)
    }
}

/// A set of effects
#[derive(Debug, Clone, Default)]
pub struct EffectSet {
    effects: HashSet<EffectKind>,
    targets: HashMap<EffectKind, HashSet<String>>,
}

impl EffectSet {
    pub fn new() -> Self {
        EffectSet {
            effects: HashSet::new(),
            targets: HashMap::new(),
        }
    }
    
    pub fn pure() -> Self {
        let mut set = Self::new();
        set.effects.insert(EffectKind::Pure);
        set
    }
    
    pub fn unknown() -> Self {
        let mut set = Self::new();
        set.effects.insert(EffectKind::Unknown);
        set
    }
    
    pub fn insert(&mut self, effect: EffectKind) {
        self.effects.insert(effect);
    }
    
    pub fn insert_with_target(&mut self, effect: EffectKind, target: String) {
        self.effects.insert(effect.clone());
        self.targets.entry(effect).or_default().insert(target);
    }
    
    pub fn extend(&mut self, other: &EffectSet) {
        for effect in &other.effects {
            self.effects.insert(effect.clone());
        }
        for (kind, targets) in &other.targets {
            self.targets
                .entry(kind.clone())
                .or_default()
                .extend(targets.iter().cloned());
        }
    }
    
    pub fn contains(&self, effect: &EffectKind) -> bool {
        self.effects.contains(effect)
    }
    
    pub fn is_pure(&self) -> bool {
        self.effects.len() == 1 && self.effects.contains(&EffectKind::Pure)
    }
    
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &EffectKind> {
        self.effects.iter()
    }
    
    pub fn union(&self, other: &EffectSet) -> EffectSet {
        let mut result = self.clone();
        result.extend(other);
        result
    }
    
    pub fn has_io(&self) -> bool {
        self.effects.iter().any(|e| e.is_io())
    }
    
    pub fn has_state(&self) -> bool {
        self.effects.iter().any(|e| e.is_state())
    }
    
    pub fn has_async(&self) -> bool {
        self.effects.iter().any(|e| e.is_async())
    }
    
    pub fn may_throw(&self) -> bool {
        self.effects.contains(&EffectKind::Throw)
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.effects.is_empty() {
            return write!(f, "{{}}");
        }
        
        let effects: Vec<_> = self.effects.iter().collect();
        write!(f, "{{")?;
        for (i, effect) in effects.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", effect)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_creation() {
        let effect = Effect::io_read();
        assert!(effect.is_io());
        assert!(!effect.is_pure());
    }

    #[test]
    fn test_effect_set() {
        let mut set = EffectSet::new();
        set.insert(EffectKind::IoRead);
        set.insert(EffectKind::IoWrite);
        
        assert!(set.has_io());
        assert!(!set.is_pure());
    }

    #[test]
    fn test_effect_set_union() {
        let mut set1 = EffectSet::new();
        set1.insert(EffectKind::IoRead);
        
        let mut set2 = EffectSet::new();
        set2.insert(EffectKind::StateWrite);
        
        let union = set1.union(&set2);
        assert!(union.contains(&EffectKind::IoRead));
        assert!(union.contains(&EffectKind::StateWrite));
    }

    #[test]
    fn test_effect_display() {
        assert_eq!(EffectKind::IoRead.to_string(), "IO.read");
        assert_eq!(EffectKind::Pure.to_string(), "Pure");
    }
}
