//! Priority management for real-time tasks

use std::cmp::Ordering;
use std::fmt;

/// Task priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Priority(pub u32);

impl Priority {
    pub const MIN: Priority = Priority(0);
    pub const MAX: Priority = Priority(255);
    pub const DEFAULT: Priority = Priority(128);
    
    pub fn new(level: u32) -> Self {
        Priority(level.min(255))
    }
    
    pub fn level(&self) -> u32 {
        self.0
    }
    
    pub fn is_higher(&self, other: &Priority) -> bool {
        self.0 > other.0
    }
    
    pub fn is_lower(&self, other: &Priority) -> bool {
        self.0 < other.0
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.0)
    }
}

/// Extended priority with inheritance support
#[derive(Debug, Clone)]
pub struct TaskPriority {
    pub base: Priority,
    pub current: Priority,
    pub inherited: Option<Priority>,
    pub boosted_until: Option<std::time::Instant>,
}

impl TaskPriority {
    pub fn new(level: u32) -> Self {
        let p = Priority::new(level);
        TaskPriority {
            base: p,
            current: p,
            inherited: None,
            boosted_until: None,
        }
    }
    
    pub fn from_priority(priority: Priority) -> Self {
        TaskPriority {
            base: priority,
            current: priority,
            inherited: None,
            boosted_until: None,
        }
    }
    
    pub fn effective(&self) -> Priority {
        let now = std::time::Instant::now();
        
        if let Some(until) = self.boosted_until {
            if now < until {
                return self.current;
            }
        }
        
        self.current
    }
    
    pub fn inherit(&mut self, priority: Priority) {
        if priority.is_higher(&self.current) {
            self.inherited = Some(priority);
            self.current = priority;
        }
    }
    
    pub fn boost(&mut self, priority: Priority, duration: std::time::Duration) {
        if priority.is_higher(&self.current) {
            self.current = priority;
            self.boosted_until = Some(std::time::Instant::now() + duration);
        }
    }
    
    pub fn reset(&mut self) {
        self.current = self.base;
        self.inherited = None;
        self.boosted_until = None;
    }
    
    pub fn is_boosted(&self) -> bool {
        if let Some(until) = self.boosted_until {
            std::time::Instant::now() < until
        } else {
            false
        }
    }
    
    pub fn has_inherited(&self) -> bool {
        self.inherited.is_some()
    }
}

impl PartialEq for TaskPriority {
    fn eq(&self, other: &Self) -> bool {
        self.effective() == other.effective()
    }
}

impl Eq for TaskPriority {}

impl PartialOrd for TaskPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaskPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.effective().cmp(&other.effective())
    }
}

/// Priority inheritance protocol to prevent priority inversion
pub struct PriorityInheritance {
    holders: Vec<(u64, TaskPriority)>,
    waiters: Vec<(u64, TaskPriority)>,
}

impl PriorityInheritance {
    pub fn new() -> Self {
        PriorityInheritance {
            holders: Vec::new(),
            waiters: Vec::new(),
        }
    }
    
    pub fn acquire(&mut self, task_id: u64, mut priority: TaskPriority) {
        for (holder_id, holder_priority) in &mut self.holders {
            if priority.effective().is_higher(&holder_priority.effective()) {
                holder_priority.inherit(priority.effective());
            }
        }
        
        self.holders.push((task_id, priority));
    }
    
    pub fn release(&mut self, task_id: u64) {
        self.holders.retain(|(id, _)| *id != task_id);
        
        for (_, priority) in &mut self.holders {
            priority.reset();
        }
    }
    
    pub fn add_waiter(&mut self, task_id: u64, priority: TaskPriority) {
        for (holder_id, holder_priority) in &mut self.holders {
            if priority.effective().is_higher(&holder_priority.effective()) {
                holder_priority.inherit(priority.effective());
            }
        }
        
        self.waiters.push((task_id, priority));
    }
    
    pub fn remove_waiter(&mut self, task_id: u64) {
        self.waiters.retain(|(id, _)| *id != task_id);
    }
}

impl Default for PriorityInheritance {
    fn default() -> Self {
        Self::new()
    }
}

/// Priority ceiling protocol
pub struct PriorityCeiling {
    ceiling: Priority,
    holder: Option<u64>,
}

impl PriorityCeiling {
    pub fn new(ceiling: Priority) -> Self {
        PriorityCeiling {
            ceiling,
            holder: None,
        }
    }
    
    pub fn acquire(&mut self, task_id: u64, mut priority: TaskPriority) -> bool {
        if self.holder.is_some() {
            return false;
        }
        
        priority.boost(self.ceiling, std::time::Duration::from_secs(3600));
        self.holder = Some(task_id);
        true
    }
    
    pub fn release(&mut self, task_id: u64) -> bool {
        if self.holder != Some(task_id) {
            return false;
        }
        
        self.holder = None;
        true
    }
    
    pub fn is_held(&self) -> bool {
        self.holder.is_some()
    }
    
    pub fn holder(&self) -> Option<u64> {
        self.holder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_basic() {
        let p1 = Priority::new(100);
        let p2 = Priority::new(200);
        
        assert!(p2.is_higher(&p1));
        assert!(p1.is_lower(&p2));
        assert_eq!(p1.level(), 100);
    }

    #[test]
    fn test_priority_clamp() {
        let p = Priority::new(300);
        assert_eq!(p.level(), 255);
    }

    #[test]
    fn test_task_priority_inheritance() {
        let mut tp = TaskPriority::new(100);
        let higher = Priority::new(200);
        
        tp.inherit(higher);
        assert!(tp.has_inherited());
        assert_eq!(tp.effective(), higher);
        
        tp.reset();
        assert!(!tp.has_inherited());
        assert_eq!(tp.effective(), Priority::new(100));
    }

    #[test]
    fn test_priority_ordering() {
        let mut tp1 = TaskPriority::new(100);
        let tp2 = TaskPriority::new(200);
        
        assert!(tp2 > tp1);
        
        tp1.inherit(Priority::new(250));
        assert!(tp1 > tp2);
    }

    #[test]
    fn test_priority_inheritance_protocol() {
        let mut protocol = PriorityInheritance::new();
        
        protocol.acquire(1, TaskPriority::new(100));
        protocol.add_waiter(2, TaskPriority::new(200));
        
        let holder = &protocol.holders[0];
        assert_eq!(holder.1.effective(), Priority::new(200));
        
        protocol.release(1);
        assert!(protocol.holders.is_empty());
    }

    #[test]
    fn test_priority_ceiling() {
        let mut ceiling = PriorityCeiling::new(Priority::new(150));
        
        assert!(ceiling.acquire(1, TaskPriority::new(100)));
        assert!(ceiling.is_held());
        assert!(!ceiling.acquire(2, TaskPriority::new(50)));
        
        assert!(ceiling.release(1));
        assert!(!ceiling.is_held());
    }
}
