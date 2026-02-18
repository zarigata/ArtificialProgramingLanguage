//! Deadline management for real-time tasks

use std::time::{Duration, Instant};
use std::fmt;

/// Deadline specification with constraints
#[derive(Debug, Clone)]
pub struct Deadline {
    pub absolute: Instant,
    pub relative: Duration,
    pub constraint: DeadlineConstraint,
}

impl Deadline {
    pub fn from_relative(duration: Duration) -> Self {
        Deadline {
            absolute: Instant::now() + duration,
            relative: duration,
            constraint: DeadlineConstraint::Hard,
        }
    }

    pub fn from_absolute(instant: Instant) -> Self {
        let now = Instant::now();
        Deadline {
            absolute: instant,
            relative: if instant > now {
                instant.duration_since(now)
            } else {
                Duration::ZERO
            },
            constraint: DeadlineConstraint::Hard,
        }
    }

    pub fn soft(duration: Duration) -> Self {
        let mut dl = Self::from_relative(duration);
        dl.constraint = DeadlineConstraint::Soft;
        dl
    }

    pub fn firm(duration: Duration) -> Self {
        let mut dl = Self::from_relative(duration);
        dl.constraint = DeadlineConstraint::Firm;
        dl
    }

    pub fn hard(duration: Duration) -> Self {
        Self::from_relative(duration)
    }

    pub fn with_constraint(mut self, constraint: DeadlineConstraint) -> Self {
        self.constraint = constraint;
        self
    }

    pub fn is_expired(&self) -> bool {
        Instant::now() > self.absolute
    }

    pub fn time_remaining(&self) -> Duration {
        let now = Instant::now();
        if now >= self.absolute {
            Duration::ZERO
        } else {
            self.absolute.duration_since(now)
        }
    }

    pub fn utilization_factor(&self, wcet: Duration) -> f64 {
        if self.relative.is_zero() {
            0.0
        } else {
            wcet.as_secs_f64() / self.relative.as_secs_f64()
        }
    }

    pub fn check(&self) -> DeadlineResult {
        let now = Instant::now();
        
        if now > self.absolute {
            DeadlineResult::Missed(now.duration_since(self.absolute))
        } else if now + Duration::from_micros(100) > self.absolute {
            DeadlineResult::Critical(self.absolute.duration_since(now))
        } else {
            DeadlineResult::OnTime(self.absolute.duration_since(now))
        }
    }
}

impl fmt::Display for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let remaining = self.time_remaining();
        write!(
            f, 
            "Deadline({:?} remaining, {:?})", 
            remaining, 
            self.constraint
        )
    }
}

impl PartialEq for Deadline {
    fn eq(&self, other: &Self) -> bool {
        self.absolute == other.absolute
    }
}

impl PartialOrd for Deadline {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.absolute.partial_cmp(&other.absolute)
    }
}

/// Deadline constraint type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlineConstraint {
    /// Hard deadline - missing it is a system failure
    Hard,
    /// Soft deadline - missing it degrades quality but system continues
    Soft,
    /// Firm deadline - late results are useless but not fatal
    Firm,
}

impl fmt::Display for DeadlineConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeadlineConstraint::Hard => write!(f, "hard"),
            DeadlineConstraint::Soft => write!(f, "soft"),
            DeadlineConstraint::Firm => write!(f, "firm"),
        }
    }
}

/// Result of deadline checking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlineResult {
    /// Deadline is on track with time remaining
    OnTime(Duration),
    /// Deadline is critical - less than 100μs remaining
    Critical(Duration),
    /// Deadline was missed by the given duration
    Missed(Duration),
}

impl DeadlineResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, DeadlineResult::OnTime(_))
    }

    pub fn is_critical(&self) -> bool {
        matches!(self, DeadlineResult::Critical(_))
    }

    pub fn is_missed(&self) -> bool {
        matches!(self, DeadlineResult::Missed(_))
    }

    pub fn time_remaining(&self) -> Option<Duration> {
        match self {
            DeadlineResult::OnTime(d) => Some(*d),
            DeadlineResult::Critical(d) => Some(*d),
            DeadlineResult::Missed(_) => None,
        }
    }

    pub fn time_overdue(&self) -> Option<Duration> {
        match self {
            DeadlineResult::Missed(d) => Some(*d),
            _ => None,
        }
    }
}

/// Deadline monitoring for multiple tasks
pub struct DeadlineMonitor {
    deadlines: Vec<(u64, Deadline)>,
    warnings_issued: u64,
    deadlines_missed: u64,
}

impl DeadlineMonitor {
    pub fn new() -> Self {
        DeadlineMonitor {
            deadlines: Vec::new(),
            warnings_issued: 0,
            deadlines_missed: 0,
        }
    }

    pub fn add(&mut self, task_id: u64, deadline: Deadline) {
        self.deadlines.push((task_id, deadline));
    }

    pub fn remove(&mut self, task_id: u64) {
        self.deadlines.retain(|(id, _)| *id != task_id);
    }

    pub fn check_all(&mut self) -> Vec<(u64, DeadlineResult)> {
        let mut results = Vec::new();
        
        for (task_id, deadline) in &self.deadlines {
            let result = deadline.check();
            
            if result.is_critical() {
                self.warnings_issued += 1;
            }
            
            if result.is_missed() {
                self.deadlines_missed += 1;
            }
            
            results.push((*task_id, result));
        }
        
        results
    }

    pub fn earliest_deadline(&self) -> Option<&Deadline> {
        self.deadlines.iter().min_by_key(|(_, d)| d.absolute).map(|(_, d)| d)
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.warnings_issued, self.deadlines_missed)
    }

    pub fn clear_completed(&mut self) {
        let now = Instant::now();
        self.deadlines.retain(|(_, d)| d.absolute > now);
    }
}

impl Default for DeadlineMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deadline_from_relative() {
        let dl = Deadline::from_relative(Duration::from_millis(100));
        assert!(!dl.is_expired());
        assert!(dl.time_remaining() <= Duration::from_millis(100));
    }

    #[test]
    fn test_deadline_check() {
        let dl = Deadline::from_relative(Duration::from_secs(10));
        let result = dl.check();
        assert!(result.is_ok());
        assert!(result.time_remaining().unwrap() > Duration::from_secs(9));
    }

    #[test]
    fn test_deadline_constraints() {
        let hard = Deadline::hard(Duration::from_millis(100));
        assert_eq!(hard.constraint, DeadlineConstraint::Hard);
        
        let soft = Deadline::soft(Duration::from_millis(100));
        assert_eq!(soft.constraint, DeadlineConstraint::Soft);
        
        let firm = Deadline::firm(Duration::from_millis(100));
        assert_eq!(firm.constraint, DeadlineConstraint::Firm);
    }

    #[test]
    fn test_deadline_monitor() {
        let mut monitor = DeadlineMonitor::new();
        
        monitor.add(1, Deadline::from_relative(Duration::from_secs(10)));
        monitor.add(2, Deadline::from_relative(Duration::from_secs(5)));
        
        let earliest = monitor.earliest_deadline();
        assert!(earliest.is_some());
        
        let results = monitor.check_all();
        assert_eq!(results.len(), 2);
        
        for (_, result) in results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_utilization_factor() {
        let dl = Deadline::from_relative(Duration::from_millis(100));
        let wcet = Duration::from_millis(20);
        
        let util = dl.utilization_factor(wcet);
        assert!((util - 0.2).abs() < 0.001);
    }
}
