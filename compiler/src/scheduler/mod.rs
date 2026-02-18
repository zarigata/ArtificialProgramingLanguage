//! Deterministic Execution Scheduler
//!
//! Provides real-time task scheduling with guaranteed timing bounds.
//! Designed for embedded systems, robotics, and safety-critical applications.

pub mod realtime;
pub mod deadline;
pub mod priority;

pub use realtime::RealtimeScheduler;
pub use deadline::{Deadline, DeadlineResult};
pub use priority::{Priority, TaskPriority};

use std::time::{Duration, Instant};

/// Task state in the scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is ready to run
    Ready,
    /// Task is currently executing
    Running,
    /// Task is blocked waiting for a resource
    Blocked,
    /// Task completed successfully
    Completed,
    /// Task failed to meet its deadline
    MissedDeadline,
    /// Task was suspended
    Suspended,
}

/// Task handle for scheduler operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// Scheduler statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub tasks_scheduled: u64,
    pub tasks_completed: u64,
    pub deadlines_missed: u64,
    pub total_execution_time: Duration,
    pub context_switches: u64,
    pub preemptions: u64,
}

impl SchedulerStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn deadline_miss_rate(&self) -> f64 {
        if self.tasks_scheduled == 0 {
            0.0
        } else {
            self.deadlines_missed as f64 / self.tasks_scheduled as f64
        }
    }
    
    pub fn avg_execution_time(&self) -> Duration {
        if self.tasks_completed == 0 {
            Duration::ZERO
        } else {
            self.total_execution_time / self.tasks_completed as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_state() {
        assert_eq!(TaskState::Ready, TaskState::Ready);
        assert_ne!(TaskState::Ready, TaskState::Running);
    }

    #[test]
    fn test_task_id() {
        let id1 = TaskId(1);
        let id2 = TaskId(2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut stats = SchedulerStats::new();
        stats.tasks_scheduled = 100;
        stats.tasks_completed = 95;
        stats.deadlines_missed = 5;
        
        assert_eq!(stats.deadline_miss_rate(), 0.05);
    }
}
