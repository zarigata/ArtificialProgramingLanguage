//! Real-time scheduler with EDF (Earliest Deadline First) and RM (Rate Monotonic) policies

use std::collections::BinaryHeap;
use std::time::{Duration, Instant};
use std::cmp::Ordering;
use super::{TaskId, TaskState, SchedulerStats};

/// Scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePolicy {
    /// Earliest Deadline First - dynamic priority based on deadline
    EarliestDeadlineFirst,
    /// Rate Monotonic - static priority based on period (shorter = higher)
    RateMonotonic,
    /// Fixed Priority - user-defined priorities
    FixedPriority,
    /// Deadline Monotonic - static priority based on relative deadline
    DeadlineMonotonic,
}

/// Real-time task definition
#[derive(Debug, Clone)]
pub struct RealtimeTask {
    pub id: TaskId,
    pub period: Duration,
    pub deadline: Instant,
    pub relative_deadline: Duration,
    pub worst_case_exec_time: Duration,
    pub remaining_exec_time: Duration,
    pub state: TaskState,
    pub release_time: Instant,
    pub priority: u32,
}

impl PartialEq for RealtimeTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RealtimeTask {}

impl PartialOrd for RealtimeTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RealtimeTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior (earliest deadline first)
        other.deadline.cmp(&self.deadline)
    }
}

/// Real-time scheduler implementation
pub struct RealtimeScheduler {
    policy: SchedulePolicy,
    ready_queue: BinaryHeap<RealtimeTask>,
    current_task: Option<TaskId>,
    stats: SchedulerStats,
    tick_interval: Duration,
    last_tick: Instant,
    utilization: f64,
}

impl RealtimeScheduler {
    pub fn new(policy: SchedulePolicy) -> Self {
        RealtimeScheduler {
            policy,
            ready_queue: BinaryHeap::new(),
            current_task: None,
            stats: SchedulerStats::new(),
            tick_interval: Duration::from_micros(100),
            last_tick: Instant::now(),
            utilization: 0.0,
        }
    }

    pub fn with_tick_interval(mut self, interval: Duration) -> Self {
        self.tick_interval = interval;
        self
    }

    pub fn add_task(&mut self, task: RealtimeTask) {
        self.stats.tasks_scheduled += 1;
        self.ready_queue.push(task);
        self.recalculate_utilization();
    }

    pub fn remove_task(&mut self, id: TaskId) -> Option<RealtimeTask> {
        let mut removed = None;
        let mut new_queue = BinaryHeap::new();
        
        while let Some(task) = self.ready_queue.pop() {
            if task.id == id {
                removed = Some(task);
            } else {
                new_queue.push(task);
            }
        }
        
        self.ready_queue = new_queue;
        self.recalculate_utilization();
        removed
    }

    pub fn schedule(&mut self) -> Option<TaskId> {
        match self.policy {
            SchedulePolicy::EarliestDeadlineFirst => self.schedule_edf(),
            SchedulePolicy::RateMonotonic => self.schedule_rm(),
            SchedulePolicy::FixedPriority => self.schedule_fp(),
            SchedulePolicy::DeadlineMonotonic => self.schedule_dm(),
        }
    }

    fn schedule_edf(&mut self) -> Option<TaskId> {
        let now = Instant::now();
        
        while let Some(mut task) = self.ready_queue.pop() {
            if task.state != TaskState::Ready {
                continue;
            }
            
            if now > task.deadline {
                task.state = TaskState::MissedDeadline;
                self.stats.deadlines_missed += 1;
                continue;
            }
            
            task.state = TaskState::Running;
            self.current_task = Some(task.id);
            let id = task.id;
            self.ready_queue.push(task);
            return Some(id);
        }
        
        None
    }

    fn schedule_rm(&mut self) -> Option<TaskId> {
        let mut highest_priority: Option<RealtimeTask> = None;
        let mut other_tasks = Vec::new();
        
        while let Some(task) = self.ready_queue.pop() {
            if task.state == TaskState::Ready {
                if highest_priority.is_none() || task.period < highest_priority.as_ref().unwrap().period {
                    if let Some(prev) = highest_priority.take() {
                        other_tasks.push(prev);
                    }
                    highest_priority = Some(task);
                } else {
                    other_tasks.push(task);
                }
            } else {
                other_tasks.push(task);
            }
        }
        
        for task in other_tasks {
            self.ready_queue.push(task);
        }
        
        if let Some(mut task) = highest_priority {
            task.state = TaskState::Running;
            self.current_task = Some(task.id);
            let id = task.id;
            self.ready_queue.push(task);
            return Some(id);
        }
        
        None
    }

    fn schedule_fp(&mut self) -> Option<TaskId> {
        let mut highest_priority: Option<RealtimeTask> = None;
        let mut other_tasks = Vec::new();
        
        while let Some(task) = self.ready_queue.pop() {
            if task.state == TaskState::Ready {
                if highest_priority.is_none() || task.priority > highest_priority.as_ref().unwrap().priority {
                    if let Some(prev) = highest_priority.take() {
                        other_tasks.push(prev);
                    }
                    highest_priority = Some(task);
                } else {
                    other_tasks.push(task);
                }
            } else {
                other_tasks.push(task);
            }
        }
        
        for task in other_tasks {
            self.ready_queue.push(task);
        }
        
        if let Some(mut task) = highest_priority {
            task.state = TaskState::Running;
            self.current_task = Some(task.id);
            let id = task.id;
            self.ready_queue.push(task);
            return Some(id);
        }
        
        None
    }

    fn schedule_dm(&mut self) -> Option<TaskId> {
        let mut highest_priority: Option<RealtimeTask> = None;
        let mut other_tasks = Vec::new();
        
        while let Some(task) = self.ready_queue.pop() {
            if task.state == TaskState::Ready {
                if highest_priority.is_none() || 
                   task.relative_deadline < highest_priority.as_ref().unwrap().relative_deadline {
                    if let Some(prev) = highest_priority.take() {
                        other_tasks.push(prev);
                    }
                    highest_priority = Some(task);
                } else {
                    other_tasks.push(task);
                }
            } else {
                other_tasks.push(task);
            }
        }
        
        for task in other_tasks {
            self.ready_queue.push(task);
        }
        
        if let Some(mut task) = highest_priority {
            task.state = TaskState::Running;
            self.current_task = Some(task.id);
            let id = task.id;
            self.ready_queue.push(task);
            return Some(id);
        }
        
        None
    }

    pub fn tick(&mut self) -> Vec<TaskId> {
        let now = Instant::now();
        let mut ready_tasks = Vec::new();
        
        if now.duration_since(self.last_tick) >= self.tick_interval {
            self.last_tick = now;
            
            let mut updated_tasks = Vec::new();
            while let Some(mut task) = self.ready_queue.pop() {
                if now >= task.release_time && task.state == TaskState::Blocked {
                    task.state = TaskState::Ready;
                }
                
                if now > task.deadline && task.state != TaskState::Completed {
                    task.state = TaskState::MissedDeadline;
                    self.stats.deadlines_missed += 1;
                }
                
                if task.state == TaskState::Ready {
                    ready_tasks.push(task.id);
                }
                
                updated_tasks.push(task);
            }
            
            for task in updated_tasks {
                self.ready_queue.push(task);
            }
        }
        
        ready_tasks
    }

    pub fn complete_task(&mut self, id: TaskId, exec_time: Duration) {
        self.stats.tasks_completed += 1;
        self.stats.total_execution_time += exec_time;
        
        let mut updated_tasks = Vec::new();
        while let Some(mut task) = self.ready_queue.pop() {
            if task.id == id {
                task.state = TaskState::Completed;
                task.remaining_exec_time = Duration::ZERO;
            }
            updated_tasks.push(task);
        }
        
        for task in updated_tasks {
            self.ready_queue.push(task);
        }
        
        if self.current_task == Some(id) {
            self.current_task = None;
            self.stats.context_switches += 1;
        }
    }

    pub fn preempt(&mut self) -> Option<TaskId> {
        if let Some(current) = self.current_task {
            self.stats.preemptions += 1;
            self.stats.context_switches += 1;
            
            let mut updated_tasks = Vec::new();
            while let Some(mut task) = self.ready_queue.pop() {
                if task.id == current {
                    task.state = TaskState::Ready;
                }
                updated_tasks.push(task);
            }
            
            for task in updated_tasks {
                self.ready_queue.push(task);
            }
            
            self.current_task = None;
            return Some(current);
        }
        None
    }

    pub fn utilization(&self) -> f64 {
        self.utilization
    }

    fn recalculate_utilization(&mut self) {
        let mut total_util = 0.0;
        let tasks: Vec<_> = self.ready_queue.iter().collect();
        
        for task in tasks {
            let exec_time = task.worst_case_exec_time.as_secs_f64();
            let period = task.period.as_secs_f64();
            if period > 0.0 {
                total_util += exec_time / period;
            }
        }
        
        self.utilization = total_util;
    }

    pub fn is_schedulable(&self) -> bool {
        let n = self.ready_queue.len() as f64;
        
        match self.policy {
            SchedulePolicy::EarliestDeadlineFirst => {
                self.utilization <= 1.0
            }
            SchedulePolicy::RateMonotonic | SchedulePolicy::DeadlineMonotonic => {
                let bound = n * (2.0_f64.powf(1.0 / n) - 1.0);
                self.utilization <= bound
            }
            SchedulePolicy::FixedPriority => {
                self.utilization <= 1.0
            }
        }
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    pub fn current_task(&self) -> Option<TaskId> {
        self.current_task
    }

    pub fn pending_count(&self) -> usize {
        self.ready_queue.iter()
            .filter(|t| t.state == TaskState::Ready)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edf_scheduler() {
        let mut scheduler = RealtimeScheduler::new(SchedulePolicy::EarliestDeadlineFirst);
        
        let now = Instant::now();
        let task1 = RealtimeTask {
            id: TaskId(1),
            period: Duration::from_millis(100),
            deadline: now + Duration::from_millis(50),
            relative_deadline: Duration::from_millis(50),
            worst_case_exec_time: Duration::from_millis(10),
            remaining_exec_time: Duration::from_millis(10),
            state: TaskState::Ready,
            release_time: now,
            priority: 1,
        };
        
        let task2 = RealtimeTask {
            id: TaskId(2),
            period: Duration::from_millis(200),
            deadline: now + Duration::from_millis(30),
            relative_deadline: Duration::from_millis(30),
            worst_case_exec_time: Duration::from_millis(5),
            remaining_exec_time: Duration::from_millis(5),
            state: TaskState::Ready,
            release_time: now,
            priority: 2,
        };
        
        scheduler.add_task(task1);
        scheduler.add_task(task2);
        
        let scheduled = scheduler.schedule();
        assert_eq!(scheduled, Some(TaskId(2))); // Earlier deadline
    }

    #[test]
    fn test_rm_scheduler() {
        let mut scheduler = RealtimeScheduler::new(SchedulePolicy::RateMonotonic);
        
        let now = Instant::now();
        let task1 = RealtimeTask {
            id: TaskId(1),
            period: Duration::from_millis(100),
            deadline: now + Duration::from_millis(100),
            relative_deadline: Duration::from_millis(100),
            worst_case_exec_time: Duration::from_millis(20),
            remaining_exec_time: Duration::from_millis(20),
            state: TaskState::Ready,
            release_time: now,
            priority: 1,
        };
        
        let task2 = RealtimeTask {
            id: TaskId(2),
            period: Duration::from_millis(50),
            deadline: now + Duration::from_millis(50),
            relative_deadline: Duration::from_millis(50),
            worst_case_exec_time: Duration::from_millis(10),
            remaining_exec_time: Duration::from_millis(10),
            state: TaskState::Ready,
            release_time: now,
            priority: 2,
        };
        
        scheduler.add_task(task1);
        scheduler.add_task(task2);
        
        let scheduled = scheduler.schedule();
        assert_eq!(scheduled, Some(TaskId(2))); // Shorter period = higher priority
    }

    #[test]
    fn test_schedulability() {
        let mut scheduler = RealtimeScheduler::new(SchedulePolicy::EarliestDeadlineFirst);
        
        let now = Instant::now();
        let task = RealtimeTask {
            id: TaskId(1),
            period: Duration::from_millis(100),
            deadline: now + Duration::from_millis(100),
            relative_deadline: Duration::from_millis(100),
            worst_case_exec_time: Duration::from_millis(10),
            remaining_exec_time: Duration::from_millis(10),
            state: TaskState::Ready,
            release_time: now,
            priority: 1,
        };
        
        scheduler.add_task(task);
        assert!(scheduler.is_schedulable());
    }
}
