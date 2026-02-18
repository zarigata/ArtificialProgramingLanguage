//! Timing Intrinsics for deterministic execution

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Timer {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    pub fn elapsed_nanos(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }
    
    pub fn elapsed_micros(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
    
    pub fn elapsed_millis(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
    
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingOp {
    Rdtsc,
    Rdtscp,
    Pause,
    Yield,
    Sleep,
    BusyWait,
    GetTime,
    GetTimeMonotonic,
    GetTimeReal,
}

pub struct TimingIntrinsic {
    pub op: TimingOp,
    pub deadline_ns: Option<u64>,
}

impl TimingIntrinsic {
    pub fn new(op: TimingOp) -> Self {
        TimingIntrinsic { op, deadline_ns: None }
    }
    
    pub fn with_deadline(op: TimingOp, deadline_ns: u64) -> Self {
        TimingIntrinsic { 
            op, 
            deadline_ns: Some(deadline_ns) 
        }
    }
    
    pub fn to_llvm(&self) -> String {
        match self.op {
            TimingOp::Rdtsc => "llvm.x86.rdtsc".to_string(),
            TimingOp::Rdtscp => "llvm.x86.rdtscp".to_string(),
            TimingOp::Pause => "llvm.x86.sse2.pause".to_string(),
            _ => format!("{:?}", self.op),
        }
    }
}

#[inline]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::_rdtsc;
        _rdtsc()
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

#[inline]
pub fn pause() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::_mm_pause;
        _mm_pause();
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::thread::yield_now();
    }
}

#[inline]
pub fn cpu_relax() {
    pause()
}

#[inline]
pub fn busy_wait_ns(ns: u64) {
    let start = rdtsc();
    let cycles = ns * 3;
    while rdtsc().wrapping_sub(start) < cycles {
        pause();
    }
}

#[inline]
pub fn deadline_check(deadline_ns: u64, start_ns: u64) -> bool {
    rdtsc().wrapping_sub(start_ns) < deadline_ns
}

pub struct Deadline {
    deadline_ns: u64,
    start_ns: u64,
}

impl Deadline {
    pub fn from_ns(duration_ns: u64) -> Self {
        Deadline {
            deadline_ns: duration_ns,
            start_ns: rdtsc(),
        }
    }
    
    pub fn from_micros(duration_us: u64) -> Self {
        Self::from_ns(duration_us * 1000)
    }
    
    pub fn from_millis(duration_ms: u64) -> Self {
        Self::from_ns(duration_ms * 1_000_000)
    }
    
    pub fn remaining_ns(&self) -> u64 {
        let elapsed = rdtsc().wrapping_sub(self.start_ns);
        if elapsed >= self.deadline_ns {
            0
        } else {
            self.deadline_ns - elapsed
        }
    }
    
    pub fn is_expired(&self) -> bool {
        rdtsc().wrapping_sub(self.start_ns) >= self.deadline_ns
    }
    
    pub fn check(&self) -> bool {
        !self.is_expired()
    }
    
    pub fn reset(&mut self) {
        self.start_ns = rdtsc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed_millis() >= 10);
    }

    #[test]
    fn test_deadline() {
        let deadline = Deadline::from_ns(1_000_000_000);
        assert!(!deadline.is_expired());
    }

    #[test]
    fn test_rdtsc() {
        let t1 = rdtsc();
        let t2 = rdtsc();
        assert!(t2 >= t1);
    }
}
