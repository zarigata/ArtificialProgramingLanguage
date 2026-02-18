//! Atomic Intrinsics for lock-free programming
//!
//! Provides atomic operations for thread-safe memory access without locks.
//! Used for low-level synchronization in AI-generated systems code.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering as StdOrdering};
use std::fmt;

/// Memory ordering constraints for atomic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// No ordering constraints, only atomicity
    Relaxed,
    /// Data-dependent ordering (maps to Acquire in practice)
    Consume,
    /// Prevents reordering of reads before this point
    Acquire,
    /// Prevents reordering of writes after this point
    Release,
    /// Combined Acquire and Release
    AcqRel,
    /// Sequentially consistent ordering
    SeqCst,
}

impl MemoryOrder {
    /// Convert to standard library Ordering
    pub fn to_std(&self) -> StdOrdering {
        match self {
            MemoryOrder::Relaxed => StdOrdering::Relaxed,
            MemoryOrder::Consume => StdOrdering::Acquire, // Consume maps to Acquire
            MemoryOrder::Acquire => StdOrdering::Acquire,
            MemoryOrder::Release => StdOrdering::Release,
            MemoryOrder::AcqRel => StdOrdering::AcqRel,
            MemoryOrder::SeqCst => StdOrdering::SeqCst,
        }
    }
}

impl fmt::Display for MemoryOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryOrder::Relaxed => write!(f, "relaxed"),
            MemoryOrder::Consume => write!(f, "consume"),
            MemoryOrder::Acquire => write!(f, "acquire"),
            MemoryOrder::Release => write!(f, "release"),
            MemoryOrder::AcqRel => write!(f, "acq_rel"),
            MemoryOrder::SeqCst => write!(f, "seq_cst"),
        }
    }
}

/// Atomic operation types for code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    /// Load value atomically
    Load,
    /// Store value atomically
    Store,
    /// Exchange (swap) value atomically
    Exchange,
    /// Compare and exchange (strong)
    CompareExchange,
    /// Compare and exchange (weak)
    CompareExchangeWeak,
    /// Fetch and add
    FetchAdd,
    /// Fetch and subtract
    FetchSub,
    /// Fetch and AND
    FetchAnd,
    /// Fetch and OR
    FetchOr,
    /// Fetch and XOR
    FetchXor,
    /// Fetch and NAND
    FetchNand,
    /// Fetch and max
    FetchMax,
    /// Fetch and min
    FetchMin,
    /// Fetch and update with function
    FetchUpdate,
    /// Memory fence
    Fence,
}

/// Represents an atomic intrinsic for LLVM code generation
pub struct AtomicIntrinsic {
    /// The atomic operation type
    pub op: AtomicOp,
    /// Memory ordering for success
    pub order: MemoryOrder,
    /// Memory ordering for failure (used in compare_exchange)
    pub failure_order: Option<MemoryOrder>,
}

impl AtomicIntrinsic {
    /// Create a new atomic intrinsic
    pub fn new(op: AtomicOp, order: MemoryOrder) -> Self {
        AtomicIntrinsic { 
            op, 
            order, 
            failure_order: None 
        }
    }
    
    /// Create a compare_exchange intrinsic with separate success/failure ordering
    pub fn with_failure_order(op: AtomicOp, success: MemoryOrder, failure: MemoryOrder) -> Self {
        AtomicIntrinsic { 
            op, 
            order: success, 
            failure_order: Some(failure) 
        }
    }
    
    /// Generate LLVM IR representation
    pub fn to_llvm(&self) -> String {
        match self.op {
            AtomicOp::Load => format!("atomic.load.{}", self.order),
            AtomicOp::Store => format!("atomic.store.{}", self.order),
            AtomicOp::Exchange => format!("atomic.xchg.{}", self.order),
            AtomicOp::CompareExchange => format!("atomic.cmpxchg.{}", self.order),
            AtomicOp::CompareExchangeWeak => format!("atomic.cmpxchg.weak.{}", self.order),
            AtomicOp::FetchAdd => format!("atomic.add.{}", self.order),
            AtomicOp::FetchSub => format!("atomic.sub.{}", self.order),
            AtomicOp::FetchAnd => format!("atomic.and.{}", self.order),
            AtomicOp::FetchOr => format!("atomic.or.{}", self.order),
            AtomicOp::FetchXor => format!("atomic.xor.{}", self.order),
            AtomicOp::FetchNand => format!("atomic.nand.{}", self.order),
            AtomicOp::FetchMax => format!("atomic.max.{}", self.order),
            AtomicOp::FetchMin => format!("atomic.min.{}", self.order),
            AtomicOp::Fence => format!("atomic.fence.{}", self.order),
            AtomicOp::FetchUpdate => format!("atomic.update.{}", self.order),
        }
    }
}

/// Atomic load for usize values
#[inline]
pub fn atomic_load_usize(ptr: *const usize, order: MemoryOrder) -> usize {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.load(order.to_std())
}

/// Atomic store for usize values
#[inline]
pub fn atomic_store_usize(ptr: *mut usize, val: usize, order: MemoryOrder) {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.store(val, order.to_std());
}

/// Atomic exchange (swap) for usize values
#[inline]
pub fn atomic_exchange_usize(ptr: *mut usize, val: usize, order: MemoryOrder) -> usize {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.swap(val, order.to_std())
}

/// Atomic compare and exchange for usize values
/// Returns (old_value, success)
#[inline]
pub fn atomic_compare_exchange_usize(
    ptr: *mut usize, 
    expected: &mut usize, 
    new: usize, 
    success: MemoryOrder,
    failure: MemoryOrder
) -> bool {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    match atomic.compare_exchange(
        *expected,
        new,
        success.to_std(),
        failure.to_std()
    ) {
        Ok(_old) => true,
        Err(actual) => {
            *expected = actual;
            false
        }
    }
}

/// Atomic fetch-add for usize values
#[inline]
pub fn atomic_fetch_add_usize(ptr: *mut usize, val: usize, order: MemoryOrder) -> usize {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.fetch_add(val, order.to_std())
}

/// Atomic fetch-sub for usize values
#[inline]
pub fn atomic_fetch_sub_usize(ptr: *mut usize, val: usize, order: MemoryOrder) -> usize {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.fetch_sub(val, order.to_std())
}

/// Atomic fetch-and for usize values
#[inline]
pub fn atomic_fetch_and_usize(ptr: *mut usize, val: usize, order: MemoryOrder) -> usize {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.fetch_and(val, order.to_std())
}

/// Atomic fetch-or for usize values
#[inline]
pub fn atomic_fetch_or_usize(ptr: *mut usize, val: usize, order: MemoryOrder) -> usize {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.fetch_or(val, order.to_std())
}

/// Atomic fetch-xor for usize values
#[inline]
pub fn atomic_fetch_xor_usize(ptr: *mut usize, val: usize, order: MemoryOrder) -> usize {
    let atomic = unsafe { &*(ptr as *const AtomicUsize) };
    atomic.fetch_xor(val, order.to_std())
}

/// Atomic load for pointers
#[inline]
pub fn atomic_load_ptr<T>(ptr: *const *mut T, order: MemoryOrder) -> *mut T {
    let atomic = unsafe { &*(ptr as *const AtomicPtr<T>) };
    atomic.load(order.to_std())
}

/// Atomic store for pointers
#[inline]
pub fn atomic_store_ptr<T>(ptr: *mut *mut T, val: *mut T, order: MemoryOrder) {
    let atomic = unsafe { &*(ptr as *const AtomicPtr<T>) };
    atomic.store(val, order.to_std());
}

/// Spin loop hint (pause instruction on x86, yield on ARM)
#[inline]
pub fn spin_loop_hint() {
    std::hint::spin_loop();
}

/// A simple spin lock for low-latency synchronization
pub struct SpinLock {
    locked: std::sync::atomic::AtomicBool,
}

impl SpinLock {
    /// Create a new unlocked spin lock
    pub fn new() -> Self {
        SpinLock {
            locked: std::sync::atomic::AtomicBool::new(false),
        }
    }
    
    /// Acquire the lock, spinning until available
    pub fn lock(&self) {
        while self.locked.compare_exchange(
            false, 
            true, 
            StdOrdering::Acquire, 
            StdOrdering::Relaxed
        ).is_err() {
            while self.locked.load(StdOrdering::Relaxed) {
                spin_loop_hint();
            }
        }
    }
    
    /// Try to acquire the lock without blocking
    pub fn try_lock(&self) -> bool {
        self.locked.compare_exchange(
            false, 
            true, 
            StdOrdering::Acquire, 
            StdOrdering::Relaxed
        ).is_ok()
    }
    
    /// Release the lock
    pub fn unlock(&self) {
        self.locked.store(false, StdOrdering::Release);
    }
    
    /// Check if the lock is currently held
    pub fn is_locked(&self) -> bool {
        self.locked.load(StdOrdering::Relaxed)
    }
}

impl Default for SpinLock {
    fn default() -> Self {
        Self::new()
    }
}

/// A ticket lock for fair spin locking
pub struct TicketLock {
    ticket: std::sync::atomic::AtomicUsize,
    serving: std::sync::atomic::AtomicUsize,
}

impl TicketLock {
    /// Create a new unlocked ticket lock
    pub fn new() -> Self {
        TicketLock {
            ticket: std::sync::atomic::AtomicUsize::new(0),
            serving: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Acquire the lock, returns the ticket number
    pub fn lock(&self) -> usize {
        let ticket = self.ticket.fetch_add(1, StdOrdering::Relaxed);
        while self.serving.load(StdOrdering::Acquire) != ticket {
            spin_loop_hint();
        }
        ticket
    }
    
    /// Release the lock
    pub fn unlock(&self) {
        self.serving.fetch_add(1, StdOrdering::Release);
    }
    
    /// Check if the lock is currently held
    pub fn is_locked(&self) -> bool {
        let ticket = self.ticket.load(StdOrdering::Relaxed);
        let serving = self.serving.load(StdOrdering::Relaxed);
        ticket != serving
    }
}

impl Default for TicketLock {
    fn default() -> Self {
        Self::new()
    }
}

/// SeqLock (sequence lock) for read-mostly workloads
/// Writers serialize with a spin lock, readers verify sequence numbers
pub struct SeqLock {
    sequence: std::sync::atomic::AtomicUsize,
}

impl SeqLock {
    /// Create a new sequence lock
    pub fn new() -> Self {
        SeqLock {
            sequence: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Begin a read section, returns the current sequence number
    /// If odd, a write is in progress
    pub fn read_begin(&self) -> usize {
        loop {
            let seq = self.sequence.load(StdOrdering::Acquire);
            if seq & 1 == 0 {
                return seq;
            }
            spin_loop_hint();
        }
    }
    
    /// Validate a read section, returns true if no concurrent write occurred
    pub fn read_retry(&self, start_seq: usize) -> bool {
        std::sync::atomic::fence(StdOrdering::AcqRel);
        self.sequence.load(StdOrdering::Acquire) != start_seq
    }
    
    /// Begin a write section
    pub fn write_begin(&self) {
        let seq = self.sequence.load(StdOrdering::Relaxed);
        if seq & 1 != 0 {
            panic!("SeqLock: recursive write");
        }
        self.sequence.store(seq + 1, StdOrdering::Relaxed);
        std::sync::atomic::fence(StdOrdering::Release);
    }
    
    /// End a write section
    pub fn write_end(&self) {
        let seq = self.sequence.load(StdOrdering::Relaxed);
        self.sequence.store(seq + 1, StdOrdering::Release);
    }
}

impl Default for SeqLock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_order() {
        assert_eq!(MemoryOrder::SeqCst.to_std(), StdOrdering::SeqCst);
        assert_eq!(MemoryOrder::Relaxed.to_std(), StdOrdering::Relaxed);
        assert_eq!(MemoryOrder::Acquire.to_std(), StdOrdering::Acquire);
        assert_eq!(MemoryOrder::Release.to_std(), StdOrdering::Release);
    }

    #[test]
    fn test_spin_lock() {
        let lock = SpinLock::new();
        assert!(!lock.is_locked());
        
        lock.lock();
        assert!(lock.is_locked());
        
        lock.unlock();
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_spin_lock_try() {
        let lock = SpinLock::new();
        assert!(lock.try_lock());
        assert!(!lock.try_lock());
        lock.unlock();
        assert!(lock.try_lock());
        lock.unlock();
    }

    #[test]
    fn test_ticket_lock() {
        let lock = TicketLock::new();
        assert!(!lock.is_locked());
        
        let t1 = lock.lock();
        assert!(lock.is_locked());
        
        lock.unlock();
        assert!(!lock.is_locked());
        
        let t2 = lock.lock();
        assert_ne!(t1, t2);
        lock.unlock();
    }

    #[test]
    fn test_seq_lock() {
        let lock = SeqLock::new();
        
        // Read without write
        let seq = lock.read_begin();
        assert!(!lock.read_retry(seq));
        
        // Write
        lock.write_begin();
        lock.write_end();
        
        // Read after write
        let seq = lock.read_begin();
        assert!(!lock.read_retry(seq));
    }

    #[test]
    fn test_atomic_usize_operations() {
        let mut val: usize = 42;
        let ptr = &mut val as *mut usize;
        
        // Load
        assert_eq!(atomic_load_usize(ptr, MemoryOrder::SeqCst), 42);
        
        // Store
        atomic_store_usize(ptr, 100, MemoryOrder::SeqCst);
        assert_eq!(atomic_load_usize(ptr, MemoryOrder::SeqCst), 100);
        
        // Fetch add
        let old = atomic_fetch_add_usize(ptr, 10, MemoryOrder::SeqCst);
        assert_eq!(old, 100);
        assert_eq!(atomic_load_usize(ptr, MemoryOrder::SeqCst), 110);
        
        // Exchange
        let old = atomic_exchange_usize(ptr, 999, MemoryOrder::SeqCst);
        assert_eq!(old, 110);
        assert_eq!(atomic_load_usize(ptr, MemoryOrder::SeqCst), 999);
    }

    #[test]
    fn test_atomic_compare_exchange() {
        let mut val: usize = 42;
        let ptr = &mut val as *mut usize;
        let mut expected = 42;
        
        // Should succeed
        let success = atomic_compare_exchange_usize(
            ptr, 
            &mut expected, 
            100, 
            MemoryOrder::SeqCst, 
            MemoryOrder::SeqCst
        );
        assert!(success);
        assert_eq!(atomic_load_usize(ptr, MemoryOrder::SeqCst), 100);
        
        // Should fail
        let mut expected = 50;
        let success = atomic_compare_exchange_usize(
            ptr, 
            &mut expected, 
            200, 
            MemoryOrder::SeqCst, 
            MemoryOrder::SeqCst
        );
        assert!(!success);
        assert_eq!(expected, 100); // expected updated to actual value
    }
}
