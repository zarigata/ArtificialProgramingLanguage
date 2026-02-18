//! Region-Based Memory Allocator
//!
//! Provides memory safety without garbage collection using lexical scoping.
//! Memory is allocated within regions and automatically freed when the region exits.

pub mod region;
pub mod pool;
pub mod arena;

pub use region::{Region, RegionGuard, RegionId};
pub use pool::MemoryPool;
pub use arena::Arena;

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::fmt;

/// Allocation statistics
#[derive(Debug, Clone, Default)]
pub struct AllocStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_freed: u64,
    pub peak_usage: u64,
    pub active_regions: u64,
}

impl AllocStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn current_usage(&self) -> u64 {
        self.bytes_allocated.saturating_sub(self.bytes_freed)
    }

    pub fn allocation_count(&self) -> u64 {
        self.total_allocations
    }

    pub fn fragmentation_ratio(&self) -> f64 {
        let current = self.current_usage();
        if self.peak_usage == 0 {
            0.0
        } else {
            1.0 - (current as f64 / self.peak_usage as f64)
        }
    }
}

/// Error type for allocation failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    OutOfMemory,
    InvalidLayout,
    RegionNotFound,
    InvalidPointer,
    DoubleFree,
    SizeTooLarge,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::OutOfMemory => write!(f, "out of memory"),
            AllocError::InvalidLayout => write!(f, "invalid memory layout"),
            AllocError::RegionNotFound => write!(f, "region not found"),
            AllocError::InvalidPointer => write!(f, "invalid pointer"),
            AllocError::DoubleFree => write!(f, "double free detected"),
            AllocError::SizeTooLarge => write!(f, "allocation size too large"),
        }
    }
}

impl std::error::Error for AllocError {}

/// Allocation result type
pub type AllocResult<T> = std::result::Result<T, AllocError>;

/// A raw allocation block
#[derive(Debug)]
pub struct Block {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl Block {
    pub fn new(layout: Layout) -> AllocResult<Self> {
        let ptr = unsafe { alloc(layout) };
        NonNull::new(ptr)
            .map(|ptr| Block { ptr, layout })
            .ok_or(AllocError::OutOfMemory)
    }

    pub fn from_raw(ptr: *mut u8, layout: Layout) -> AllocResult<Self> {
        NonNull::new(ptr)
            .map(|ptr| Block { ptr, layout })
            .ok_or(AllocError::InvalidPointer)
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

unsafe impl Send for Block {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_stats() {
        let mut stats = AllocStats::new();
        stats.total_allocations = 100;
        stats.total_deallocations = 50;
        stats.bytes_allocated = 10000;
        stats.bytes_freed = 5000;
        stats.peak_usage = 8000;

        assert_eq!(stats.current_usage(), 5000);
        assert_eq!(stats.allocation_count(), 100);
        assert!((stats.fragmentation_ratio() - 0.375).abs() < 0.001);
    }

    #[test]
    fn test_alloc_error_display() {
        assert_eq!(AllocError::OutOfMemory.to_string(), "out of memory");
        assert_eq!(AllocError::InvalidPointer.to_string(), "invalid pointer");
    }

    #[test]
    fn test_block_allocation() {
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let block = Block::new(layout).unwrap();
        
        assert_eq!(block.size(), 1024);
        assert!(!block.ptr().is_null());
    }
}
