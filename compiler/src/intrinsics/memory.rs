//! Memory Intrinsics for direct memory control

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHint {
    Normal,
    Streaming,
    NonTemporal,
    PrefetchL1,
    PrefetchL2,
    PrefetchL3,
    EvictFirst,
    EvictLast,
}

impl fmt::Display for CacheHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheHint::Normal => write!(f, "cache_normal"),
            CacheHint::Streaming => write!(f, "cache_streaming"),
            CacheHint::NonTemporal => write!(f, "cache_nontemporal"),
            CacheHint::PrefetchL1 => write!(f, "prefetch_l1"),
            CacheHint::PrefetchL2 => write!(f, "prefetch_l2"),
            CacheHint::PrefetchL3 => write!(f, "prefetch_l3"),
            CacheHint::EvictFirst => write!(f, "evict_first"),
            CacheHint::EvictLast => write!(f, "evict_last"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOp {
    Fence,
    FenceLoad,
    FenceStore,
    PrefetchRead,
    PrefetchWrite,
    PrefetchReadT0,
    PrefetchReadT1,
    PrefetchReadT2,
    PrefetchReadNTA,
    PrefetchWriteT0,
    PrefetchWriteT1,
    PrefetchWriteT2,
    PrefetchWriteNTA,
    ClFlush,
    ClFlushOpt,
    ClWB,
    LoadUnaligned,
    StoreUnaligned,
    LoadVolatile,
    StoreVolatile,
    MemCopy,
    MemMove,
    MemSet,
    MemCmp,
}

pub struct MemoryIntrinsic {
    pub op: MemoryOp,
    pub hint: Option<CacheHint>,
}

impl MemoryIntrinsic {
    pub fn new(op: MemoryOp) -> Self {
        MemoryIntrinsic { op, hint: None }
    }
    
    pub fn with_hint(op: MemoryOp, hint: CacheHint) -> Self {
        MemoryIntrinsic { op, hint: Some(hint) }
    }
    
    pub fn to_llvm(&self) -> String {
        match self.op {
            MemoryOp::Fence => "llvm.memory.barrier".to_string(),
            MemoryOp::PrefetchRead => "llvm.prefetch".to_string(),
            MemoryOp::ClFlush => "llvm.x86.clflush".to_string(),
            MemoryOp::MemCopy => "llvm.memcpy".to_string(),
            MemoryOp::MemSet => "llvm.memset".to_string(),
            _ => format!("{:?}", self.op),
        }
    }
}

#[inline]
pub fn mem_fence() {
    unsafe { std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst) }
}

#[inline]
pub fn mem_fence_load() {
    unsafe { std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire) }
}

#[inline]
pub fn mem_fence_store() {
    unsafe { std::sync::atomic::fence(std::sync::atomic::Ordering::Release) }
}

#[inline]
pub fn prefetch_read<T>(ptr: *const T) {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }
}

#[inline]
pub fn prefetch_write<T>(ptr: *mut T) {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }
}

#[inline]
pub fn cache_line_size() -> usize {
    64
}

#[inline]
pub fn align_to_cache<T>(ptr: *const T) -> usize {
    let addr = ptr as usize;
    let cache_line = cache_line_size();
    (addr + cache_line - 1) & !(cache_line - 1)
}

pub fn load_unaligned<T: Copy>(ptr: *const T) -> T {
    unsafe { std::ptr::read_unaligned(ptr) }
}

pub fn store_unaligned<T: Copy>(ptr: *mut T, val: T) {
    unsafe { std::ptr::write_unaligned(ptr, val) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hint_display() {
        assert_eq!(CacheHint::PrefetchL1.to_string(), "prefetch_l1");
    }

    #[test]
    fn test_memory_intrinsic() {
        let intr = MemoryIntrinsic::new(MemoryOp::Fence);
        assert_eq!(intr.to_llvm(), "llvm.memory.barrier");
    }

    #[test]
    fn test_cache_line_size() {
        assert!(cache_line_size() >= 32);
    }
}
