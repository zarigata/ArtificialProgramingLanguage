//! Arena allocator for bulk allocations with single deallocation

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::marker::PhantomData;
use super::{AllocError, AllocResult, AllocStats};

const DEFAULT_CHUNK_SIZE: usize = 4096;

struct Chunk {
    ptr: NonNull<u8>,
    size: usize,
    used: usize,
}

impl Chunk {
    fn new(size: usize) -> AllocResult<Self> {
        let layout = Layout::from_size_align(size, 16)
            .map_err(|_| AllocError::InvalidLayout)?;
        
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError::OutOfMemory)?;
        
        Ok(Chunk {
            ptr,
            size,
            used: 0,
        })
    }

    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let align = layout.align();
        let size = layout.size();
        
        let current = self.ptr.as_ptr() as usize + self.used;
        let aligned = (current + align - 1) & !(align - 1);
        let offset = aligned - self.ptr.as_ptr() as usize;
        
        if offset + size > self.size {
            return None;
        }
        
        self.used = offset + size;
        
        let ptr = unsafe { self.ptr.as_ptr().add(offset) };
        NonNull::new(ptr)
    }

    fn remaining(&self) -> usize {
        self.size - self.used
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, 16).unwrap();
        unsafe {
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

unsafe impl Send for Chunk {}

pub struct Arena {
    chunks: Vec<Chunk>,
    chunk_size: usize,
    stats: AllocStats,
}

impl Arena {
    pub fn new() -> Self {
        Arena {
            chunks: Vec::new(),
            chunk_size: DEFAULT_CHUNK_SIZE,
            stats: AllocStats::new(),
        }
    }

    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Arena {
            chunks: Vec::new(),
            chunk_size,
            stats: AllocStats::new(),
        }
    }

    pub fn allocate<T>(&mut self) -> AllocResult<NonNull<T>> {
        let layout = Layout::new::<T>();
        let ptr = self.allocate_layout(layout)?;
        self.stats.total_allocations += 1;
        self.stats.bytes_allocated += layout.size() as u64;
        Ok(ptr.cast())
    }

    pub fn allocate_bytes(&mut self, size: usize, align: usize) -> AllocResult<NonNull<u8>> {
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| AllocError::InvalidLayout)?;
        self.allocate_layout(layout)
    }

    fn allocate_layout(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        if layout.size() > self.chunk_size / 2 {
            let mut chunk = Chunk::new(layout.size().next_power_of_two())?;
            let ptr = chunk.allocate(layout).ok_or(AllocError::OutOfMemory)?;
            self.chunks.push(chunk);
            return Ok(ptr);
        }
        
        if let Some(chunk) = self.chunks.last_mut() {
            if chunk.remaining() >= layout.size() + layout.align() {
                if let Some(ptr) = chunk.allocate(layout) {
                    return Ok(ptr);
                }
            }
        }
        
        let mut chunk = Chunk::new(self.chunk_size)?;
        let ptr = chunk.allocate(layout).ok_or(AllocError::OutOfMemory)?;
        self.chunks.push(chunk);
        Ok(ptr)
    }

    pub fn allocate_slice<T>(&mut self, count: usize) -> AllocResult<NonNull<T>> {
        let layout = Layout::array::<T>(count)
            .map_err(|_| AllocError::SizeTooLarge)?;
        let ptr = self.allocate_layout(layout)?;
        self.stats.total_allocations += 1;
        self.stats.bytes_allocated += layout.size() as u64;
        Ok(ptr.cast())
    }

    pub fn reset(&mut self) {
        self.stats.total_deallocations = self.stats.total_allocations;
        self.stats.bytes_freed = self.stats.bytes_allocated;
        self.stats.total_allocations = 0;
        self.stats.bytes_allocated = 0;
        
        for chunk in &mut self.chunks {
            chunk.used = 0;
        }
    }

    pub fn clear(&mut self) {
        self.stats.total_deallocations = self.stats.total_allocations;
        self.stats.bytes_freed = self.stats.bytes_allocated;
        self.stats.total_allocations = 0;
        self.stats.bytes_allocated = 0;
        self.chunks.clear();
    }

    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    pub fn chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn total_capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.size).sum()
    }

    pub fn total_used(&self) -> usize {
        self.chunks.iter().map(|c| c.used).sum()
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        self.clear();
    }
}

pub struct TypedArena<T> {
    arena: Arena,
    _marker: PhantomData<T>,
}

impl<T> TypedArena<T> {
    pub fn new() -> Self {
        TypedArena {
            arena: Arena::new(),
            _marker: PhantomData,
        }
    }

    pub fn with_chunk_size(chunk_size: usize) -> Self {
        TypedArena {
            arena: Arena::with_chunk_size(chunk_size),
            _marker: PhantomData,
        }
    }

    pub fn alloc(&mut self, value: T) -> &mut T {
        let ptr = self.arena.allocate::<T>().expect("allocation failed");
        unsafe {
            std::ptr::write(ptr.as_ptr(), value);
            &mut *ptr.as_ptr()
        }
    }

    pub fn alloc_slice(&mut self, values: &[T]) -> &mut [T]
    where
        T: Clone,
    {
        let ptr = self.arena.allocate_slice::<T>(values.len())
            .expect("allocation failed");
        
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), values.len());
            for (i, value) in values.iter().enumerate() {
                slice[i] = value.clone();
            }
            slice
        }
    }

    pub fn reset(&mut self) {
        self.arena.reset();
    }

    pub fn stats(&self) -> &AllocStats {
        self.arena.stats()
    }
}

impl<T> Default for TypedArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let mut arena = Arena::new();
        
        let ptr1 = arena.allocate::<u64>().unwrap();
        let ptr2 = arena.allocate::<u32>().unwrap();
        
        assert!(!ptr1.as_ptr().is_null());
        assert!(!ptr2.as_ptr().is_null());
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr() as *mut u64);
    }

    #[test]
    fn test_arena_bytes() {
        let mut arena = Arena::new();
        
        let ptr = arena.allocate_bytes(1024, 16).unwrap();
        assert!(!ptr.as_ptr().is_null());
        
        let aligned = ptr.as_ptr() as usize % 16;
        assert_eq!(aligned, 0);
    }

    #[test]
    fn test_arena_slice() {
        let mut arena = Arena::new();
        
        let ptr = arena.allocate_slice::<u8>(100).unwrap();
        assert!(!ptr.as_ptr().is_null());
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new();
        
        arena.allocate::<u64>().unwrap();
        arena.allocate::<u64>().unwrap();
        
        let used_before = arena.total_used();
        assert!(used_before > 0);
        
        arena.reset();
        
        assert_eq!(arena.total_used(), 0);
    }

    #[test]
    fn test_typed_arena() {
        let mut arena = TypedArena::<String>::new();
        
        let s1 = arena.alloc(String::from("hello"));
        let s2 = arena.alloc(String::from("world"));
        
        assert_eq!(s1, "hello");
        assert_eq!(s2, "world");
    }

    #[test]
    fn test_typed_arena_slice() {
        let mut arena = TypedArena::<i32>::new();
        
        let values = [1, 2, 3, 4, 5];
        let slice = arena.alloc_slice(&values);
        
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_arena_stats() {
        let mut arena = Arena::new();
        
        arena.allocate::<u64>().unwrap();
        arena.allocate::<u32>().unwrap();
        
        assert_eq!(arena.stats().total_allocations, 2);
    }

    #[test]
    fn test_arena_chunk_growth() {
        let mut arena = Arena::with_chunk_size(256);
        
        for _ in 0..100 {
            arena.allocate::<u64>().unwrap();
        }
        
        assert!(arena.chunks() > 1);
    }
}
