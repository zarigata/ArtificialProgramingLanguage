//! Memory pool for fixed-size allocations

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::marker::PhantomData;
use super::{AllocError, AllocResult, AllocStats};

pub struct MemoryPool<T> {
    blocks: Vec<NonNull<u8>>,
    free_list: *mut FreeNode,
    block_size: usize,
    items_per_block: usize,
    layout: Layout,
    stats: AllocStats,
    _marker: PhantomData<T>,
}

struct FreeNode {
    next: *mut FreeNode,
}

impl<T> MemoryPool<T> {
    pub fn new(items_per_block: usize) -> AllocResult<Self> {
        let item_size = std::mem::size_of::<T>();
        let item_align = std::mem::align_of::<T>();
        
        if item_size == 0 {
            return Err(AllocError::InvalidLayout);
        }
        
        let block_size = item_size * items_per_block;
        let layout = Layout::from_size_align(block_size, item_align)
            .map_err(|_| AllocError::InvalidLayout)?;
        
        Ok(MemoryPool {
            blocks: Vec::new(),
            free_list: std::ptr::null_mut(),
            block_size,
            items_per_block,
            layout,
            stats: AllocStats::new(),
            _marker: PhantomData,
        })
    }

    pub fn allocate(&mut self) -> AllocResult<NonNull<T>> {
        self.stats.total_allocations += 1;
        
        if self.free_list.is_null() {
            self.allocate_block()?;
        }
        
        let node = self.free_list;
        self.free_list = unsafe { (*node).next };
        
        let ptr = NonNull::new(node as *mut T).ok_or(AllocError::InvalidPointer)?;
        self.stats.bytes_allocated += std::mem::size_of::<T>() as u64;
        
        Ok(ptr)
    }

    pub fn deallocate(&mut self, ptr: NonNull<T>) {
        self.stats.total_deallocations += 1;
        self.stats.bytes_freed += std::mem::size_of::<T>() as u64;
        
        let node = ptr.as_ptr() as *mut FreeNode;
        unsafe {
            (*node).next = self.free_list;
        }
        self.free_list = node;
    }

    fn allocate_block(&mut self) -> AllocResult<()> {
        let ptr = unsafe { alloc(self.layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError::OutOfMemory)?;
        
        let item_size = std::mem::size_of::<T>();
        let base = ptr.as_ptr();
        
        for i in 0..self.items_per_block {
            let item_ptr = unsafe { base.add(i * item_size) } as *mut FreeNode;
            unsafe {
                (*item_ptr).next = self.free_list;
            }
            self.free_list = item_ptr;
        }
        
        self.blocks.push(ptr);
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.items_per_block
    }

    pub fn available(&self) -> usize {
        let mut count = 0;
        let mut current = self.free_list;
        while !current.is_null() {
            count += 1;
            current = unsafe { (*current).next };
        }
        count
    }

    pub fn allocated(&self) -> usize {
        self.capacity() - self.available()
    }

    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    pub fn clear(&mut self) {
        self.free_list = std::ptr::null_mut();
        
        for block in self.blocks.drain(..) {
            unsafe {
                dealloc(block.as_ptr().cast(), self.layout);
            }
        }
    }
}

impl<T> Drop for MemoryPool<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

unsafe impl<T: Send> Send for MemoryPool<T> {}
unsafe impl<T: Sync> Sync for MemoryPool<T> {}

/// An object pool that reuses objects
pub struct ObjectPool<T> {
    pool: MemoryPool<T>,
    constructor: fn() -> T,
    reset: Option<fn(&mut T)>,
}

impl<T> ObjectPool<T> {
    pub fn new(items_per_block: usize, constructor: fn() -> T) -> AllocResult<Self> {
        Ok(ObjectPool {
            pool: MemoryPool::new(items_per_block)?,
            constructor,
            reset: None,
        })
    }

    pub fn with_reset(items_per_block: usize, constructor: fn() -> T, reset: fn(&mut T)) -> AllocResult<Self> {
        Ok(ObjectPool {
            pool: MemoryPool::new(items_per_block)?,
            constructor,
            reset: Some(reset),
        })
    }

    pub fn acquire(&mut self) -> AllocResult<PoolObject<T>> {
        let ptr = self.pool.allocate()?;
        
        if self.pool.available() < self.pool.capacity() / 4 {
            unsafe {
                std::ptr::write(ptr.as_ptr(), (self.constructor)());
            }
        } else if let Some(reset) = self.reset {
            unsafe {
                reset(&mut *ptr.as_ptr());
            }
        }
        
        Ok(PoolObject {
            ptr,
            pool: self as *const Self as *mut Self,
        })
    }

    pub fn release(&mut self, ptr: NonNull<T>) {
        self.pool.deallocate(ptr);
    }

    pub fn stats(&self) -> &AllocStats {
        self.pool.stats()
    }
}

/// A pooled object that returns to the pool on drop
pub struct PoolObject<T> {
    ptr: NonNull<T>,
    pool: *mut ObjectPool<T>,
}

impl<T> std::ops::Deref for PoolObject<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> std::ops::DerefMut for PoolObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> Drop for PoolObject<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.pool.is_null() {
                (*self.pool).release(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let mut pool: MemoryPool<u64> = MemoryPool::new(16).unwrap();
        
        let ptr1 = pool.allocate().unwrap();
        let ptr2 = pool.allocate().unwrap();
        
        assert_eq!(pool.allocated(), 2);
        assert_eq!(pool.available(), 14);
        
        pool.deallocate(ptr1);
        pool.deallocate(ptr2);
        
        assert_eq!(pool.allocated(), 0);
        assert_eq!(pool.available(), 16);
    }

    #[test]
    fn test_memory_pool_growth() {
        let mut pool: MemoryPool<u32> = MemoryPool::new(8).unwrap();
        
        for _ in 0..8 {
            pool.allocate().unwrap();
        }
        
        assert_eq!(pool.capacity(), 8);
        
        pool.allocate().unwrap();
        
        assert_eq!(pool.capacity(), 16);
    }

    #[test]
    fn test_object_pool() {
        let mut pool: ObjectPool<Vec<u8>> = ObjectPool::new(4, || Vec::new()).unwrap();
        
        {
            let mut obj = pool.acquire().unwrap();
            obj.push(1);
            obj.push(2);
            assert_eq!(*obj, vec![1, 2]);
        }
        
        assert_eq!(pool.stats().total_allocations, 1);
    }

    #[test]
    fn test_pool_clear() {
        let mut pool: MemoryPool<i32> = MemoryPool::new(4).unwrap();
        
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        
        pool.clear();
        
        assert_eq!(pool.capacity(), 0);
        assert_eq!(pool.available(), 0);
    }
}
