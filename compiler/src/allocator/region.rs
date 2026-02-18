//! Memory region management for scoped allocations

use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::cell::RefCell;
use std::rc::Rc;
use super::{AllocError, AllocResult, AllocStats};

/// Unique identifier for a memory region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u64);

impl std::fmt::Display for RegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "region#{}", self.0)
    }
}

/// A memory region that owns allocations
pub struct Region {
    id: RegionId,
    name: String,
    allocations: HashMap<usize, Allocation>,
    parent: Option<RegionId>,
    stats: AllocStats,
}

#[derive(Debug)]
struct Allocation {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl Region {
    pub fn new(id: RegionId, name: &str) -> Self {
        Region {
            id,
            name: name.to_string(),
            allocations: HashMap::new(),
            parent: None,
            stats: AllocStats::new(),
        }
    }

    pub fn with_parent(id: RegionId, name: &str, parent: RegionId) -> Self {
        Region {
            id,
            name: name.to_string(),
            allocations: HashMap::new(),
            parent: Some(parent),
            stats: AllocStats::new(),
        }
    }

    pub fn id(&self) -> RegionId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn parent(&self) -> Option<RegionId> {
        self.parent
    }

    pub fn allocate(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        let ptr = unsafe { alloc(layout) };
        
        let ptr = NonNull::new(ptr).ok_or(AllocError::OutOfMemory)?;
        
        let key = ptr.as_ptr() as usize;
        self.allocations.insert(key, Allocation { ptr, layout });
        
        self.stats.total_allocations += 1;
        self.stats.bytes_allocated += layout.size() as u64;
        
        let current = self.stats.bytes_allocated - self.stats.bytes_freed;
        if current > self.stats.peak_usage {
            self.stats.peak_usage = current;
        }
        
        Ok(ptr)
    }

    pub fn deallocate(&mut self, ptr: NonNull<u8>) -> AllocResult<()> {
        let key = ptr.as_ptr() as usize;
        
        let allocation = self.allocations.remove(&key)
            .ok_or(AllocError::InvalidPointer)?;
        
        self.stats.total_deallocations += 1;
        self.stats.bytes_freed += allocation.layout.size() as u64;
        
        unsafe {
            dealloc(ptr.as_ptr(), allocation.layout);
        }
        
        Ok(())
    }

    pub fn contains(&self, ptr: NonNull<u8>) -> bool {
        self.allocations.contains_key(&(ptr.as_ptr() as usize))
    }

    pub fn clear(&mut self) {
        for (_, allocation) in self.allocations.drain() {
            self.stats.total_deallocations += 1;
            self.stats.bytes_freed += allocation.layout.size() as u64;
            unsafe {
                dealloc(allocation.ptr.as_ptr(), allocation.layout);
            }
        }
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Guard for automatic region cleanup
pub struct RegionGuard {
    manager: Rc<RefCell<RegionManager>>,
    id: RegionId,
}

impl RegionGuard {
    pub fn id(&self) -> RegionId {
        self.id
    }
}

impl Drop for RegionGuard {
    fn drop(&mut self) {
        self.manager.borrow_mut().exit_region(self.id);
    }
}

/// Manages a hierarchy of memory regions
pub struct RegionManager {
    regions: HashMap<RegionId, Region>,
    current: Option<RegionId>,
    next_id: u64,
    global_stats: AllocStats,
}

impl RegionManager {
    pub fn new() -> Self {
        RegionManager {
            regions: HashMap::new(),
            current: None,
            next_id: 1,
            global_stats: AllocStats::new(),
        }
    }

    pub fn create_region(&mut self, name: &str) -> RegionId {
        let id = RegionId(self.next_id);
        self.next_id += 1;
        
        let region = match self.current {
            Some(parent) => Region::with_parent(id, name, parent),
            None => Region::new(id, name),
        };
        
        self.global_stats.active_regions += 1;
        self.regions.insert(id, region);
        
        id
    }

    pub fn enter_region(&mut self, name: &str) -> (RegionId, RegionGuard) {
        let id = self.create_region(name);
        self.current = Some(id);
        
        let guard = RegionGuard {
            manager: Rc::new(RefCell::new(RegionManager {
                regions: std::mem::take(&mut self.regions),
                current: self.current,
                next_id: self.next_id,
                global_stats: std::mem::take(&mut self.global_stats),
            })),
            id,
        };
        
        // For simplicity, we'll just return the ID and a simplified guard
        // In a real implementation, this would use proper Rc<RefCell<>> sharing
        (id, guard)
    }

    pub fn exit_region(&mut self, id: RegionId) {
        if let Some(region) = self.regions.remove(&id) {
            self.global_stats.bytes_freed += region.stats.bytes_allocated - region.stats.bytes_freed;
            self.global_stats.total_deallocations += region.stats.total_allocations;
            self.global_stats.active_regions = self.global_stats.active_regions.saturating_sub(1);
        }
        
        if self.current == Some(id) {
            self.current = self.regions.keys().next().copied();
        }
    }

    pub fn allocate(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        let region_id = self.current.ok_or(AllocError::RegionNotFound)?;
        let region = self.regions.get_mut(&region_id).ok_or(AllocError::RegionNotFound)?;
        region.allocate(layout)
    }

    pub fn deallocate(&mut self, ptr: NonNull<u8>) -> AllocResult<()> {
        for region in self.regions.values_mut() {
            if region.contains(ptr) {
                return region.deallocate(ptr);
            }
        }
        Err(AllocError::InvalidPointer)
    }

    pub fn current_region(&self) -> Option<RegionId> {
        self.current
    }

    pub fn get_region(&self, id: RegionId) -> Option<&Region> {
        self.regions.get(&id)
    }

    pub fn get_region_mut(&mut self, id: RegionId) -> Option<&mut Region> {
        self.regions.get_mut(&id)
    }

    pub fn stats(&self) -> &AllocStats {
        &self.global_stats
    }

    pub fn region_count(&self) -> usize {
        self.regions.len()
    }
}

impl Default for RegionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro for creating a scoped region
#[macro_export]
macro_rules! region {
    ($manager:expr, $name:expr, $body:block) => {
        {
            let (region_id, _guard) = $manager.enter_region($name);
            let _ = region_id;
            $body
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_creation() {
        let region = Region::new(RegionId(1), "test");
        assert_eq!(region.id(), RegionId(1));
        assert_eq!(region.name(), "test");
        assert!(region.parent().is_none());
    }

    #[test]
    fn test_region_allocation() {
        let mut region = Region::new(RegionId(1), "test");
        let layout = Layout::from_size_align(1024, 8).unwrap();
        
        let ptr = region.allocate(layout).unwrap();
        assert!(!ptr.as_ptr().is_null());
        assert_eq!(region.allocation_count(), 1);
        
        region.deallocate(ptr).unwrap();
        assert_eq!(region.allocation_count(), 0);
    }

    #[test]
    fn test_region_clear() {
        let mut region = Region::new(RegionId(1), "test");
        let layout = Layout::from_size_align(1024, 8).unwrap();
        
        region.allocate(layout).unwrap();
        region.allocate(layout).unwrap();
        assert_eq!(region.allocation_count(), 2);
        
        region.clear();
        assert_eq!(region.allocation_count(), 0);
    }

    #[test]
    fn test_region_manager() {
        let mut manager = RegionManager::new();
        
        let id = manager.create_region("root");
        manager.current = Some(id);
        
        let layout = Layout::from_size_align(256, 8).unwrap();
        let ptr = manager.allocate(layout).unwrap();
        
        assert!(!ptr.as_ptr().is_null());
        assert_eq!(manager.region_count(), 1);
        
        manager.exit_region(id);
        assert_eq!(manager.region_count(), 0);
    }

    #[test]
    fn test_region_id_display() {
        let id = RegionId(42);
        assert_eq!(format!("{}", id), "region#42");
    }
}
