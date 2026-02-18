//! GPU Intrinsics for direct GPU access

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArch {
    NvidiaSm70,
    NvidiaSm75,
    NvidiaSm80,
    NvidiaSm90,
    AmdGfx900,
    AmdGfx906,
    AmdGfx908,
    AmdGfx90a,
    AmdGfx1030,
    AmdGfx1100,
    IntelXe,
    AppleM1,
    AppleM2,
    AppleM3,
}

impl GpuArch {
    pub fn warp_size(&self) -> u32 {
        match self {
            GpuArch::NvidiaSm70 | GpuArch::NvidiaSm75 | 
            GpuArch::NvidiaSm80 | GpuArch::NvidiaSm90 => 32,
            GpuArch::AmdGfx900 | GpuArch::AmdGfx906 | GpuArch::AmdGfx908 |
            GpuArch::AmdGfx90a | GpuArch::AmdGfx1030 | GpuArch::AmdGfx1100 => 64,
            GpuArch::IntelXe => 16,
            GpuArch::AppleM1 | GpuArch::AppleM2 | GpuArch::AppleM3 => 32,
        }
    }
    
    pub fn max_threads_per_block(&self) -> u32 {
        match self {
            GpuArch::NvidiaSm80 | GpuArch::NvidiaSm90 => 1536,
            GpuArch::NvidiaSm70 | GpuArch::NvidiaSm75 => 1024,
            GpuArch::AmdGfx90a | GpuArch::AmdGfx1100 => 1024,
            _ => 1024,
        }
    }
    
    pub fn shared_memory_size(&self) -> u64 {
        match self {
            GpuArch::NvidiaSm90 => 227328,
            GpuArch::NvidiaSm80 => 163840,
            GpuArch::AmdGfx90a => 65536,
            _ => 49152,
        }
    }
    
    pub fn supports_fp16(&self) -> bool {
        true
    }
    
    pub fn supports_fp64(&self) -> bool {
        matches!(self, 
            GpuArch::NvidiaSm70 | GpuArch::NvidiaSm75 | 
            GpuArch::NvidiaSm80 | GpuArch::NvidiaSm90 |
            GpuArch::AmdGfx906 | GpuArch::AmdGfx908 | GpuArch::AmdGfx90a
        )
    }
    
    pub fn supports_tensor_cores(&self) -> bool {
        matches!(self,
            GpuArch::NvidiaSm70 | GpuArch::NvidiaSm75 |
            GpuArch::NvidiaSm80 | GpuArch::NvidiaSm90
        )
    }
    
    pub fn supports_mfma(&self) -> bool {
        matches!(self,
            GpuArch::AmdGfx908 | GpuArch::AmdGfx90a | GpuArch::AmdGfx1100
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOp {
    ThreadIdx,
    BlockIdx,
    BlockDim,
    GridDim,
    SyncThreads,
    SyncWarp,
    Barrier,
    AtomicAdd,
    AtomicSub,
    AtomicExch,
    AtomicMin,
    AtomicMax,
    AtomicInc,
    AtomicDec,
    AtomicCAS,
    SharedMemory,
    GlobalMemory,
    ConstantMemory,
    TextureMemory,
    LdGlobal,
    StGlobal,
    LdShared,
    StShared,
    Prefetch,
    MemFence,
    ShuffleXor,
    ShuffleUp,
    ShuffleDown,
    ShuffleIdx,
    VoteAll,
    VoteAny,
    VoteBallot,
    ReduceAdd,
    ReduceMin,
    ReduceMax,
    ScanExclusive,
    ScanInclusive,
    Histogram,
    Sort,
    RadixSort,
    MatrixMul,
    Convolution,
    ReductionTree,
}

impl fmt::Display for GpuOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuOp::ThreadIdx => write!(f, "threadIdx"),
            GpuOp::BlockIdx => write!(f, "blockIdx"),
            GpuOp::BlockDim => write!(f, "blockDim"),
            GpuOp::GridDim => write!(f, "gridDim"),
            GpuOp::SyncThreads => write!(f, "__syncthreads"),
            GpuOp::SyncWarp => write!(f, "__syncwarp"),
            GpuOp::AtomicAdd => write!(f, "atomicAdd"),
            GpuOp::AtomicCAS => write!(f, "atomicCAS"),
            GpuOp::SharedMemory => write!(f, "__shared__"),
            GpuOp::ShuffleXor => write!(f, "__shfl_xor_sync"),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub struct GpuIntrinsic {
    pub op: GpuOp,
    pub arch: GpuArch,
}

impl GpuIntrinsic {
    pub fn new(op: GpuOp, arch: GpuArch) -> Self {
        GpuIntrinsic { op, arch }
    }
    
    pub fn to_cuda(&self) -> String {
        match self.op {
            GpuOp::ThreadIdx => "threadIdx.x".to_string(),
            GpuOp::BlockIdx => "blockIdx.x".to_string(),
            GpuOp::BlockDim => "blockDim.x".to_string(),
            GpuOp::GridDim => "gridDim.x".to_string(),
            GpuOp::SyncThreads => "__syncthreads()".to_string(),
            GpuOp::SyncWarp => "__syncwarp()".to_string(),
            GpuOp::AtomicAdd => "atomicAdd".to_string(),
            GpuOp::ShuffleXor => "__shfl_xor_sync".to_string(),
            _ => format!("{:?}", self.op),
        }
    }
    
    pub fn to_hip(&self) -> String {
        match self.op {
            GpuOp::ThreadIdx => "hipThreadIdx_x".to_string(),
            GpuOp::BlockIdx => "hipBlockIdx_x".to_string(),
            GpuOp::BlockDim => "hipBlockDim_x".to_string(),
            GpuOp::GridDim => "hipGridDim_x".to_string(),
            GpuOp::SyncThreads => "__syncthreads()".to_string(),
            _ => self.to_cuda(),
        }
    }
    
    pub fn to_metal(&self) -> String {
        match self.op {
            GpuOp::ThreadIdx => "thread_position_in_threadgroup".to_string(),
            GpuOp::BlockIdx => "threadgroup_position_in_grid".to_string(),
            GpuOp::BlockDim => "threads_per_threadgroup".to_string(),
            GpuOp::SyncThreads => "threadgroup_barrier(mem_flags::mem_threadgroup)".to_string(),
            _ => format!("{:?}", self.op),
        }
    }
}

pub fn gpu_thread_idx() -> u32 {
    0
}

pub fn gpu_block_idx() -> u32 {
    0
}

pub fn gpu_sync_threads() {
}

pub fn gpu_warp_shuffle_xor(value: u32, mask: u32) -> u32 {
    value ^ mask
}

pub fn gpu_atomic_add(ptr: *mut i32, value: i32) -> i32 {
    unsafe {
        let old = *ptr;
        *ptr = old.wrapping_add(value);
        old
    }
}

pub fn gpu_shared_array<T: Copy, const N: usize>() -> [T; N] {
    unsafe { std::mem::zeroed() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_arch() {
        let arch = GpuArch::NvidiaSm80;
        assert_eq!(arch.warp_size(), 32);
        assert!(arch.supports_tensor_cores());
    }

    #[test]
    fn test_amd_arch() {
        let arch = GpuArch::AmdGfx90a;
        assert_eq!(arch.warp_size(), 64);
        assert!(arch.supports_mfma());
    }

    #[test]
    fn test_gpu_intrinsic() {
        let intr = GpuIntrinsic::new(GpuOp::ThreadIdx, GpuArch::NvidiaSm80);
        assert_eq!(intr.to_cuda(), "threadIdx.x");
    }
}
