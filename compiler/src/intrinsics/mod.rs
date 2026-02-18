//! AI-Optimized Hardware Intrinsics
//!
//! This module provides direct access to CPU and GPU hardware for AI-generated code.
//! Key features:
//! - SIMD intrinsics (AVX, AVX2, AVX-512, NEON)
//! - GPU compute intrinsics
//! - Memory fence operations
//! - Cache control
//! - Timing primitives

pub mod simd;
pub mod gpu;
pub mod memory;
pub mod timing;
pub mod atomic;

pub use simd::{SimdIntrinsic, VectorWidth};
pub use gpu::{GpuIntrinsic, GpuArch};
pub use memory::{MemoryIntrinsic, CacheHint};
pub use timing::{TimingIntrinsic, Timer};
pub use atomic::{AtomicIntrinsic, MemoryOrder};
