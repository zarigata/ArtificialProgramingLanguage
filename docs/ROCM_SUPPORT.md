# VeZ ROCm/AMD GPU Support

Complete support for AMD GPU compute using ROCm (Radeon Open Compute) platform.

## Overview

VeZ provides first-class support for AMD GPUs through the ROCm platform, enabling high-performance compute on:

- **AMD Instinct** - MI100, MI200, MI300 series (CDNA architecture)
- **AMD Radeon Pro** - Professional workstation GPUs
- **AMD Radeon RX** - Consumer gaming GPUs (RDNA architecture)

## Supported Architectures

| Architecture | GPUs | gfx ID | Features |
|--------------|------|--------|----------|
| **CDNA 3** | MI300X, MI300A | gfx940 | MFMA, FP64, BF16 |
| **CDNA 2** | MI250X, MI210 | gfx90a | MFMA, FP64 |
| **CDNA 1** | MI100 | gfx908 | MFMA, FP64 |
| **RDNA 3** | RX 7900 XTX/XT | gfx1100 | SIMD32 |
| **RDNA 2** | RX 6900 XT, RX 6800 | gfx1030 | SIMD32 |
| **RDNA 1** | RX 5700 XT | gfx1010 | SIMD32 |
| **GCN 5** | Radeon VII, MI50 | gfx906 | FP64 |
| **GCN 4** | Vega 56/64 | gfx900 | - |

## Basic Usage

### Simple Vector Add

```zari
@rocm(arch="gfx1100", workgroup=256)
def vector_add(a: &f32, b: &f32, c: &mut f32, n: usize):
    let idx = @global_idx()
    if idx < n:
        c[idx] = a[idx] + b[idx]
```

### Matrix Multiplication

```zari
@rocm(arch="gfx940", workgroup=16, grid=(64, 64))
def matrix_mul(A: &f32, B: &f32, C: &mut f32, M: usize, N: usize, K: usize):
    let row = @global_idx_x()
    let col = @global_idx_y()
    
    if row < M and col < N:
        let mut sum = 0.0
        for k in range(K):
            sum = sum + A[row * K + k] * B[k * N + col]
        C[row * N + col] = sum
```

### Shared Memory (LDS)

```zari
@rocm(arch="gfx940", workgroup=256, shared=4096)
def reduce_sum(data: &f32, result: &mut f32, n: usize):
    let local_idx = @local_idx()
    let global_idx = @global_idx()
    
    @shared_memory
    var shared_data: [f32; 256]
    
    if global_idx < n:
        shared_data[local_idx] = data[global_idx]
    else:
        shared_data[local_idx] = 0.0
    
    @barrier()
    
    var stride = 128
    while stride > 0:
        if local_idx < stride:
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride]
        stride = stride / 2
        @barrier()
    
    if local_idx == 0:
        result[@group_idx()] = shared_data[0]
```

## Annotations

### @rocm

Main annotation for AMD GPU kernels.

```zari
@rocm(
    arch="gfx940",           // Target architecture
    workgroup=256,            // Workgroup size (single dimension)
    workgroup=(16, 16),       // Workgroup size (2D)
    workgroup=(8, 8, 8),      // Workgroup size (3D)
    grid=1024,                // Grid size (single dimension)
    grid=(64, 64),            // Grid size (2D)
    shared=4096,              // Shared memory in bytes
    uses_matrix=true,         // Enable MFMA matrix cores
    vgprs=256,                // VGPR limit
    sgprs=100,                // SGPR limit
)
```

## Intrinsics

### Thread Indexing

| Intrinsic | Description |
|-----------|-------------|
| `@global_idx()` | Global linear index |
| `@global_idx_x()` | Global X index |
| `@global_idx_y()` | Global Y index |
| `@global_idx_z()` | Global Z index |
| `@local_idx()` | Local linear index |
| `@local_idx_x()` | Local X index |
| `@local_idx_y()` | Local Y index |
| `@local_idx_z()` | Local Z index |
| `@group_idx()` | Workgroup index |

### Memory

| Intrinsic | Description |
|-----------|-------------|
| `@shared_memory` | Declare LDS shared memory |
| `@barrier()` | Workgroup barrier |
| `@mem_fence()` | Memory fence |

### Math

| Intrinsic | Description |
|-----------|-------------|
| `@exp(x)` | Exponential |
| `@log(x)` | Natural logarithm |
| `@sqrt(x)` | Square root |
| `@rsqrt(x)` | Reciprocal square root |
| `@sin(x)` | Sine |
| `@cos(x)` | Cosine |
| `@pow(x, y)` | Power |

## Matrix Cores (MFMA)

For CDNA architectures (MI100, MI200, MI300):

```zari
@rocm(arch="gfx940", workgroup=256, uses_matrix=true)
def gemm(M: usize, N: usize, K: usize, alpha: f32, A: &f32, B: &f32, beta: f32, C: &mut f32):
    let row = @global_idx_x()
    let col = @global_idx_y()
    
    if row < M and col < N:
        var sum = 0.0
        for k in range(K):
            sum = sum + A[row * K + k] * B[k * N + col]
        C[row * N + col] = alpha * sum + beta * C[row * N + col]
```

### MFMA Shapes

| Architecture | Supported Shapes |
|--------------|-----------------|
| CDNA 1 | 32x32x1, 16x16x4, 4x4x4 |
| CDNA 2 | 32x32x1, 16x16x4, 4x4x4, 16x16x16 |
| CDNA 3 | 32x32x2, 16x16x4, 4x4x4, 16x16x16 |

## Compilation

```bash
# Compile for AMD MI300
vezc kernel.zari -o kernel --target=rocm --arch=gfx940

# Compile for RX 7900 XTX
vezc kernel.zari -o kernel --target=rocm --arch=gfx1100

# Run with ROCm runtime
./kernel
```

## Optimization

VeZ includes AMD-specific optimizations:

1. **Wavefront Optimization** - Aligns workgroups to wavefront boundaries
2. **LDS Optimization** - Optimizes shared memory bank conflicts
3. **MFMA Optimization** - Uses matrix cores for GEMM operations
4. **Register Allocation** - Optimizes VGPR/SGPR usage for occupancy

### Occupancy Calculator

```zari
let occupancy = @rocm_occupancy(
    arch="gfx940",
    vgprs=128,
    sgprs=32,
    shared_mem=4096
)
@println("Occupancy: {}%", occupancy * 100)
```

## Device Selection

```zari
@entry
def main():
    @rocm_init()
    
    let device_count = @rocm_device_count()
    @println("Found {} AMD GPUs", device_count)
    
    for i in range(device_count):
        let device = @rocm_get_device(i)
        @println("GPU {}: {} ({} CUs)", i, device.name, device.compute_units)
    
    @rocm_cleanup()
    return 0
```

## Memory Management

```zari
@entry
def main():
    @rocm_init()
    
    let n = 1024 * 1024
    
    var a = @gpu_alloc(n * 4)       // Allocate on device
    var b = @gpu_alloc(n * 4)
    var c = @gpu_alloc(n * 4)
    
    @gpu_copy_to_device(a, host_data_a)
    
    let start = @rocm_event()
    vector_add<<<4096, 256>>>(a, b, c, n)
    let end = @rocm_event()
    
    let elapsed = @rocm_elapsed(start, end)
    @println("Kernel took {} ms", elapsed)
    
    @gpu_copy_to_host(host_result, c)
    
    @gpu_free(a)
    @gpu_free(b)
    @gpu_free(c)
    
    @rocm_cleanup()
    return 0
```

## Multi-GPU

```zari
@entry
def main():
    @rocm_init()
    
    let gpu_count = @rocm_device_count()
    
    for gpu in range(gpu_count):
        @rocm_set_device(gpu)
        
        let data = @gpu_alloc(1024 * 1024)
        process_kernel<<<1024, 256>>>(data)
        
        @gpu_free(data)
    
    @rocm_cleanup()
    return 0
```

## Comparison with CUDA

| Feature | CUDA | ROCm/HIP |
|---------|------|----------|
| Thread block | blockDim | hipBlockDim |
| Thread index | threadIdx | hipThreadIdx |
| Block index | blockIdx | hipBlockIdx |
| Grid dimension | gridDim | hipGridDim |
| Shared memory | __shared__ | @shared_memory |
| Barrier | __syncthreads() | @barrier() |
| Wave size | 32 (warp) | 64 (wavefront) |

## Requirements

- ROCm 5.0+ runtime
- HIP runtime libraries
- AMD GPU driver

## Installation

```bash
# Install ROCm on Linux
sudo apt install rocm-hip-sdk

# Verify installation
rocminfo
hipconfig --version
```

## Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [AMD GPU Architecture](https://developer.amd.com/resources/rocm-learning-center/)
