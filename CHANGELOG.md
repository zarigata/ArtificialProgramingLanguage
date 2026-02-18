# Changelog

All notable changes to the VeZ Programming Language will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-02-17

### Added

#### Compiler Optimizations
- **Loop Unrolling Pass** (`compiler/src/optimizer/loop_unroll.rs`)
  - Automatic loop unrolling for constant trip counts
  - Strength reduction transformations
  - Loop-invariant code motion

- **SIMD Vectorizer** (`compiler/src/optimizer/vectorizer.rs`)
  - Auto-vectorization for inner loops
  - SLP (Superword Level Parallelism) vectorization
  - Support for AVX, AVX2, AVX-512, and NEON instruction sets

- **Devirtualization Pass** (`compiler/src/optimizer/devirtualizer.rs`)
  - Virtual call to direct call optimization
  - Guarded devirtualization with inline caching
  - Class hierarchy analysis for speculative optimization

- **Escape Analysis** (`compiler/src/optimizer/escape_analysis.rs`)
  - Stack allocation of heap objects
  - Scalar replacement of aggregates
  - Lock elision optimization

#### Effect System
- **Effect Types** (`compiler/src/effects/effect.rs`)
  - IO, State, Async, Exception, GPU, Allocation effects
  - Effect sets with subtyping

- **Effect Inference** (`compiler/src/effects/inference.rs`)
  - Automatic effect inference for expressions
  - Function effect inference from body

- **Effect Checker** (`compiler/src/effects/checker.rs`)
  - Effect subtyping validation
  - Unhandled effect detection

#### Dependent Types
- **Type-Level Naturals** (`compiler/src/dependent_types/types.rs`)
  - Nat types for compile-time integers
  - Ranged integer types
  - Sized array types

- **Dependent Type Checker** (`compiler/src/dependent_types/checker.rs`)
  - Constraint solving for type-level expressions
  - Bounds checking at compile time

- **Dependent Type Inference** (`compiler/src/dependent_types/inference.rs`)
  - Index constraint inference for arrays
  - Range inference for arithmetic operations

#### Hardware Intrinsics
- **SIMD Intrinsics** (`compiler/src/intrinsics/simd.rs`)
  - AVX, AVX2, AVX-512 intrinsics for x86
  - NEON intrinsics for ARM
  - Portable SIMD operations

- **GPU Intrinsics** (`compiler/src/intrinsics/gpu.rs`)
  - CUDA kernel launch intrinsics
  - ROCm/HIP intrinsics for AMD GPUs
  - Metal shader intrinsics for Apple Silicon

- **Memory Intrinsics** (`compiler/src/intrinsics/memory.rs`)
  - Memory fences (SeqCst, Acquire, Release)
  - Cache control (prefetch, clflush)
  - Non-temporal memory operations

- **Timing Intrinsics** (`compiler/src/intrinsics/timing.rs`)
  - High-precision timing (rdtsc)
  - Deadline checking
  - Spin loop hints

- **Atomic Intrinsics** (`compiler/src/intrinsics/atomic.rs`)
  - All atomic operations (load, store, CAS, fetch-add, etc.)
  - Memory ordering support
  - SpinLock, TicketLock, SeqLock implementations

#### Deterministic Execution Scheduler
- **Real-time Scheduler** (`compiler/src/scheduler/realtime.rs`)
  - Earliest Deadline First (EDF) policy
  - Rate Monotonic (RM) policy
  - Deadline Monotonic (DM) policy
  - Fixed Priority (FP) policy
  - Schedulability analysis

- **Deadline Management** (`compiler/src/scheduler/deadline.rs`)
  - Hard, soft, and firm deadlines
  - Deadline monitoring
  - Utilization factor calculation

- **Priority Management** (`compiler/src/scheduler/priority.rs`)
  - Priority inheritance protocol
  - Priority ceiling protocol
  - Support for preventing priority inversion

#### Region-Based Memory Allocator
- **Memory Regions** (`compiler/src/allocator/region.rs`)
  - Scoped memory allocation
  - Automatic cleanup on region exit
  - Hierarchical region support

- **Memory Pools** (`compiler/src/allocator/pool.rs`)
  - Fixed-size allocation pools
  - Object pool with reset support
  - Pool object RAII guards

- **Arena Allocator** (`compiler/src/allocator/arena.rs`)
  - Bulk allocation with single deallocation
  - Typed arenas for specific types
  - Reset capability for reuse

#### Documentation Generator
- **Doc Extractor** (`compiler/src/docgen/extract.rs`)
  - Extract documentation from VeZ source
  - Parse function, struct, enum, const docs
  - Support for @param, @return, @example tags

- **HTML Renderer** (`compiler/src/docgen/html.rs`)
  - Full HTML documentation generation
  - CSS theming support
  - Cross-referencing

- **Markdown Renderer** (`compiler/src/docgen/markdown.rs`)
  - Markdown output for GitHub/GitLab
  - Table of contents generation

#### Compile-Time Reflection
- **Reflection System** (`compiler/src/macro_system/reflection.rs`)
  - Type information at compile time
  - Derive macro generation
  - Struct/enum field iteration

### Changed
- Updated `compiler/src/ir/types.rs` with `Vector` type variant
- Updated `compiler/src/optimizer/mod.rs` to export new optimization passes
- Updated `compiler/src/lib.rs` to include new modules

### Dependencies
- No new external dependencies

## [0.1.0] - 2025-01-10

### Added
- Initial project structure
- Lexer implementation
- Parser implementation
- AST definitions
- Basic type system
- Error handling infrastructure
- LSP server
- REPL tool
- Package manager (vpm)
- VS Code extension
- Benchmark suite
- ROCm/AMD GPU support
- Standard library I/O (buffered, net, fs)
- Standard library sync (mutex, rwlock, channel, atomic)
- AI context extraction system

[0.2.0]: https://github.com/vezz-lang/vezz/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vezz-lang/vezz/releases/tag/v0.1.0
