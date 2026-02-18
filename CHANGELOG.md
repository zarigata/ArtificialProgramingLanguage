# Changelog

All notable changes to the VeZ Programming Language will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-02-18

### Added

#### Standard Library Expansion
- **String Module** (`stdlib/string.zari`)
  - StringBuilder class for efficient concatenation
  - 50+ string manipulation functions
  - trim, split, join, format, escape/unescape
  - Levenshtein distance and similarity scoring
  - Type conversions (to_int, to_float, to_bool)

- **DateTime Module** (`stdlib/datetime.zari`)
  - Duration type with time unit conversions
  - Date struct with calendar operations
  - Time struct for time of day
  - DateTime struct combining date and time
  - Stopwatch for measuring elapsed time
  - Format strings (%Y, %m, %d, %H, %M, %S, etc.)
  - Utility functions: age, is_leap_year, days_between

- **JSON Module** (`stdlib/json.zari`)
  - JsonValue class for any JSON type
  - JSON parser with error handling
  - JSON serialization (compact and pretty)
  - JsonBuilder for constructing complex JSON
  - Query and transform functions (map_array, filter_array)
  - Path-based access (get_path)

#### Enhanced Diagnostics
- **Suggestion Engine** (`compiler/src/diagnostics/suggestions.rs`)
  - Intelligent typo correction
  - Type mismatch suggestions with common fixes
  - Import suggestions for missing modules
  - Spell checking using Levenshtein distance
  - Common mistake pattern database

#### Profile-Guided Optimization (PGO)
- **PGO Support** (`compiler/src/pgo/mod.rs`)
  - Profile data collection and storage
  - Function/block execution counting
  - Branch probability tracking
  - Loop iteration profiling
  - Value profiling for indirect calls
  - PGO optimizer with hot/cold code detection
  - Inlining recommendations based on profile data
  - Profile data file I/O (Text and JSON formats)

#### Build Profiles System
- **Build Profiles** (`compiler/src/profiles/mod.rs`)
  - Predefined profiles: debug, release, profiling, bench, min-size
  - Optimization levels (O0, O1, O2, O3, Os, Oz)
  - Debug info levels (none, line-tables, full, maximum)
  - LTO modes (none, thin, full)
  - Panic strategies (unwind, abort)
  - Profile manager with load/save to TOML
  - Build configuration with source/output directories

#### FFI Improvements
- **FFI Module** (`compiler/src/ffi/mod.rs`)
  - C type representation (primitives, pointers, arrays, structs)
  - Type size and alignment calculation
  - VeZ type conversion
  - FFI bindings generation
  - C header file generation
  - C header parser for automatic binding extraction

#### Profiler Integration
- **Profiler Module** (`compiler/src/profiler/mod.rs`)
  - Multiple profiler types (CPU, memory, heap, lock, I/O, GPU)
  - Profile session management
  - Sample recording and statistics
  - Hot function/path detection
  - Memory allocation tracking
  - Report generation (Text, JSON, HTML)
  - Instrumentation helpers

### Dependencies
- Added `serde` and `serde_json` for serialization
- Added `toml` for configuration files

## [0.2.1] - 2025-02-18

### Added

#### Professional Testing Framework
- **Test Runner** (`compiler/src/testing/runner.rs`)
  - Sequential and parallel test execution
  - Test filtering by name and tags
  - Fail-fast and timeout support
  - Configurable thread pool

- **Assertions Library** (`compiler/src/testing/assert.rs`)
  - `assert_eq`, `assert_ne`, `assert_true`, `assert_false`
  - `assert_gt`, `assert_lt`, `assert_ge`, `assert_le`
  - `assert_near` for floating-point comparisons
  - `assert_contains`, `assert_empty`, `assert_len`
  - `assert_ok`, `assert_err`, `assert_some`, `assert_none`
  - `assert_panics`, `assert_no_panic`
  - Fluent assertion API with `expect(value).equal_to(expected)`

- **Mock Objects** (`compiler/src/testing/mock.rs`)
  - Expectation-based mocking
  - Call count verification
  - Mock registry for multiple mocks

- **Test Fixtures** (`compiler/src/testing/fixture.rs`)
  - Lifecycle hooks (setup/teardown)
  - Temporary file and directory management
  - Database, HTTP, and file fixtures

- **Benchmarking** (`compiler/src/testing/bench.rs`)
  - Statistical analysis (mean, median, std dev)
  - Percentile reporting (p10, p25, p50, p75, p90, p95, p99)
  - Benchmark comparison with significance testing
  - Benchmark groups

- **Test Reporting** (`compiler/src/testing/report.rs`)
  - Multiple formats: Plain, Color, JSON, JUnit, TAP, HTML
  - Summary statistics and success rate calculation

#### Code Formatter (vezfmt)
- **Formatter Core** (`compiler/src/formatter/format.rs`)
  - Token-based formatting engine
  - Automatic indentation
  - Operator spacing
  - Brace style options

- **Formatter Configuration** (`compiler/src/formatter/config.rs`)
  - Indent style (tabs or configurable spaces)
  - Line width limits
  - Brace style (same line, next line, always next line)
  - Trailing comma handling
  - Comment wrapping options

- **Indent Management** (`compiler/src/formatter/indent.rs`)
  - Hierarchical indentation tracking
  - Indent/dedent utilities

- **Comment Formatting** (`compiler/src/formatter/comment.rs`)
  - Doc comment normalization
  - Comment wrapping at configurable width

### Dependencies
- Added `crossbeam-channel` for parallel test execution
- Added `num_cpus` for thread pool sizing
- Added `uuid` for unique temp file names

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
