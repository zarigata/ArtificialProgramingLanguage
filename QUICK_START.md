# VeZ Quick Start Guide

## Project Status

**Current Phase**: Phase 0 - Foundation & Planning  
**Status**: Initial architecture and documentation  
**Version**: 0.1.0-alpha (planning)

---

## What is VeZ?

**VeZ** (pronounced "vez") is the world's first **AI-first programming language** - designed from the ground up for AI code generation while maintaining:

- 🚀 **C++ level performance**
- 🛡️ **Rust level memory safety**
- 🎮 **First-class GPU support**
- 🤖 **Optimized for AI comprehension**

---

## Vision

Traditional programming languages were designed for humans. As AI becomes the primary code generator, we need a language that:

1. **AI can understand deeply** - Regular patterns, clear semantics
2. **Compiles to optimal code** - No performance compromises
3. **Guarantees safety** - Memory safe without garbage collection
4. **Accesses hardware directly** - CPU, GPU, memory control
5. **Generates predictable results** - Deterministic compilation

---

## Current State

### ✅ What We Have

- **Comprehensive documentation**
  - Vision and philosophy
  - Technical architecture
  - Language specification (draft)
  - AI integration strategy
  
- **Detailed roadmap**
  - 5-year plan with phases
  - Milestone breakdowns
  - Success metrics
  
- **Research**
  - Competitive analysis
  - Design considerations
  - Naming options

### 🚧 What We're Building

**Phase 1** (Months 4-12): Core compiler
- Lexer and parser
- Type checker
- Code generator (LLVM)
- Basic standard library

**Phase 2** (Months 13-24): Advanced features
- Ownership/borrowing system
- Concurrency primitives
- GPU integration
- Optimizations

**Phase 3** (Months 25-36): AI integration
- Training dataset (100K+ examples)
- Fine-tuned AI models
- AI-powered tools
- Self-hosted compiler

---

## How to Get Involved

### 1. Learn About the Project

Read the documentation:
- [`README.md`](README.md) - Project overview
- [`docs/VISION.md`](docs/VISION.md) - Long-term vision
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Technical design
- [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md) - Development plan

### 2. Join the Discussion

- **GitHub Issues**: Feature proposals, bug reports
- **GitHub Discussions**: Questions, ideas, feedback
- **Community** (coming soon): Discord/forum

### 3. Contribute

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

**Current opportunities**:
- Language design feedback
- Documentation improvements
- Research contributions
- Naming suggestions
- Logo/branding ideas

---

## Key Documents

### For Understanding the Project
- [`docs/VISION.md`](docs/VISION.md) - Why we're building this
- [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) - Language design
- [`research/COMPETITIVE_ANALYSIS.md`](research/COMPETITIVE_ANALYSIS.md) - How we compare

### For Contributing
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - How to contribute
- [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md) - What we're building
- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) - Repository layout

### For Technical Details
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - System design
- [`docs/AI_INTEGRATION.md`](docs/AI_INTEGRATION.md) - AI strategy
- [`roadmap/PHASE_1.md`](roadmap/PHASE_1.md) - First implementation phase

---

## Timeline

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: Planning (Months 1-3)                              │
│ ✅ Architecture ✅ Specification 🚧 Naming                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Core Implementation (Months 4-12)                  │
│ Compiler • Standard Library • Basic Tools                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Advanced Features (Months 13-24)                   │
│ Memory Safety • Concurrency • GPU • Optimizations           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: AI Integration (Months 25-36)                      │
│ Training Data • Fine-tuning • AI Tools • Self-hosting       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Ecosystem (Months 37-48)                           │
│ Package Manager • IDE Support • Community                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Production (Months 49-60)                          │
│ v1.0 Release • Industry Adoption • Research                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Example Code (Conceptual)

Here's what VeZ code looks like (syntax subject to change):

```vex
// Simple function
fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

// GPU kernel
#[kernel]
fn vector_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let idx = thread_idx();
    if idx < a.len() {
        c[idx] = a[idx] + b[idx];
    }
}

// Async function
async fn fetch_data(url: &str) -> Result<String, Error> {
    let response = http::get(url).await?;
    Ok(response.text().await?)
}

// Main function
fn main() {
    println!("Hello from VeZ!");
    
    let result = fibonacci(10);
    println!("Fibonacci(10) = {}", result);
}
```

---

## FAQ

### When can I use this language?

**Phase 1 completion** (Month 12): Basic programs  
**Phase 2 completion** (Month 24): Production-ready  
**Phase 3 completion** (Month 36): AI-optimized

### How is this different from Rust/Mojo/Carbon?

- **vs Rust**: Designed for AI from day one, simpler borrow checker
- **vs Mojo**: Open source, broader focus than just AI/ML
- **vs Carbon**: AI-first, not C++ interop focused

See [`research/COMPETITIVE_ANALYSIS.md`](research/COMPETITIVE_ANALYSIS.md) for details.

### What will the language be called?

Under discussion! See [`docs/NAMING_CONSIDERATIONS.md`](docs/NAMING_CONSIDERATIONS.md).

Top candidates: Synapse, Vexil, Zephyr, Quasar, Forge

### How can AI help with this project?

AI can:
- Generate compiler components
- Create test cases
- Write documentation
- Optimize algorithms
- Eventually: write the compiler in itself!

### Is this open source?

Yes! License to be determined (likely MIT or Apache 2.0).

---

## Next Steps

1. **Read the vision**: [`docs/VISION.md`](docs/VISION.md)
2. **Explore the roadmap**: [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md)
3. **Join discussions**: GitHub Discussions
4. **Contribute**: See [`CONTRIBUTING.md`](CONTRIBUTING.md)
5. **Stay updated**: Watch the repository

---

## Contact

- **GitHub**: [Repository Issues/Discussions]
- **Community**: Coming soon
- **Email**: To be established

---

## Acknowledgments

This project stands on the shoulders of giants:
- **Rust**: Memory safety inspiration
- **LLVM**: Code generation backend
- **Python**: AI/ML ecosystem
- **C++**: Performance standards
- **Go**: Simplicity principles

---

**Let's build the future of AI-generated code together!** 🚀🤖

---

*Last Updated: January 2026*
