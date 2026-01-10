# VeZ Standard Library

The VeZ standard library provides essential types, traits, and functions for VeZ programs.

## Structure

```
stdlib/
├── core/           # Core types and traits
│   ├── option.zari
│   ├── result.zari
│   ├── ptr.zari
│   └── ops.zari
├── collections/    # Data structures
│   ├── vec.zari
│   ├── string.zari
│   └── hashmap.zari
├── io/            # Input/Output
│   ├── mod.zari
│   ├── stdio.zari
│   └── file.zari
├── mem/           # Memory management
│   ├── mod.zari
│   └── allocator.zari
├── fmt/           # Formatting
│   ├── mod.zari
│   └── display.zari
└── prelude.zari   # Commonly used items
```

## Core Types

- `Option<T>` - Optional values
- `Result<T, E>` - Error handling
- `Vec<T>` - Dynamic array
- `String` - UTF-8 string
- `Box<T>` - Heap allocation
- `Rc<T>` - Reference counting
- `Arc<T>` - Atomic reference counting

## Traits

- `Clone` - Explicit cloning
- `Copy` - Implicit copying
- `Drop` - Cleanup on destruction
- `Display` - User-facing formatting
- `Debug` - Debug formatting
- `Iterator` - Iteration protocol
- `From/Into` - Type conversions

## Usage

```vex
use std::prelude::*;

fn main() {
    let v = Vec::new();
    v.push(1);
    v.push(2);
    
    let s = String::from("Hello, VeZ!");
    println!("{}", s);
}
```
