# VeZ Memory Model

## Overview

VeZ's memory model ensures memory safety without garbage collection through compile-time ownership and borrowing rules. This document specifies the complete memory model.

---

## Memory Regions

### Stack

**Characteristics**:
- Fast allocation/deallocation (pointer bump)
- Automatic cleanup (RAII)
- Fixed size at compile time
- Thread-local
- Limited size (typically 1-8 MB)

**Usage**:
```vex
fn example() {
    let x = 42;           // Stack allocated
    let arr = [1, 2, 3];  // Stack allocated
    let tuple = (1, 2);   // Stack allocated
}  // All automatically deallocated
```

### Heap

**Characteristics**:
- Dynamic allocation
- Manual control via ownership
- Unlimited size (system memory)
- Shared across threads (with synchronization)
- Slower than stack

**Usage**:
```vex
fn example() {
    let boxed = Box::new(42);        // Heap allocated
    let vec = Vec::new();            // Heap allocated
    let string = String::from("hi"); // Heap allocated
}  // All automatically deallocated via Drop
```

### Static/Global

**Characteristics**:
- Program lifetime
- Fixed address
- Initialized before main()
- Immutable by default

**Usage**:
```vex
static GLOBAL: i32 = 42;
static mut MUTABLE: i32 = 0;  // Unsafe to access

const CONSTANT: i32 = 100;  // Inlined, not in memory
```

### GPU Memory

**Characteristics**:
- Separate address space
- Explicit transfers required
- High bandwidth
- Various memory types (global, shared, local)

**Usage**:
```vex
let device_buffer = DeviceBuffer::new(size);
device_buffer.copy_from_host(&host_data);
```

---

## Ownership System

### Ownership Rules

1. **Each value has exactly one owner**
2. **When the owner goes out of scope, the value is dropped**
3. **Ownership can be transferred (moved)**

### Move Semantics

```vex
let s1 = String::from("hello");
let s2 = s1;  // s1 moved to s2, s1 is now invalid

// println!("{}", s1);  // ERROR: value borrowed after move
println!("{}", s2);     // OK
```

### Copy Types

Types that implement `Copy` are copied instead of moved:

```vex
let x = 42;
let y = x;  // x is copied, both x and y are valid

println!("{}, {}", x, y);  // OK
```

**Copy types**:
- All primitive types (integers, floats, bool, char)
- Tuples of Copy types
- Arrays of Copy types
- References

**Non-Copy types**:
- String, Vec, Box (heap allocated)
- Types with custom Drop
- Types containing non-Copy fields

---

## Borrowing System

### Borrowing Rules

1. **At any time, you can have either:**
   - One mutable reference (`&mut T`)
   - Any number of immutable references (`&T`)
2. **References must always be valid (no dangling pointers)**

### Immutable Borrows

```vex
let s = String::from("hello");
let r1 = &s;  // OK
let r2 = &s;  // OK - multiple immutable borrows allowed

println!("{}, {}", r1, r2);
```

### Mutable Borrows

```vex
let mut s = String::from("hello");
let r = &mut s;  // OK

// let r2 = &mut s;  // ERROR: cannot borrow as mutable more than once
// let r3 = &s;      // ERROR: cannot borrow as immutable while mutable exists

r.push_str(" world");
println!("{}", r);
```

### Borrow Scope

```vex
let mut s = String::from("hello");

{
    let r = &mut s;
    r.push_str(" world");
}  // r goes out of scope

let r2 = &s;  // OK - no active mutable borrow
println!("{}", r2);
```

### Non-Lexical Lifetimes (NLL)

```vex
let mut s = String::from("hello");

let r1 = &s;
let r2 = &s;
println!("{}, {}", r1, r2);
// r1 and r2 are no longer used after this point

let r3 = &mut s;  // OK - immutable borrows ended
r3.push_str(" world");
```

---

## Lifetimes

### Lifetime Basics

Lifetimes ensure references are valid:

```vex
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

let s1 = String::from("long string");
let result;
{
    let s2 = String::from("short");
    result = longest(&s1, &s2);  // ERROR: s2 doesn't live long enough
}
```

### Lifetime Elision

Compiler infers lifetimes in common cases:

```vex
// Written:
fn first_word(s: &str) -> &str

// Compiler expands to:
fn first_word<'a>(s: &'a str) -> &'a str
```

### Static Lifetime

```vex
let s: &'static str = "I live forever";

// String literals have 'static lifetime
static GLOBAL: &str = "Global string";
```

### Lifetime Bounds

```vex
struct Ref<'a, T: 'a> {
    value: &'a T,
}

impl<'a, T: 'a> Ref<'a, T> {
    fn new(value: &'a T) -> Self {
        Ref { value }
    }
}
```

---

## Drop and RAII

### Automatic Cleanup

```vex
{
    let s = String::from("hello");
    // Use s
}  // s.drop() called automatically
```

### Drop Order

1. Variables dropped in reverse order of declaration
2. Struct fields dropped in declaration order
3. Tuple elements dropped in order

```vex
{
    let a = String::from("a");
    let b = String::from("b");
    let c = String::from("c");
}  // Dropped in order: c, b, a
```

### Custom Drop

```vex
struct Resource {
    id: i32,
}

impl Drop for Resource {
    fn drop(&mut self) {
        println!("Releasing resource {}", self.id);
    }
}
```

### Manual Drop

```vex
let x = String::from("hello");
drop(x);  // Explicitly drop
// println!("{}", x);  // ERROR: value used after drop
```

---

## Memory Layout

### Struct Layout

```vex
struct Point {
    x: f64,  // 8 bytes, aligned to 8
    y: f64,  // 8 bytes, aligned to 8
}
// Size: 16 bytes, alignment: 8
```

### Padding

```vex
struct Mixed {
    a: u8,   // 1 byte
    // 3 bytes padding
    b: u32,  // 4 bytes
    c: u16,  // 2 bytes
    // 2 bytes padding
}
// Size: 12 bytes, alignment: 4
```

### Packed Structs

```vex
#[repr(packed)]
struct Packed {
    a: u8,
    b: u32,
    c: u16,
}
// Size: 7 bytes, alignment: 1
```

### Enum Layout

```vex
enum Message {
    Quit,                       // 0 bytes
    Move { x: i32, y: i32 },   // 8 bytes
    Write(String),              // 24 bytes
}
// Size: 24 + discriminant (usually 1-4 bytes)
```

---

## Smart Pointers

### Box<T>

Heap allocation with single ownership:

```vex
let b = Box::new(42);
// Allocated on heap, dropped when b goes out of scope
```

### Rc<T>

Reference counting for shared ownership:

```vex
use std::rc::Rc;

let a = Rc::new(42);
let b = Rc::clone(&a);  // Reference count: 2
let c = Rc::clone(&a);  // Reference count: 3

drop(b);  // Reference count: 2
drop(c);  // Reference count: 1
// a still valid
```

### Arc<T>

Atomic reference counting for thread-safe sharing:

```vex
use std::sync::Arc;
use std::thread;

let data = Arc::new(vec![1, 2, 3]);
let data_clone = Arc::clone(&data);

thread::spawn(move || {
    println!("{:?}", data_clone);
});
```

### RefCell<T>

Interior mutability with runtime borrow checking:

```vex
use std::cell::RefCell;

let x = RefCell::new(42);
*x.borrow_mut() += 1;  // Runtime mutable borrow
println!("{}", x.borrow());  // Runtime immutable borrow
```

---

## Unsafe Memory Operations

### Raw Pointers

```vex
let x = 42;
let r1: *const i32 = &x;  // Immutable raw pointer
let r2: *mut i32 = &mut x as *mut i32;  // Mutable raw pointer

unsafe {
    println!("{}", *r1);  // Dereferencing requires unsafe
}
```

### Manual Memory Management

```vex
use std::alloc::{alloc, dealloc, Layout};

unsafe {
    let layout = Layout::new::<i32>();
    let ptr = alloc(layout) as *mut i32;
    *ptr = 42;
    println!("{}", *ptr);
    dealloc(ptr as *mut u8, layout);
}
```

---

## Memory Safety Guarantees

### Compile-Time Checks

1. **No null pointers**: References always valid
2. **No dangling pointers**: Lifetime checking
3. **No data races**: Ownership and borrowing rules
4. **No use-after-free**: Ownership tracking
5. **No double-free**: Single ownership

### Runtime Checks (Debug Mode)

1. **Array bounds checking**
2. **Integer overflow detection**
3. **RefCell borrow checking**

---

## Memory Optimization

### Zero-Cost Abstractions

```vex
// This:
let v: Vec<i32> = (0..10).map(|x| x * 2).collect();

// Compiles to equivalent of:
let mut v = Vec::with_capacity(10);
for i in 0..10 {
    v.push(i * 2);
}
```

### Inline Optimization

```vex
#[inline]
fn add(a: i32, b: i32) -> i32 {
    a + b
}
// Function call eliminated, code inlined
```

### Stack Allocation Optimization

```vex
// Small Vec may be stack-allocated
let v = vec![1, 2, 3];  // May use stack if small enough
```

---

## GPU Memory Model

### Memory Hierarchy

1. **Global Memory**: Accessible by all threads, slow
2. **Shared Memory**: Shared within block, fast
3. **Local Memory**: Private to thread, registers
4. **Constant Memory**: Read-only, cached

### Memory Transfer

```vex
// Host to device
let host_data = vec![1, 2, 3, 4];
let device_buffer = DeviceBuffer::from_slice(&host_data);

// Device to host
let result = device_buffer.to_vec();
```

### Unified Memory

```vex
let unified = UnifiedBuffer::new(size);
// Accessible from both CPU and GPU
// Automatic migration
```

---

## Memory Profiling

### Allocation Tracking

```vex
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn main() {
    // Track allocations
    let stats = ALLOCATOR.stats();
    println!("Allocated: {} bytes", stats.allocated);
}
```

### Leak Detection

```vex
// In debug mode, leaks are detected
let leaked = Box::new(42);
std::mem::forget(leaked);  // Intentional leak - warning in debug
```

---

## Best Practices for AI

### Clear Ownership

```vex
// Good - clear ownership transfer
fn process(data: Vec<i32>) -> Vec<i32> {
    data.into_iter().map(|x| x * 2).collect()
}

// Avoid - unclear ownership
fn process(data: &mut Vec<i32>) {
    *data = data.iter().map(|x| x * 2).collect();
}
```

### Explicit Lifetimes

```vex
// Good - explicit lifetime relationships
fn first<'a>(x: &'a [i32]) -> &'a i32 {
    &x[0]
}

// Avoid - relying on elision in complex cases
```

### Prefer Borrowing

```vex
// Good - borrow when possible
fn print(s: &str) {
    println!("{}", s);
}

// Avoid - taking ownership unnecessarily
fn print(s: String) {
    println!("{}", s);
}
```

---

## Summary

VeZ's memory model provides:
- **Safety**: No undefined behavior in safe code
- **Performance**: Zero-cost abstractions
- **Predictability**: Clear ownership and lifetime rules
- **Control**: Manual memory management when needed

**Key Principle**: Memory safety without garbage collection through compile-time verification.

---

**Next**: See compiler implementation in `compiler/` directory.
