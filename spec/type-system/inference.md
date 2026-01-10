# VeZ Type System: Type Inference

## Overview

VeZ uses a Hindley-Milner based type inference system with extensions for traits, lifetimes, and const generics. The goal is to minimize explicit type annotations while maintaining predictability for AI code generation.

---

## Inference Algorithm

### Hindley-Milner Core

VeZ's type inference is based on the Hindley-Milner algorithm with the following extensions:
- Trait constraints
- Lifetime inference
- Associated types
- Const generics

### Inference Process

1. **Collection**: Gather type constraints from the AST
2. **Unification**: Solve constraints to find most general types
3. **Generalization**: Introduce type variables where needed
4. **Instantiation**: Replace type variables with concrete types

---

## Local Type Inference

### Variable Bindings

```vex
// Type inferred from literal
let x = 42;              // x: i32 (default integer type)
let y = 3.14;            // y: f64 (default float type)
let z = true;            // z: bool
let s = "hello";         // s: &str

// Type inferred from usage
let mut v = Vec::new();  // Type unknown yet
v.push(1);               // Now v: Vec<i32>

// Explicit type annotation
let a: u64 = 42;         // a: u64
```

### Function Return Types

```vex
// Inferred from return expression
fn add(a: i32, b: i32) {
    a + b  // Return type inferred as i32
}

// Explicit return type
fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

// Multiple return points must unify
fn abs(x: i32) -> i32 {
    if x < 0 {
        -x     // i32
    } else {
        x      // i32
    }
    // Both branches must have same type
}
```

---

## Generic Type Inference

### Function Generics

```vex
// Generic function
fn identity<T>(x: T) -> T {
    x
}

// Usage - T inferred from argument
let a = identity(42);      // T = i32
let b = identity("hello"); // T = &str

// Explicit type parameter
let c = identity::<f64>(3.14);
```

### Struct Generics

```vex
struct Container<T> {
    value: T,
}

// Type inferred from field
let c1 = Container { value: 42 };        // Container<i32>
let c2 = Container { value: "hello" };   // Container<&str>

// Explicit type parameter
let c3 = Container::<f64> { value: 3.14 };
```

### Method Generics

```vex
impl<T> Container<T> {
    fn new(value: T) -> Self {
        Container { value }
    }
    
    fn get(&self) -> &T {
        &self.value
    }
}

// Type inferred from argument
let c = Container::new(42);  // Container<i32>
```

---

## Trait Constraint Inference

### Trait Bounds

```vex
// Trait bound on generic parameter
fn print<T: Display>(x: T) {
    println!("{}", x);
}

// Multiple bounds
fn process<T: Clone + Debug>(x: T) {
    let y = x.clone();
    println!("{:?}", y);
}

// Where clause for complex bounds
fn complex<T, U>(x: T, y: U) -> T
where
    T: Clone + Display,
    U: Into<T>,
{
    let z = y.into();
    x.clone()
}
```

### Automatic Trait Derivation

```vex
// Compiler infers which traits are needed
fn compare<T>(a: T, b: T) -> bool {
    a == b  // Compiler infers T: PartialEq
}

// Multiple trait requirements
fn sort<T>(items: &mut [T]) {
    // Compiler infers T: Ord
    items.sort();
}
```

---

## Lifetime Inference

### Lifetime Elision Rules

**Rule 1**: Each elided lifetime in parameters gets a distinct lifetime:
```vex
fn foo(x: &i32) -> &i32
// Expands to:
fn foo<'a>(x: &'a i32) -> &'a i32
```

**Rule 2**: If there's exactly one input lifetime, it's assigned to all output lifetimes:
```vex
fn first(x: &Vec<i32>) -> &i32
// Expands to:
fn first<'a>(x: &'a Vec<i32>) -> &'a i32
```

**Rule 3**: If there's a `&self` or `&mut self`, its lifetime is assigned to all output lifetimes:
```vex
impl<T> Container<T> {
    fn get(&self) -> &T
    // Expands to:
    fn get<'a>(&'a self) -> &'a T
}
```

### Explicit Lifetimes

When elision rules don't apply:
```vex
// Multiple input lifetimes - must be explicit
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// Different lifetimes
fn first_word<'a, 'b>(x: &'a str, y: &'b str) -> &'a str {
    x.split_whitespace().next().unwrap_or(x)
}
```

### Lifetime Bounds

```vex
// T must outlive 'a
fn reference<'a, T: 'a>(x: &'a T) -> &'a T {
    x
}

// Struct with lifetime bounds
struct Ref<'a, T: 'a> {
    value: &'a T,
}
```

---

## Associated Type Inference

### Associated Types in Traits

```vex
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

// Implementation specifies associated type
impl Iterator for Counter {
    type Item = i32;
    
    fn next(&mut self) -> Option<i32> {
        // ...
    }
}

// Usage - Item type inferred from implementation
fn sum<I: Iterator>(iter: I) -> i32
where
    I::Item: Add<Output = i32>,
{
    // Compiler knows I::Item is i32 for Counter
}
```

---

## Const Generic Inference

### Array Sizes

```vex
// Array size inferred from literal
let arr = [1, 2, 3, 4, 5];  // [i32; 5]

// Generic over array size
fn first<T, const N: usize>(arr: [T; N]) -> T {
    arr[0]
}

// N inferred from argument
let x = first([1, 2, 3]);  // N = 3
```

---

## Type Inference Rules

### Default Types

When type cannot be inferred, defaults are used:

| Context | Default Type |
|---------|--------------|
| Integer literal | `i32` |
| Float literal | `f64` |
| Empty array | Error (must specify type) |
| Empty vec | Error (must specify type) |

### Numeric Type Coercion

VeZ does **not** perform implicit numeric coercion:

```vex
let x: i32 = 42;
let y: i64 = x;  // ERROR: mismatched types

// Must use explicit cast
let y: i64 = x as i64;  // OK
```

### Reference Coercion

Automatic coercions that **are** allowed:

```vex
// &mut T to &T
let mut x = 42;
let r: &i32 = &mut x;  // OK

// &T to *const T (in unsafe)
unsafe {
    let p: *const i32 = &x;  // OK
}

// Array to slice
let arr = [1, 2, 3];
let slice: &[i32] = &arr;  // OK
```

---

## Inference Limitations

### Ambiguous Cases

```vex
// ERROR: Cannot infer type
let v = Vec::new();
// Must specify:
let v: Vec<i32> = Vec::new();
// Or:
let v = Vec::<i32>::new();

// ERROR: Multiple possible types
let x = "42".parse();
// Must specify:
let x: i32 = "42".parse().unwrap();
```

### Turbofish Syntax

When type parameters cannot be inferred:

```vex
// Specify type parameters explicitly
let v = Vec::<i32>::new();
let x = parse::<i32>("42");
let iter = (0..10).collect::<Vec<_>>();  // Partial inference
```

---

## Inference for AI Code Generation

### Best Practices

**1. Explicit types for function signatures**:
```vex
// Good - clear intent
fn process(input: &str) -> Result<i32, Error> {
    // ...
}

// Avoid - unclear return type
fn process(input: &str) {
    // ...
}
```

**2. Type annotations for complex expressions**:
```vex
// Good - explicit intermediate type
let parsed: i32 = input.parse()?;
let result = parsed * 2;

// Harder to understand
let result = input.parse()? * 2;
```

**3. Turbofish for clarity**:
```vex
// Good - clear what's being collected
let numbers = (0..10).collect::<Vec<i32>>();

// Less clear
let numbers: Vec<i32> = (0..10).collect();
```

### AI-Friendly Patterns

**Explicit over implicit**:
```vex
// Prefer explicit types in public APIs
pub fn create_buffer(size: usize) -> Vec<u8> {
    vec![0; size]
}

// Over inferred return
pub fn create_buffer(size: usize) {
    vec![0; size]
}
```

**Clear generic constraints**:
```vex
// Explicit trait bounds
fn sort_items<T: Ord>(items: &mut [T]) {
    items.sort();
}

// Over implicit
fn sort_items<T>(items: &mut [T]) {
    items.sort();  // T: Ord inferred but not visible
}
```

---

## Type Inference Errors

### Common Error Messages

**Cannot infer type**:
```
error: cannot infer type for `T`
  --> example.zari:2:9
   |
2  |     let x = Vec::new();
   |         ^ consider giving `x` a type
```

**Type mismatch**:
```
error: mismatched types
  --> example.zari:3:9
   |
3  |     let y: i64 = x;
   |                  ^ expected `i64`, found `i32`
   |
help: you can convert an `i32` to `i64`
   |
3  |     let y: i64 = x as i64;
   |                  ^^^^^^^^
```

**Conflicting types**:
```
error: conflicting types for `T`
  --> example.zari:4:5
   |
3  |     v.push(1);
   |            - `T` inferred to be `i32` here
4  |     v.push("hello");
   |            ^^^^^^^ expected `i32`, found `&str`
```

---

## Advanced Inference

### Higher-Ranked Trait Bounds (HRTB)

```vex
// For any lifetime 'a
fn apply<F>(f: F)
where
    F: for<'a> Fn(&'a str) -> &'a str,
{
    let result = f("hello");
}
```

### Recursive Type Inference

```vex
// Infers recursive structure
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

let list = List::Cons(1, Box::new(List::Cons(2, Box::new(List::Nil))));
// Type: List<i32>
```

---

## Inference Performance

### Compile-Time Complexity

- **Local inference**: O(n) where n is expression size
- **Global inference**: O(n²) worst case, O(n log n) typical
- **Trait resolution**: O(n × m) where m is trait count

### Optimization Strategies

1. **Early type annotation**: Reduces inference work
2. **Monomorphization**: Generates specialized code
3. **Incremental compilation**: Reuses inference results

---

## Summary

VeZ's type inference provides:
- **Minimal annotations**: Infer types where obvious
- **Predictability**: Clear rules for AI to follow
- **Explicitness when needed**: Type annotations for clarity
- **Performance**: Compile-time inference, zero runtime cost

**Key Principle**: Inference should reduce boilerplate without sacrificing clarity or predictability.

---

**Next**: See [Generics](generics.md) for detailed generic programming guide.
