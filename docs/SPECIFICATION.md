# Language Specification (Draft v0.1)

## Introduction

This document provides the formal specification for the AI-first programming language. This is a living document and will evolve as the language develops.

**Status**: Draft  
**Version**: 0.1.0  
**Last Updated**: January 2026

---

## Design Goals

1. **AI-Optimized**: Syntax and semantics optimized for transformer-based AI models
2. **Performance**: Compile to efficient machine code
3. **Safety**: Memory safety without garbage collection
4. **Predictability**: Deterministic behavior
5. **Hardware Access**: Direct CPU, GPU, and memory control

---

## Lexical Structure

### Character Set

**Encoding**: UTF-8  
**Source files**: Must be valid UTF-8

### Identifiers

**Pattern**: `[a-zA-Z_][a-zA-Z0-9_]*`

**Conventions**:
- `snake_case` for variables, functions, modules
- `PascalCase` for types, traits
- `SCREAMING_SNAKE_CASE` for constants

**Reserved**: Cannot start with `__` (reserved for compiler)

### Keywords

```
Core:      fn let mut const type struct enum union trait impl
Control:   if else match loop while for break continue return
Memory:    ref mut move copy drop unsafe
Async:     async await yield
Modules:   mod use pub import export
Types:     i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize
           f32 f64 bool char str void never
GPU:       kernel device shared global local barrier sync
Other:     as in where self Self static extern inline
```

### Operators

```
Arithmetic:    + - * / % **
Bitwise:       & | ^ ~ << >>
Logical:       && || !
Comparison:    == != < > <= >=
Assignment:    = += -= *= /= %= &= |= ^= <<= >>=
Range:         .. ..=
Access:        . -> :: []
Other:         ? @ # $
```

### Literals

#### Integer Literals
```
Decimal:       42, 1_000_000
Hexadecimal:   0xFF, 0xDEAD_BEEF
Octal:         0o755
Binary:        0b1010_1010
```

**Type suffix**: `42i32`, `0xFFu8`

#### Float Literals
```
Standard:      3.14, 2.5e10
Scientific:    1.23e-4
```

**Type suffix**: `3.14f32`, `2.5f64`

#### Boolean Literals
```
true, false
```

#### Character Literals
```
'a', '\\n', '\\u{1F600}'
```

#### String Literals
```
"hello world"
"multi\\nline"
r"raw string \\n not escaped"
r#"raw with "quotes""#
```

### Comments

```
Single-line:   // comment
Multi-line:    /* comment */
Doc comments:  /// documentation
               /** block doc */
```

---

## Type System

### Primitive Types

#### Integer Types
```
Signed:    i8, i16, i32, i64, i128, isize (pointer-sized)
Unsigned:  u8, u16, u32, u64, u128, usize (pointer-sized)
```

**Range**:
- `i8`: -128 to 127
- `u8`: 0 to 255
- etc.

#### Floating-Point Types
```
f32:  IEEE 754 single precision
f64:  IEEE 754 double precision
```

#### Boolean Type
```
bool: true or false
```

#### Character Type
```
char: Unicode scalar value (4 bytes)
```

#### Unit Type
```
(): zero-sized type (like void)
```

#### Never Type
```
!: type of diverging functions
```

### Compound Types

#### Arrays
```
Fixed size:  [T; N]
Example:     [i32; 5]
Literal:     [1, 2, 3, 4, 5]
Repeat:      [0; 100]
```

#### Slices
```
Type:        [T]
Reference:   &[T], &mut [T]
```

#### Tuples
```
Type:        (T1, T2, ..., Tn)
Example:     (i32, f64, bool)
Literal:     (42, 3.14, true)
Access:      tuple.0, tuple.1
```

#### Structs
```
Named fields:
struct Point {
    x: f64,
    y: f64,
}

Tuple struct:
struct Color(u8, u8, u8);

Unit struct:
struct Marker;
```

#### Enums
```
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

With discriminant:
enum Status {
    Active = 1,
    Inactive = 0,
}
```

#### Unions
```
union Data {
    i: i32,
    f: f32,
}
```

**Note**: Accessing union fields is unsafe

### Reference Types

#### Shared Reference
```
Type:        &T
Properties:  Immutable, copyable, multiple allowed
```

#### Mutable Reference
```
Type:        &mut T
Properties:  Exclusive, not copyable, only one allowed
```

#### Raw Pointers
```
Type:        *const T, *mut T
Properties:  Unsafe, no guarantees
```

### Function Types

```
Function pointer:  fn(i32, i32) -> i32
Closure:          |i32, i32| -> i32
```

### Generic Types

```
struct Container<T> {
    value: T,
}

fn identity<T>(x: T) -> T {
    x
}
```

**Constraints**:
```
fn print<T: Display>(x: T) {
    // T must implement Display trait
}
```

---

## Expressions

### Literal Expressions
```
42
3.14
true
'a'
"hello"
```

### Variable Expressions
```
x
my_variable
```

### Operator Expressions
```
a + b
x * y
!flag
a && b
```

### Function Calls
```
function(arg1, arg2)
object.method(arg)
```

### Array/Tuple Access
```
array[index]
tuple.0
```

### Struct Initialization
```
Point { x: 1.0, y: 2.0 }
Color(255, 0, 0)
```

### Block Expressions
```
{
    let x = 5;
    let y = 10;
    x + y  // last expression is return value
}
```

### If Expressions
```
if condition {
    value1
} else {
    value2
}
```

### Match Expressions
```
match value {
    pattern1 => result1,
    pattern2 => result2,
    _ => default,
}
```

### Loop Expressions
```
loop {
    // infinite loop
    break;
}

while condition {
    // conditional loop
}

for item in iterator {
    // iteration
}
```

### Range Expressions
```
1..10      // exclusive end
1..=10     // inclusive end
```

---

## Statements

### Let Binding
```
let x = 5;
let mut y = 10;
let z: i32 = 15;
```

### Assignment
```
x = 5;
y += 10;
```

### Expression Statement
```
function_call();
```

### Item Declaration
```
fn function() { }
struct Type { }
```

---

## Functions

### Function Declaration
```
fn name(param1: Type1, param2: Type2) -> ReturnType {
    // body
}
```

### Generic Functions
```
fn generic<T>(param: T) -> T {
    param
}
```

### Methods
```
impl Type {
    fn method(&self, param: i32) -> i32 {
        // self is immutable reference
    }
    
    fn method_mut(&mut self) {
        // self is mutable reference
    }
    
    fn associated_fn() {
        // no self parameter
    }
}
```

### Closures
```
let closure = |x, y| x + y;
let typed = |x: i32, y: i32| -> i32 { x + y };
```

---

## Memory Management

### Ownership

**Rules**:
1. Each value has one owner
2. When owner goes out of scope, value is dropped
3. Ownership can be transferred (move)

```
let s1 = String::from("hello");
let s2 = s1;  // s1 moved to s2, s1 invalid
```

### Borrowing

**Immutable borrow**:
```
let s = String::from("hello");
let r1 = &s;
let r2 = &s;  // multiple immutable borrows OK
```

**Mutable borrow**:
```
let mut s = String::from("hello");
let r = &mut s;  // only one mutable borrow
```

### Lifetimes

**Explicit annotation**:
```
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

**Struct lifetimes**:
```
struct Ref<'a> {
    value: &'a i32,
}
```

---

## Traits

### Trait Definition
```
trait Display {
    fn display(&self) -> String;
}
```

### Trait Implementation
```
impl Display for Point {
    fn display(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}
```

### Trait Bounds
```
fn print<T: Display>(x: T) {
    println!("{}", x.display());
}
```

### Multiple Bounds
```
fn process<T: Display + Clone>(x: T) { }
```

### Where Clauses
```
fn complex<T, U>(x: T, y: U)
where
    T: Display + Clone,
    U: Debug,
{
    // body
}
```

---

## Modules

### Module Declaration
```
mod module_name {
    // items
}
```

### File-based Modules
```
mod my_module;  // loads my_module.rs
```

### Visibility
```
pub fn public_function() { }
fn private_function() { }

pub struct PublicStruct {
    pub public_field: i32,
    private_field: i32,
}
```

### Use Declarations
```
use std::collections::HashMap;
use std::io::{Read, Write};
use std::fs::*;
```

---

## Concurrency

### Threads
```
use std::thread;

let handle = thread::spawn(|| {
    // thread code
});

handle.join().unwrap();
```

### Channels
```
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();

tx.send(42).unwrap();
let value = rx.recv().unwrap();
```

### Mutex
```
use std::sync::Mutex;

let m = Mutex::new(5);
{
    let mut num = m.lock().unwrap();
    *num = 6;
}
```

### Async/Await
```
async fn fetch_data() -> Result<Data, Error> {
    let response = http::get("url").await?;
    Ok(response.parse()?)
}

async fn main() {
    let data = fetch_data().await;
}
```

---

## GPU Programming

### Kernel Definition
```
#[kernel]
fn vector_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let idx = thread_idx();
    if idx < a.len() {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Device Memory
```
let device_a = DeviceBuffer::from_slice(&host_a);
let device_b = DeviceBuffer::from_slice(&host_b);
let mut device_c = DeviceBuffer::new(size);
```

### Kernel Launch
```
launch_kernel(
    vector_add,
    grid_size,
    block_size,
    (&device_a, &device_b, &mut device_c)
);
```

---

## Unsafe Code

### Unsafe Blocks
```
unsafe {
    // unsafe operations
}
```

### Unsafe Functions
```
unsafe fn dangerous() {
    // unsafe code
}
```

### Unsafe Operations
- Dereferencing raw pointers
- Calling unsafe functions
- Accessing mutable statics
- Implementing unsafe traits
- Accessing union fields

---

## Macros

### Declarative Macros
```
macro_rules! vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
```

### Procedural Macros
```
#[derive(Debug, Clone)]
struct MyStruct { }
```

---

## Attributes

### Common Attributes
```
#[inline]              // inline function
#[derive(Debug)]       // derive trait
#[cfg(target_os)]      // conditional compilation
#[test]                // test function
#[allow(dead_code)]    // suppress warning
#[deprecated]          // mark as deprecated
```

---

## Error Handling

### Result Type
```
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### Option Type
```
enum Option<T> {
    Some(T),
    None,
}
```

### Error Propagation
```
fn may_fail() -> Result<i32, Error> {
    let value = operation()?;  // ? operator
    Ok(value)
}
```

### Panic
```
panic!("error message");
assert!(condition);
assert_eq!(a, b);
```

---

## Standard Library Overview

### Core Modules
```
core::       Platform-independent primitives
alloc::      Allocation and collections
std::        Full standard library
```

### Common Types
```
String       Growable string
Vec<T>       Growable array
HashMap<K,V> Hash map
Option<T>    Optional value
Result<T,E>  Error handling
Box<T>       Heap allocation
Rc<T>        Reference counting
Arc<T>       Atomic reference counting
```

---

## Grammar (EBNF)

```ebnf
program = { item } ;

item = function
     | struct_def
     | enum_def
     | trait_def
     | impl_block
     | use_decl
     | mod_decl ;

function = "fn" identifier [ generic_params ] "(" [ params ] ")" 
           [ "->" type ] block ;

struct_def = "struct" identifier [ generic_params ] 
             ( "{" [ struct_fields ] "}" | "(" [ tuple_fields ] ")" | ";" ) ;

enum_def = "enum" identifier [ generic_params ] 
           "{" enum_variants "}" ;

type = primitive_type
     | array_type
     | tuple_type
     | reference_type
     | function_type
     | path_type ;

expression = literal
           | identifier
           | binary_op
           | unary_op
           | call
           | block
           | if_expr
           | match_expr
           | loop_expr ;

statement = let_stmt
          | expr_stmt
          | item ;
```

---

## AI Training Considerations

### Syntax Regularity
- Consistent keyword placement
- Predictable nesting patterns
- Clear delimiters
- Minimal special cases

### Semantic Clarity
- Explicit type annotations (when needed)
- Clear ownership semantics
- Deterministic evaluation order
- No implicit conversions

### Pattern Recognition
- Common idioms documented
- Standard library patterns
- Error handling patterns
- Concurrency patterns

---

## Future Extensions

### Planned Features
- Effect system
- Dependent types (research)
- Compile-time reflection
- Advanced const evaluation

### Under Consideration
- Algebraic effects
- Linear types
- Gradual typing
- JIT compilation mode

---

## Appendix

### Operator Precedence
```
Highest:  . -> [] () function calls
          ! - * & (unary)
          * / %
          + -
          << >>
          &
          ^
          |
          == != < > <= >=
          &&
          ||
          = += -= *= /= ...
Lowest:   ,
```

### Reserved for Future Use
```
become, do, final, override, priv, typeof, unsized, virtual, yield
```

---

**Note**: This specification is subject to change as the language evolves. See the roadmap for planned features and timeline.
