# VeZ Type System: Primitive Types

## Overview

VeZ provides a comprehensive set of primitive types that map directly to hardware representations, ensuring predictable performance and memory layout.

---

## Integer Types

### Signed Integers

| Type | Size | Range | Use Case |
|------|------|-------|----------|
| `i8` | 8 bits | -128 to 127 | Small counters, byte operations |
| `i16` | 16 bits | -32,768 to 32,767 | Short integers |
| `i32` | 32 bits | -2³¹ to 2³¹-1 | Default integer type |
| `i64` | 64 bits | -2⁶³ to 2⁶³-1 | Large numbers, timestamps |
| `i128` | 128 bits | -2¹²⁷ to 2¹²⁷-1 | Cryptography, very large numbers |
| `isize` | Platform | Pointer-sized | Array indexing, pointer arithmetic |

### Unsigned Integers

| Type | Size | Range | Use Case |
|------|------|-------|----------|
| `u8` | 8 bits | 0 to 255 | Bytes, ASCII characters |
| `u16` | 16 bits | 0 to 65,535 | Unicode code points, ports |
| `u32` | 32 bits | 0 to 2³²-1 | Hashes, IDs |
| `u64` | 64 bits | 0 to 2⁶⁴-1 | Large unsigned values |
| `u128` | 128 bits | 0 to 2¹²⁸-1 | UUIDs, very large unsigned |
| `usize` | Platform | Pointer-sized | Array lengths, memory sizes |

### Platform-Dependent Sizes

- **32-bit platforms**: `isize` and `usize` are 32 bits
- **64-bit platforms**: `isize` and `usize` are 64 bits

### Integer Literals

```vex
let decimal = 42;
let hex = 0xFF;
let octal = 0o755;
let binary = 0b1010_1010;
let typed = 42i32;
let unsigned = 255u8;
```

### Integer Operations

**Arithmetic**: `+`, `-`, `*`, `/`, `%`, `**` (power)  
**Bitwise**: `&`, `|`, `^`, `~`, `<<`, `>>`  
**Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`

### Overflow Behavior

**Debug mode**: Panic on overflow  
**Release mode**: Wrapping (two's complement)

**Explicit control**:
```vex
let a = x.wrapping_add(y);    // Always wrap
let b = x.checked_add(y);     // Returns Option<T>
let c = x.saturating_add(y);  // Saturate at bounds
let d = x.overflowing_add(y); // Returns (result, overflow_flag)
```

---

## Floating-Point Types

| Type | Size | Precision | Range |
|------|------|-----------|-------|
| `f32` | 32 bits | ~7 decimal digits | ±3.4 × 10³⁸ |
| `f64` | 64 bits | ~15 decimal digits | ±1.7 × 10³⁰⁸ |

### IEEE 754 Compliance

Both types follow IEEE 754 standard:
- **NaN** (Not a Number)
- **Infinity** (positive and negative)
- **Subnormal numbers**
- **Signed zero**

### Float Literals

```vex
let pi = 3.14159;
let scientific = 6.022e23;
let typed = 2.5f32;
let explicit = 1.0f64;
```

### Float Operations

**Arithmetic**: `+`, `-`, `*`, `/`, `%`  
**Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`  
**Math functions**: `sqrt()`, `sin()`, `cos()`, `pow()`, etc.

### Special Values

```vex
let nan = f64::NAN;
let inf = f64::INFINITY;
let neg_inf = f64::NEG_INFINITY;

// Checking
if x.is_nan() { }
if x.is_infinite() { }
if x.is_finite() { }
```

---

## Boolean Type

| Type | Size | Values |
|------|------|--------|
| `bool` | 1 byte | `true`, `false` |

### Boolean Operations

**Logical**: `&&` (and), `||` (or), `!` (not)  
**Comparison**: `==`, `!=`

### Usage

```vex
let flag: bool = true;
let result = x > 0 && y < 10;

if condition {
    // true branch
} else {
    // false branch
}
```

---

## Character Type

| Type | Size | Values |
|------|------|--------|
| `char` | 4 bytes | Unicode scalar values |

### Unicode Support

- Represents a Unicode scalar value (U+0000 to U+D7FF and U+E000 to U+10FFFF)
- **Not** a byte or UTF-8 code unit
- Always valid Unicode

### Character Literals

```vex
let letter = 'a';
let emoji = '😀';
let unicode = '\u{1F600}';
let escape = '\n';
```

### Escape Sequences

| Escape | Meaning |
|--------|---------|
| `\n` | Newline |
| `\r` | Carriage return |
| `\t` | Tab |
| `\\` | Backslash |
| `\'` | Single quote |
| `\"` | Double quote |
| `\0` | Null character |
| `\xNN` | ASCII character (hex) |
| `\u{NNNN}` | Unicode character |

---

## String Slice Type

| Type | Size | Values |
|------|------|--------|
| `str` | Variable | UTF-8 encoded text |

### String Slices

- **Unsized type**: Cannot exist directly, only as `&str`
- **UTF-8 encoded**: Always valid UTF-8
- **Immutable**: String slices are read-only

### String Literals

```vex
let text: &str = "Hello, VeZ!";
let multiline = "Line 1
Line 2
Line 3";
let raw = r"No \n escapes";
let raw_with_quotes = r#"Can use "quotes" here"#;
```

### String Operations

```vex
let len = text.len();           // Byte length
let chars = text.chars();       // Iterator over chars
let bytes = text.bytes();       // Iterator over bytes
let slice = &text[0..5];        // Substring (byte indices)
let contains = text.contains("VeZ");
```

---

## Unit Type

| Type | Size | Values |
|------|------|--------|
| `()` | 0 bytes | `()` |

### Usage

- Represents "no value"
- Similar to `void` in C/C++
- Default return type for functions

```vex
fn do_something() {
    // Implicitly returns ()
}

fn explicit_unit() -> () {
    ()
}

let unit_value = ();
```

---

## Never Type

| Type | Size | Values |
|------|------|--------|
| `!` | N/A | None (diverges) |

### Diverging Functions

Functions that never return:

```vex
fn infinite_loop() -> ! {
    loop {
        // Never exits
    }
}

fn panic_always() -> ! {
    panic!("This function never returns");
}

fn exit_program() -> ! {
    std::process::exit(1);
}
```

### Type Theory

- `!` can coerce to any type
- Useful in match expressions and error handling

```vex
let x: i32 = match value {
    Some(v) => v,
    None => panic!("No value!"),  // ! coerces to i32
};
```

---

## Type Inference

VeZ uses Hindley-Milner type inference with extensions:

```vex
let x = 42;           // Inferred as i32 (default integer)
let y = 3.14;         // Inferred as f64 (default float)
let z = true;         // Inferred as bool
let c = 'a';          // Inferred as char
let s = "hello";      // Inferred as &str

// Explicit types when needed
let explicit: u64 = 42;
let float: f32 = 3.14;
```

---

## Type Casting

### Explicit Casts

```vex
let x: i32 = 42;
let y: i64 = x as i64;        // Widening (safe)
let z: i16 = x as i16;        // Narrowing (may truncate)
let f: f64 = x as f64;        // Int to float
let i: i32 = 3.14 as i32;     // Float to int (truncates)
```

### Casting Rules

**Safe casts** (no data loss):
- Smaller to larger integer of same signedness
- Integer to float (may lose precision for very large values)

**Unsafe casts** (potential data loss):
- Larger to smaller integer (truncates)
- Signed to unsigned or vice versa
- Float to integer (truncates decimal)

---

## Memory Layout

All primitive types have well-defined memory layouts:

| Type | Alignment | Size |
|------|-----------|------|
| `i8`, `u8` | 1 byte | 1 byte |
| `i16`, `u16` | 2 bytes | 2 bytes |
| `i32`, `u32`, `f32` | 4 bytes | 4 bytes |
| `i64`, `u64`, `f64` | 8 bytes | 8 bytes |
| `i128`, `u128` | 16 bytes | 16 bytes |
| `bool` | 1 byte | 1 byte |
| `char` | 4 bytes | 4 bytes |
| `()` | 1 byte | 0 bytes |

---

## Default Values

Primitive types have default values when uninitialized (in safe contexts):

| Type | Default |
|------|---------|
| Integers | `0` |
| Floats | `0.0` |
| `bool` | `false` |
| `char` | `'\0'` |
| `()` | `()` |

**Note**: In VeZ, variables must be initialized before use. Defaults are only for struct fields with `#[derive(Default)]`.

---

## Traits Implemented

All primitive types implement:

- `Copy`: Can be copied bitwise
- `Clone`: Can be cloned
- `Debug`: Can be debug-printed
- `Display`: Can be formatted (most types)
- `PartialEq`, `Eq`: Equality comparison
- `PartialOrd`, `Ord`: Ordering (except floats for `Ord`)
- `Hash`: Can be hashed

---

## Performance Characteristics

### Compile-Time Guarantees

- **No boxing**: Primitives are never heap-allocated
- **No overhead**: Direct hardware representation
- **Predictable size**: Known at compile time
- **Zero-cost abstractions**: Operations compile to single instructions

### CPU Instructions

Most primitive operations map to single CPU instructions:
- Integer arithmetic: ADD, SUB, MUL, DIV
- Bitwise operations: AND, OR, XOR, SHL, SHR
- Comparisons: CMP, TEST
- Float operations: FADD, FMUL, etc. (SSE/AVX)

---

## AI Code Generation Notes

### Patterns for AI

**Type annotations** (when ambiguous):
```vex
let value: i32 = parse_input();  // Clear intent
```

**Explicit casts**:
```vex
let result = (x as f64) / (y as f64);  // Avoid integer division
```

**Overflow handling**:
```vex
let sum = a.checked_add(b).expect("Overflow");  // Explicit error
```

### Common Mistakes to Avoid

❌ **Integer division when float expected**:
```vex
let result = 5 / 2;  // Result is 2, not 2.5
```

✅ **Correct**:
```vex
let result = 5.0 / 2.0;  // Result is 2.5
```

❌ **Comparing floats with ==**:
```vex
if x == 0.1 + 0.2 { }  // May fail due to precision
```

✅ **Correct**:
```vex
if (x - 0.3).abs() < 1e-10 { }  // Epsilon comparison
```

---

## Summary

VeZ's primitive types provide:
- **Predictable performance**: Direct hardware mapping
- **Memory safety**: No undefined behavior
- **Explicit semantics**: Clear overflow and casting rules
- **AI-friendly**: Regular patterns, explicit types when needed

---

**Next**: See [Compound Types](compounds.md) for arrays, tuples, structs, and enums.
