# JSON Parser Plugin

A VeZ plugin that adds JSON parsing and serialization capabilities to the language.

## Features

- `json!` macro for compile-time JSON parsing
- `JsonValue` type for representing JSON data
- Type-safe JSON operations
- Zero-cost abstractions

## Installation

```bash
vpm plugin install json-parser
```

## Usage

```vex
#![plugin(json_parser)]

fn main() {
    // Parse JSON at compile time
    let data = json!({
        "name": "VeZ",
        "version": "1.0.0",
        "features": ["fast", "safe", "extensible"]
    });
    
    // Access JSON values
    if let JsonValue::Object(obj) = data {
        if let Some(JsonValue::String(name)) = obj.get("name") {
            println!("Name: {}", name);
        }
    }
    
    // Serialize to JSON string
    let json_str = data.to_string();
    println!("{}", json_str);
}
```

## API

### Types

- `JsonValue` - Represents any JSON value
  - `Null`
  - `Bool(bool)`
  - `Number(f64)`
  - `String(String)`
  - `Array(Vec<JsonValue>)`
  - `Object(HashMap<String, JsonValue>)`

### Macros

- `json!(...)` - Parse JSON literal at compile time

### Methods

- `is_null()` - Check if value is null
- `as_bool()` - Convert to boolean
- `as_number()` - Convert to number
- `as_string()` - Convert to string
- `as_array()` - Convert to array
- `as_object()` - Convert to object

## Building from Source

```bash
cd json-parser
vpm plugin build
vpm plugin install --path .
```

## License

MIT
