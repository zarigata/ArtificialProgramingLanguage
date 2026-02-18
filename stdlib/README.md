# VeZ Standard Library

The VeZ standard library provides essential types, traits, and functions for VeZ programs, optimized for AI-generated code and high-performance systems programming.

## Structure

```
stdlib/
├── prelude.zari      # Commonly used items (auto-imported)
├── string.zari       # String manipulation functions
├── datetime.zari     # Date and time handling
└── json.zari         # JSON parsing and serialization
```

## Modules

### String Module (`std::string`)

Comprehensive string manipulation functions for text processing.

#### StringBuilder Class

Efficient string concatenation without intermediate allocations:

```zari
use std::string::StringBuilder

let sb = StringBuilder()
sb.append("Hello")
  .append(" ")
  .append("World")
  .append_line("!")
let result = sb.build()  # "Hello World!\n"
```

#### String Functions

| Function | Description |
|----------|-------------|
| `len(s)` | Returns string length |
| `trim(s)`, `trim_start(s)`, `trim_end(s)` | Remove whitespace |
| `lower(s)`, `upper(s)` | Case conversion |
| `capitalize(s)`, `title(s)` | Title case conversion |
| `reverse(s)` | Reverse string |
| `repeat(s, n)` | Repeat string n times |
| `contains(s, substr)` | Check if substring exists |
| `starts_with(s, prefix)`, `ends_with(s, suffix)` | Prefix/suffix checks |
| `find(s, substr)`, `find_last(s, substr)` | Find substring position |
| `count(s, substr)` | Count substring occurrences |
| `replace(s, old, new)` | Replace all occurrences |
| `split(s, sep)`, `split_lines(s)`, `split_whitespace(s)` | Split into parts |
| `join(parts, sep)` | Join parts with separator |
| `slice(s, start, end)` | Extract substring |
| `left(s, n)`, `right(s, n)` | Get first/last n characters |
| `pad_left(s, w, c)`, `pad_right(s, w, c)`, `center(s, w, c)` | Padding |
| `truncate(s, max, suffix)` | Truncate with ellipsis |
| `is_numeric(s)`, `is_alpha(s)`, `is_alphanumeric(s)` | Character classification |
| `to_int(s)`, `to_float(s)`, `to_bool(s)` | Type conversion |
| `escape(s)`, `unescape(s)` | Escape sequence handling |
| `format(template, *args)` | String formatting |
| `levenshtein_distance(s1, s2)` | Edit distance |
| `similarity(s1, s2)` | Similarity score (0.0-1.0) |

### DateTime Module (`std::datetime`)

Comprehensive date and time handling for temporal operations.

#### Duration

Represents a span of time:

```zari
use std::datetime::{Duration, Date, Time, DateTime}

let d1 = Duration.from_secs(60)
let d2 = Duration.from_minutes(5)
let d3 = d1 + d2  # 6 minutes

println(d3.as_secs())    # 360
println(d3.as_minutes()) # 6
```

#### Date

Calendar date representation:

```zari
let today = Date.today()
let date = Date(2024, 1, 15)

println(date.day_of_week())    # 1 (Monday)
println(date.is_leap_year())   # true
println(date.days_in_month())  # 31
println(date.quarter())        # 1
println(date.format("%Y-%m-%d"))  # "2024-01-15"
```

#### Time

Time of day representation:

```zari
let time = Time(14, 30, 0)  # 2:30 PM
let now = Time.now()

println(time.is_pm())           # true
println(time.hour_12())         # 2
println(time.format("%H:%M"))   # "14:30"
```

#### DateTime

Combined date and time:

```zari
let dt = DateTime(2024, 1, 15, 14, 30, 0)
let now = DateTime.now()

println(dt.to_iso8601())  # "2024-01-15T14:30:00"
println(dt.add(Duration.from_hours(2)))  # Add 2 hours
```

#### Stopwatch

Measure elapsed time:

```zari
let sw = Stopwatch()
sw.start()
# ... code to measure ...
let elapsed = sw.stop()
println(elapsed)  # e.g., "152ms"
```

#### Utility Functions

| Function | Description |
|----------|-------------|
| `now()` | Current Unix timestamp (seconds) |
| `now_nanos()` | Current timestamp (nanoseconds) |
| `sleep(duration)` | Sleep for duration |
| `parse_date(s, fmt)` | Parse date from string |
| `parse_time(s, fmt)` | Parse time from string |
| `age(birth_date)` | Calculate age in years |
| `is_leap_year(year)` | Check leap year |
| `days_in_month(year, month)` | Days in month |
| `add_days(date, n)` | Add/subtract days |
| `add_months(date, n)` | Add/subtract months |
| `days_between(d1, d2)` | Days between dates |
| `start_of_week(d)`, `end_of_week(d)` | Week boundaries |
| `start_of_month(d)`, `end_of_month(d)` | Month boundaries |

### JSON Module (`std::json`)

JSON parsing, serialization, and manipulation.

#### JsonValue

Represents any JSON value:

```zari
use std::json::{JsonValue, JsonBuilder}

# Create values
let null_val = JsonValue.null()
let bool_val = JsonValue.from_bool(true)
let num_val = JsonValue.from_int(42)
let str_val = JsonValue.from_string("hello")
let arr_val = JsonValue.from_array([num_val, str_val])
let obj_val = JsonValue.from_object({"key": str_val})

# Check types
println(str_val.is_string())  # true

# Access values
println(arr_val.get(0).as_int())  # 42
println(obj_val.get("key").as_string())  # "hello"
```

#### Parsing

```zari
use std::json::{parse, parse_safe, is_valid}

# Parse JSON string
let value = parse('{"name": "Alice", "age": 30}')

# Safe parsing with error handling
let result = parse_safe('{"invalid": json}')
if result.is_ok():
    println(result.value)
else:
    println(result.error)

# Validate JSON
if is_valid(json_string):
    # Process JSON
```

#### Serialization

```zari
use std::json::{stringify, stringify_pretty}

let obj = JsonValue.from_object({
    "name": JsonValue.from_string("Bob"),
    "active": JsonValue.from_bool(true)
})

println(stringify(obj))
# {"name":"Bob","active":true}

println(stringify_pretty(obj))
# {
#   "name": "Bob",
#   "active": true
# }
```

#### Builder Pattern

Construct complex JSON structures:

```zari
let json = JsonBuilder()
    .object()
    .put_string("name", "Alice")
    .put_int("age", 30)
    .put_object("address")
        .put_string("city", "New York")
        .put_string("zip", "10001")
    .put_array("tags")
        .push_string("developer")
        .push_string("rust")
    .to_string()
```

#### Query and Transform

```zari
use std::json::{get_path, merge, map_array, filter_array}

# Path-based access
let name = get_path(data, "users.0.name")

# Merge objects
let combined = merge(base_obj, override_obj)

# Transform arrays
let doubled = map_array(numbers, |n| n * 2)
let positive = filter_array(numbers, |n| n > 0)

# Convert to native types
let strs = to_string_array(json_array)
let ints = to_int_array(json_array)
```

## Core Types (from Prelude)

- `Option<T>` - Optional values (Some/None)
- `Result<T, E>` - Error handling (Ok/Err)
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

## Usage Examples

### Complete Example: User Management

```zari
use std::string::{StringBuilder, format, trim}
use std::datetime::{DateTime, Duration}
use std::json::{JsonValue, JsonBuilder, parse}

class User:
    def __init__(self, id: int, name: string, email: string):
        self.id = id
        self.name = name
        self.email = email
        self.created_at = DateTime.now()
    
    def to_json(self) -> JsonValue:
        return JsonBuilder()
            .object()
            .put_int("id", self.id)
            .put_string("name", self.name)
            .put_string("email", self.email)
            .put_string("created_at", self.created_at.to_iso8601())
            .build()
    
    def display_name(self) -> string:
        return format("{} <{}>", self.name, self.email)

def main():
    let users = [
        User(1, "Alice", "alice@example.com"),
        User(2, "Bob", "bob@example.com"),
    ]
    
    let output = StringBuilder()
    for user in users:
        output.append_line(user.display_name())
        output.append_line(format("  Created: {}", user.created_at.format("%Y-%m-%d")))
    
    println(output.build())

main()
```

### Example: Data Processing Pipeline

```zari
use std::json::{parse, map_array, filter_array}
use std::datetime::{Date, days_between}

def process_data(json_str: string) -> string:
    let data = parse(json_str)
    
    # Filter recent records
    let cutoff = Date.today() - Duration.from_days(30)
    
    let recent = filter_array(data.get("records"), |record|:
        let date = Date.from_iso8601(record.get("date").as_string())
        date >= cutoff
    )
    
    # Transform data
    let processed = map_array(recent, |record|:
        JsonBuilder()
            .object()
            .put_string("id", record.get("id").as_string())
            .put_float("value", record.get("value").as_float() * 1.1)
            .build()
    )
    
    return stringify_pretty(processed)
```

## Performance Notes

- `StringBuilder` uses internal buffering for O(n) concatenation
- String functions are optimized for UTF-8 handling
- JSON parser uses zero-copy techniques where possible
- DateTime calculations use efficient algorithms for date arithmetic

## Thread Safety

All standard library types are designed to be thread-safe when used with appropriate synchronization:

- Immutable types (String, Duration, Date, Time, DateTime) are freely shareable
- Mutable types (StringBuilder, JsonBuilder) should be synchronized externally
- Collections require external synchronization for concurrent access
