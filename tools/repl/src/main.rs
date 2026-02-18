//! VeZ REPL - Interactive Read-Eval-Print Loop
//!
//! Provides an interactive environment for exploring and testing VeZ code.

use anyhow::{Result, Context};
use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::{CompletionType, Config, EditMode, Editor};
use std::collections::HashMap;
use std::fmt;

/// REPL state
pub struct Repl {
    /// Variable bindings
    bindings: HashMap<String, Value>,
    /// Command history
    history: Vec<String>,
    /// Current module context
    context: EvalContext,
    /// Debug mode
    debug: bool,
}

/// Evaluation context
#[derive(Default)]
pub struct EvalContext {
    /// Imported modules
    imports: Vec<String>,
    /// Current function scope
    current_function: Option<String>,
}

/// Runtime value
#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Unit,
    Function(String),
    Array(Vec<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Unit => write!(f, "()"),
            Value::Function(name) => write!(f, "<fn {}>", name),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", items.join(", "))
            }
        }
    }
}

impl Value {
    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "i64",
            Value::Float(_) => "f64",
            Value::Bool(_) => "bool",
            Value::String(_) => "String",
            Value::Unit => "()",
            Value::Function(_) => "fn",
            Value::Array(_) => "Array",
        }
    }
}

impl Repl {
    /// Create a new REPL instance
    pub fn new() -> Self {
        Repl {
            bindings: HashMap::new(),
            history: Vec::new(),
            context: EvalContext::default(),
            debug: false,
        }
    }

    /// Enable debug mode
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Evaluate a line of input
    pub fn eval(&mut self, input: &str) -> Result<ReplOutput> {
        let input = input.trim();
        
        // Skip empty input
        if input.is_empty() {
            return Ok(ReplOutput::Empty);
        }

        // Add to history
        self.history.push(input.to_string());

        // Handle commands
        if input.starts_with(':') {
            return self.handle_command(input);
        }

        // Parse and evaluate expression
        self.eval_expression(input)
    }

    /// Handle REPL commands
    fn handle_command(&mut self, input: &str) -> Result<ReplOutput> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts.get(0).map(|s| *s).unwrap_or("");

        match cmd {
            ":help" | ":h" => {
                Ok(ReplOutput::Info(HELP_TEXT.to_string()))
            }
            ":quit" | ":q" => {
                Ok(ReplOutput::Quit)
            }
            ":type" | ":t" => {
                if let Some(expr) = parts.get(1) {
                    self.type_of(expr)
                } else {
                    Ok(ReplOutput::Error("Usage: :type <expression>".to_string()))
                }
            }
            ":doc" | ":d" => {
                if let Some(item) = parts.get(1) {
                    self.show_doc(item)
                } else {
                    Ok(ReplOutput::Error("Usage: :doc <item>".to_string()))
                }
            }
            ":load" | ":l" => {
                if let Some(path) = parts.get(1) {
                    self.load_file(path)
                } else {
                    Ok(ReplOutput::Error("Usage: :load <file>".to_string()))
                }
            }
            ":let" => {
                // Parse: :let x = 5
                if parts.len() >= 4 && parts[2] == "=" {
                    let name = parts[1].to_string();
                    let expr = parts[3..].join(" ");
                    match self.eval_expression(&expr)? {
                        ReplOutput::Value { value, ty } => {
                            self.bindings.insert(name.clone(), value.clone());
                            Ok(ReplOutput::Value { 
                                value, 
                                ty: format!("{}: {}", name, ty) 
                            })
                        }
                        other => Ok(other),
                    }
                } else {
                    Ok(ReplOutput::Error("Usage: :let <name> = <expr>".to_string()))
                }
            }
            ":debug" => {
                self.debug = !self.debug;
                Ok(ReplOutput::Info(format!(
                    "Debug mode: {}", 
                    if self.debug { "ON" } else { "OFF" }
                )))
            }
            ":clear" => {
                self.bindings.clear();
                Ok(ReplOutput::Info("Cleared all bindings".to_string()))
            }
            ":history" => {
                let history: String = self.history.iter()
                    .enumerate()
                    .map(|(i, h)| format!("{}: {}", i + 1, h))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(ReplOutput::Info(history))
            }
            _ => {
                Ok(ReplOutput::Error(format!(
                    "Unknown command: {}. Type :help for available commands.",
                    cmd
                )))
            }
        }
    }

    /// Evaluate an expression
    fn eval_expression(&mut self, input: &str) -> Result<ReplOutput> {
        // Check for variable reference
        if let Some(value) = self.bindings.get(input) {
            return Ok(ReplOutput::Value {
                value: value.clone(),
                ty: value.type_name().to_string(),
            });
        }

        // Parse literals
        if let Ok(n) = input.parse::<i64>() {
            return Ok(ReplOutput::Value {
                value: Value::Int(n),
                ty: "i64".to_string(),
            });
        }

        if let Ok(n) = input.parse::<f64>() {
            return Ok(ReplOutput::Value {
                value: Value::Float(n),
                ty: "f64".to_string(),
            });
        }

        if input == "true" {
            return Ok(ReplOutput::Value {
                value: Value::Bool(true),
                ty: "bool".to_string(),
            });
        }

        if input == "false" {
            return Ok(ReplOutput::Value {
                value: Value::Bool(false),
                ty: "bool".to_string(),
            });
        }

        // String literal
        if input.starts_with('"') && input.ends_with('"') {
            let s = input[1..input.len()-1].to_string();
            return Ok(ReplOutput::Value {
                value: Value::String(s),
                ty: "String".to_string(),
            });
        }

        // Array literal
        if input.starts_with('[') && input.ends_with(']') {
            let inner = &input[1..input.len()-1];
            if inner.is_empty() {
                return Ok(ReplOutput::Value {
                    value: Value::Array(Vec::new()),
                    ty: "Array".to_string(),
                });
            }
            
            let mut values = Vec::new();
            for item in inner.split(',') {
                let item = item.trim();
                if let Ok(n) = item.parse::<i64>() {
                    values.push(Value::Int(n));
                }
            }
            return Ok(ReplOutput::Value {
                value: Value::Array(values),
                ty: "Array".to_string(),
            });
        }

        // Function definition
        if input.starts_with("fn ") || input.starts_with("def ") {
            return self.define_function(input);
        }

        // Binary operations (simplified)
        if input.contains('+') || input.contains('-') || input.contains('*') || input.contains('/') {
            return self.eval_binary(input);
        }

        // If all else fails, treat as undefined variable
        Ok(ReplOutput::Error(format!(
            "undefined variable `{}`",
            input
        )))
    }

    /// Evaluate binary operation
    fn eval_binary(&mut self, input: &str) -> Result<ReplOutput> {
        let operators = ['+', '-', '*', '/'];
        
        for op in operators {
            if let Some(pos) = input.find(op) {
                let left = input[..pos].trim();
                let right = input[pos+1..].trim();
                
                // Evaluate both sides
                let left_val = match self.eval_expression(left)? {
                    ReplOutput::Value { value, .. } => value,
                    _ => return Ok(ReplOutput::Error("Invalid left operand".to_string())),
                };
                
                let right_val = match self.eval_expression(right)? {
                    ReplOutput::Value { value, .. } => value,
                    _ => return Ok(ReplOutput::Error("Invalid right operand".to_string())),
                };
                
                // Perform operation
                match (&left_val, &right_val) {
                    (Value::Int(l), Value::Int(r)) => {
                        let result = match op {
                            '+' => l + r,
                            '-' => l - r,
                            '*' => l * r,
                            '/' => l / r,
                            _ => unreachable!(),
                        };
                        return Ok(ReplOutput::Value {
                            value: Value::Int(result),
                            ty: "i64".to_string(),
                        });
                    }
                    (Value::Float(l), Value::Float(r)) => {
                        let result = match op {
                            '+' => l + r,
                            '-' => l - r,
                            '*' => l * r,
                            '/' => l / r,
                            _ => unreachable!(),
                        };
                        return Ok(ReplOutput::Value {
                            value: Value::Float(result),
                            ty: "f64".to_string(),
                        });
                    }
                    _ => {
                        return Ok(ReplOutput::Error(
                            "Type mismatch in binary operation".to_string()
                        ));
                    }
                }
            }
        }
        
        Ok(ReplOutput::Error("Invalid expression".to_string()))
    }

    /// Define a function
    fn define_function(&mut self, input: &str) -> Result<ReplOutput> {
        // Simplified function definition
        let name = input
            .split('(')
            .next()
            .map(|s| s.trim())
            .and_then(|s| s.split_whitespace().last())
            .unwrap_or("anonymous")
            .to_string();
        
        self.bindings.insert(name.clone(), Value::Function(name.clone()));
        
        Ok(ReplOutput::Value {
            value: Value::Function(name),
            ty: "fn".to_string(),
        })
    }

    /// Get type of expression
    fn type_of(&self, expr: &str) -> Result<ReplOutput> {
        if let Some(value) = self.bindings.get(expr) {
            return Ok(ReplOutput::Info(value.type_name().to_string()));
        }
        
        // Try to parse and get type
        match expr {
            n if n.parse::<i64>().is_ok() => Ok(ReplOutput::Info("i64".to_string())),
            n if n.parse::<f64>().is_ok() => Ok(ReplOutput::Info("f64".to_string())),
            "true" | "false" => Ok(ReplOutput::Info("bool".to_string())),
            s if s.starts_with('"') => Ok(ReplOutput::Info("String".to_string())),
            _ => Ok(ReplOutput::Error(format!("Cannot determine type of `{}`", expr))),
        }
    }

    /// Show documentation for an item
    fn show_doc(&self, item: &str) -> Result<ReplOutput> {
        let doc = match item {
            "Vec" => "pub struct Vec<T>\nA contiguous growable array type.\n\nMethods:\n- new() -> Self\n- push(&mut self, value: T)\n- pop(&mut self) -> Option<T>\n- len(&self) -> usize",
            "String" => "pub struct String\nA UTF-8 encoded, growable string.\n\nMethods:\n- new() -> Self\n- from(s: &str) -> Self\n- push_str(&mut self, s: &str)\n- len(&self) -> usize",
            "Option" => "pub enum Option<T>\n- Some(T)\n- None\n\nRepresents an optional value.",
            "Result" => "pub enum Result<T, E>\n- Ok(T)\n- Err(E)\n\nRepresents a result that can be either success or error.",
            _ => return Ok(ReplOutput::Error(format!("No documentation for `{}`", item))),
        };
        Ok(ReplOutput::Info(doc.to_string()))
    }

    /// Load a file
    fn load_file(&mut self, path: &str) -> Result<ReplOutput> {
        // Simplified file loading
        if std::path::Path::new(path).exists() {
            self.context.imports.push(path.to_string());
            Ok(ReplOutput::Info(format!("Loaded module '{}'", path)))
        } else {
            Ok(ReplOutput::Error(format!("File not found: {}", path)))
        }
    }

    /// Run the REPL loop
    pub fn run(&mut self) -> Result<()> {
        println!("{}", BANNER.bright_cyan());
        println!("Type {} for commands, {} to quit\n", 
            ":help".bright_yellow(), 
            ":quit".bright_yellow()
        );

        let config = Config::builder()
            .history_ignore_space(true)
            .completion_type(CompletionType::List)
            .edit_mode(EditMode::Emacs)
            .build();

        let mut rl: Editor<(), DefaultHistory> = Editor::with_config(config)
            .context("Failed to create line editor")?;

        loop {
            let prompt = format!("{} ", ">>>".bright_green());
            
            match rl.readline(&prompt) {
                Ok(line) => {
                    let _ = rl.add_history_entry(line.as_str());
                    
                    match self.eval(&line)? {
                        ReplOutput::Value { value, ty } => {
                            println!("{}: {} = {}", ty.bright_blue(), value.type_name().bright_magenta(), value);
                        }
                        ReplOutput::Info(msg) => {
                            println!("{}", msg);
                        }
                        ReplOutput::Error(err) => {
                            eprintln!("{} {}", "error:".bright_red(), err);
                        }
                        ReplOutput::Quit => {
                            println!("{}", "Goodbye!".bright_cyan());
                            break;
                        }
                        ReplOutput::Empty => {}
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                }
                Err(ReadlineError::Eof) => {
                    println!("{}", "Goodbye!".bright_cyan());
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {}", err);
                    break;
                }
            }
        }

        Ok(())
    }
}

/// Output from REPL evaluation
pub enum ReplOutput {
    /// A value was produced
    Value { value: Value, ty: String },
    /// Information message
    Info(String),
    /// Error message
    Error(String),
    /// No output (empty input)
    Empty,
    /// Quit command received
    Quit,
}

const BANNER: &str = r#"
███╗   ███╗ ██████╗ ████████╗███████╗
████╗ ████║██╔═══██╗╚══██╔══╝██╔════╝
██╔████╔██║██║   ██║   ██║   █████╗  
██║╚██╔╝██║██║   ██║   ██║   ██╔══╝  
██║ ╚═╝ ██║╚██████╔╝   ██║   ███████╗
╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚══════╝
            REPL v1.0.0
"#;

const HELP_TEXT: &str = r#"
VeZ REPL Commands:
  :help, :h        Show this help message
  :quit, :q        Exit the REPL
  :type <expr>     Show the type of an expression
  :doc <item>      Show documentation for an item
  :load <file>     Load a VeZ file
  :let <name> = <expr>  Bind a value to a name
  :debug           Toggle debug mode
  :clear           Clear all bindings
  :history         Show command history

Examples:
  >>> 5
  i64: i64 = 5
  
  >>> :let x = 10
  x: i64 = 10
  
  >>> x + 5
  i64: i64 = 15
  
  >>> [1, 2, 3]
  Array: Array = [1, 2, 3]
"#;

fn main() -> Result<()> {
    let mut repl = Repl::new();
    repl.run()
}
