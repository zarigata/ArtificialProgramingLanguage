# VeZ Compiler Architecture

## Overview

The VeZ compiler is a multi-stage compiler that transforms `.zari` source files into optimized machine code. It follows a traditional compiler pipeline with modern optimizations.

---

## Compiler Pipeline

```
┌─────────────────┐
│  Source (.zari) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Lexer       │  Tokenization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Parser      │  Syntax Analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   AST (Untyped) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Analyzer│  Type Checking, Borrow Checking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AST (Typed)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IR Generator   │  High-level IR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Optimizer     │  IR Optimizations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLVM IR Gen     │  LLVM IR Generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLVM Backend   │  Machine Code Generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Linker       │  Executable/Library
└─────────────────┘
```

---

## Phase 1: Lexical Analysis

### Lexer

**Input**: Source code (UTF-8 text)  
**Output**: Token stream  
**Location**: `compiler/src/lexer/`

#### Responsibilities

1. **Tokenization**: Convert character stream to tokens
2. **Position tracking**: Track line/column for error reporting
3. **Comment handling**: Strip or preserve comments
4. **String processing**: Handle escape sequences, raw strings
5. **Number parsing**: Parse integer/float literals with suffixes

#### Token Structure

```rust
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

pub struct Span {
    pub file: FileId,
    pub start: Position,
    pub end: Position,
}

pub struct Position {
    pub line: u32,
    pub column: u32,
    pub byte_offset: usize,
}
```

#### Token Types

```rust
pub enum TokenKind {
    // Keywords
    Fn, Let, Mut, Const, If, Else, Match, Loop, While, For,
    Struct, Enum, Union, Trait, Impl, Type,
    Pub, Use, Mod, Extern, Unsafe, Async, Await,
    
    // Literals
    IntLiteral(IntKind),
    FloatLiteral(FloatKind),
    StringLiteral,
    CharLiteral,
    BoolLiteral(bool),
    
    // Identifiers
    Ident,
    
    // Operators
    Plus, Minus, Star, Slash, Percent, StarStar,
    Eq, EqEq, Ne, Lt, Le, Gt, Ge,
    And, Or, Not, AndAnd, OrOr,
    Amp, Pipe, Caret, Tilde, Shl, Shr,
    
    // Delimiters
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Semi, Colon, ColonColon, Dot, DotDot, DotDotEq,
    Arrow, FatArrow,
    
    // Special
    Eof,
    Error(String),
}
```

#### Error Recovery

- Skip invalid characters
- Report errors but continue lexing
- Provide helpful error messages

---

## Phase 2: Syntax Analysis

### Parser

**Input**: Token stream  
**Output**: Abstract Syntax Tree (AST)  
**Location**: `compiler/src/parser/`

#### Strategy

**Recursive Descent Parser** with operator precedence parsing (Pratt parser)

#### Responsibilities

1. **Parse declarations**: Functions, structs, enums, traits, impls
2. **Parse expressions**: All expression forms with correct precedence
3. **Parse statements**: Let bindings, assignments, expression statements
4. **Parse patterns**: All pattern forms for match/let
5. **Parse types**: Type expressions with generics
6. **Error recovery**: Synchronize on statement boundaries

#### AST Structure

```rust
pub struct Program {
    pub items: Vec<Item>,
}

pub enum Item {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Trait(Trait),
    Impl(Impl),
    TypeAlias(TypeAlias),
    Const(Const),
    Static(Static),
    Mod(Module),
    Use(Use),
}

pub struct Function {
    pub name: Ident,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Option<Block>,
    pub span: Span,
}

pub enum Expr {
    Literal(Literal),
    Path(Path),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    MethodCall(Box<Expr>, Ident, Vec<Expr>),
    Field(Box<Expr>, Ident),
    Index(Box<Expr>, Box<Expr>),
    Block(Block),
    If(Box<Expr>, Block, Option<Box<Expr>>),
    Match(Box<Expr>, Vec<MatchArm>),
    Loop(Block),
    While(Box<Expr>, Block),
    For(Pattern, Box<Expr>, Block),
    // ... more variants
}
```

#### Error Recovery

- Panic mode: Skip to synchronization point (`;`, `}`)
- Error productions: Parse common mistakes
- Helpful suggestions: "Did you mean...?"

---

## Phase 3: Semantic Analysis

### Type Checker

**Input**: Untyped AST  
**Output**: Typed AST with type annotations  
**Location**: `compiler/src/semantic/typeck/`

#### Responsibilities

1. **Symbol resolution**: Build symbol table, resolve names
2. **Type inference**: Infer types using Hindley-Milner
3. **Type checking**: Verify type correctness
4. **Trait resolution**: Resolve trait implementations
5. **Const evaluation**: Evaluate compile-time constants

#### Symbol Table

```rust
pub struct SymbolTable {
    scopes: Vec<Scope>,
    symbols: HashMap<SymbolId, Symbol>,
}

pub struct Scope {
    parent: Option<ScopeId>,
    symbols: HashMap<String, SymbolId>,
}

pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub ty: TypeId,
    pub span: Span,
}

pub enum SymbolKind {
    Variable,
    Function,
    Type,
    Trait,
    Module,
}
```

#### Type Representation

```rust
pub enum Type {
    Primitive(PrimitiveType),
    Array(Box<Type>, usize),
    Slice(Box<Type>),
    Tuple(Vec<Type>),
    Struct(StructId),
    Enum(EnumId),
    Reference(Lifetime, Mutability, Box<Type>),
    Pointer(Mutability, Box<Type>),
    Function(Vec<Type>, Box<Type>),
    Generic(GenericId),
    Associated(TraitId, Ident),
    Never,
}
```

### Borrow Checker

**Input**: Typed AST  
**Output**: Validated AST with lifetime annotations  
**Location**: `compiler/src/semantic/borrow/`

#### Responsibilities

1. **Lifetime inference**: Infer lifetimes for references
2. **Borrow checking**: Enforce borrowing rules
3. **Move checking**: Track ownership transfers
4. **Drop checking**: Verify drop order is safe

#### Control Flow Graph

```rust
pub struct CFG {
    pub blocks: Vec<BasicBlock>,
    pub edges: Vec<Edge>,
}

pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

pub enum Terminator {
    Return,
    Goto(BlockId),
    If(Value, BlockId, BlockId),
    Switch(Value, Vec<(Value, BlockId)>, BlockId),
}
```

#### Liveness Analysis

- Track which variables are live at each point
- Detect use-after-move errors
- Verify references don't outlive referents

---

## Phase 4: IR Generation

### High-Level IR

**Input**: Typed AST  
**Output**: VeZ IR (SSA form)  
**Location**: `compiler/src/ir/`

#### IR Design

```rust
pub struct IRModule {
    pub functions: Vec<IRFunction>,
    pub globals: Vec<IRGlobal>,
}

pub struct IRFunction {
    pub name: String,
    pub params: Vec<IRParam>,
    pub return_type: Type,
    pub blocks: Vec<IRBlock>,
}

pub struct IRBlock {
    pub label: Label,
    pub instructions: Vec<IRInstruction>,
    pub terminator: IRTerminator,
}

pub enum IRInstruction {
    // Arithmetic
    Add(Value, Value, Value),
    Sub(Value, Value, Value),
    Mul(Value, Value, Value),
    Div(Value, Value, Value),
    
    // Memory
    Load(Value, Value),
    Store(Value, Value),
    Alloca(Value, Type),
    
    // Control flow
    Call(Value, Vec<Value>),
    
    // Type operations
    Cast(Value, Type),
    
    // Other
    Phi(Vec<(Value, Label)>),
}

pub enum IRTerminator {
    Return(Option<Value>),
    Branch(Label),
    CondBranch(Value, Label, Label),
    Switch(Value, Vec<(Value, Label)>, Label),
    Unreachable,
}
```

#### SSA Construction

- Convert to Static Single Assignment form
- Insert phi nodes at control flow joins
- Rename variables to ensure single assignment

---

## Phase 5: Optimization

### Optimizer

**Input**: VeZ IR  
**Output**: Optimized VeZ IR  
**Location**: `compiler/src/optimizer/`

#### Optimization Passes

**Level O0** (Debug):
- No optimizations
- Preserve debug info

**Level O1** (Basic):
- Constant folding
- Dead code elimination
- Simple inlining

**Level O2** (Default):
- All O1 optimizations
- Loop optimizations
- Function inlining
- Common subexpression elimination
- Strength reduction

**Level O3** (Aggressive):
- All O2 optimizations
- Aggressive inlining
- Loop unrolling
- Vectorization (SIMD)
- Interprocedural optimizations

**Level Os** (Size):
- Optimize for binary size
- Minimal inlining
- Code deduplication

#### Pass Manager

```rust
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

pub trait Pass {
    fn name(&self) -> &str;
    fn run(&mut self, module: &mut IRModule) -> bool;
}

// Example passes
pub struct ConstantFolding;
pub struct DeadCodeElimination;
pub struct Inliner;
pub struct LoopOptimizer;
```

---

## Phase 6: Code Generation

### LLVM IR Generator

**Input**: Optimized VeZ IR  
**Output**: LLVM IR  
**Location**: `compiler/src/codegen/`

#### LLVM Integration

```rust
use llvm_sys::*;

pub struct CodeGenerator {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
}

impl CodeGenerator {
    pub fn generate(&mut self, ir: &IRModule) -> LLVMModuleRef {
        for func in &ir.functions {
            self.generate_function(func);
        }
        self.module
    }
    
    fn generate_function(&mut self, func: &IRFunction) {
        // Create LLVM function
        // Generate basic blocks
        // Generate instructions
        // Add debug info
    }
}
```

#### Debug Information

- Generate DWARF debug info
- Source location mapping
- Variable names and types
- Stack traces

---

## Phase 7: Backend

### LLVM Backend

**Input**: LLVM IR  
**Output**: Object file (.o)  
**Location**: LLVM (external)

#### Target Platforms

- **x86_64**: Linux, Windows, macOS
- **ARM64**: Linux, macOS, iOS, Android
- **RISC-V**: Linux (future)
- **WASM**: WebAssembly (future)

#### Optimizations

- Register allocation
- Instruction selection
- Instruction scheduling
- Peephole optimizations

---

## Phase 8: Linking

### Linker

**Input**: Object files  
**Output**: Executable or library  
**Location**: `compiler/src/linker/`

#### Linking Modes

**Static linking**:
- All dependencies embedded
- Larger binary
- No runtime dependencies

**Dynamic linking**:
- Shared libraries
- Smaller binary
- Runtime dependencies

**Link-Time Optimization (LTO)**:
- Whole-program optimization
- Cross-module inlining
- Dead code elimination

---

## Compiler Driver

### Main Driver

**Location**: `compiler/src/driver/`

```rust
pub struct Compiler {
    config: CompilerConfig,
    session: Session,
}

pub struct CompilerConfig {
    pub input_files: Vec<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub optimization_level: OptLevel,
    pub target: Target,
    pub emit: EmitType,
}

pub enum EmitType {
    Executable,
    Library,
    Object,
    LlvmIr,
    Assembly,
}

impl Compiler {
    pub fn compile(&mut self) -> Result<(), Error> {
        // 1. Lex all files
        let tokens = self.lex()?;
        
        // 2. Parse to AST
        let ast = self.parse(tokens)?;
        
        // 3. Type check
        let typed_ast = self.type_check(ast)?;
        
        // 4. Borrow check
        self.borrow_check(&typed_ast)?;
        
        // 5. Generate IR
        let ir = self.generate_ir(typed_ast)?;
        
        // 6. Optimize
        let optimized_ir = self.optimize(ir)?;
        
        // 7. Generate code
        let llvm_ir = self.generate_llvm(optimized_ir)?;
        
        // 8. Compile to object
        let object = self.compile_llvm(llvm_ir)?;
        
        // 9. Link
        self.link(object)?;
        
        Ok(())
    }
}
```

---

## Error Reporting

### Error System

```rust
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Span,
    pub suggestions: Vec<Suggestion>,
}

pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Help,
}

pub struct Suggestion {
    pub message: String,
    pub replacement: Option<String>,
    pub span: Span,
}
```

### Error Messages

**Good error message**:
```
error: mismatched types
  --> example.zari:5:9
   |
5  |     let x: i32 = "hello";
   |                  ^^^^^^^ expected `i32`, found `&str`
   |
help: try converting the string to a number
   |
5  |     let x: i32 = "hello".parse()?;
   |                  ^^^^^^^^^^^^^^^^
```

---

## Incremental Compilation

### Caching Strategy

1. **Parse cache**: Cache AST per file
2. **Type cache**: Cache type information
3. **IR cache**: Cache generated IR
4. **Dependency tracking**: Recompile only changed modules

### Query System

```rust
pub trait QueryContext {
    fn parse_file(&self, file: FileId) -> &AST;
    fn type_check(&self, ast: &AST) -> &TypedAST;
    fn generate_ir(&self, ast: &TypedAST) -> &IR;
}
```

---

## Parallel Compilation

### Parallelization Points

1. **Lexing**: Parallel per file
2. **Parsing**: Parallel per file
3. **Type checking**: Parallel per module (with synchronization)
4. **Code generation**: Parallel per function
5. **Optimization**: Parallel per function

---

## Build System Integration

### Cargo-like Build Tool

```toml
[package]
name = "my_project"
version = "0.1.0"

[dependencies]
std = "1.0"

[build]
optimization = "2"
target = "x86_64-unknown-linux-gnu"
```

---

## Testing Infrastructure

### Compiler Tests

1. **Unit tests**: Test individual components
2. **Integration tests**: Test full pipeline
3. **Regression tests**: Prevent regressions
4. **Fuzzing**: Find edge cases

### Test Harness

```rust
#[test]
fn test_parse_function() {
    let source = "fn main() { }";
    let tokens = lex(source);
    let ast = parse(tokens);
    assert!(ast.is_ok());
}
```

---

## Performance Targets

### Compilation Speed

- **Small projects** (<10K LOC): <1 second
- **Medium projects** (100K LOC): <10 seconds
- **Large projects** (1M LOC): <2 minutes

### Memory Usage

- **Peak memory**: <2GB for typical projects
- **Incremental builds**: <500MB

---

## Summary

The VeZ compiler architecture provides:
- **Modularity**: Clear separation of phases
- **Performance**: Fast compilation with optimizations
- **Correctness**: Strong type and memory safety checks
- **Debuggability**: Excellent error messages and debug info

---

**Next Steps**: Begin implementation in `compiler/src/`
