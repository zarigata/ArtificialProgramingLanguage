//! Abstract Syntax Tree definitions

/// Root of the AST
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

/// Top-level items
#[derive(Debug, Clone)]
pub enum Item {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Trait(Trait),
    Impl(Impl),
    Use(UsePath),
    Mod(String, Vec<Item>),
}

/// Function declaration
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub attributes: Vec<Attribute>,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub where_clause: Option<WhereClause>,
    pub body: Vec<Stmt>,
}

/// Attribute
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub value: Option<Expr>,
}


/// Function parameter
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

/// Struct declaration
#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<Field>,
    pub where_clause: Option<WhereClause>,
}

/// Struct field
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub ty: Type,
}

/// Enum declaration
#[derive(Debug, Clone)]
pub struct Enum {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<Variant>,
    pub where_clause: Option<WhereClause>,
}

/// Enum variant
#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub data: Option<Type>,
}

/// Type expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Named(String),
    Generic(String, Vec<Type>),
    Reference(Box<Type>),
    MutableReference(Box<Type>),
    Array(Box<Type>, usize),
    Tuple(Vec<Type>),
    Function(Vec<Type>, Box<Type>),
    TraitObject(Vec<String>),
}

/// Statements
#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Option<Type>, Option<Expr>),
    Expr(Expr),
    Return(Option<Expr>),
}

/// Expressions
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    Ident(String),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    MethodCall(Box<Expr>, String, Vec<Expr>),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Block(Vec<Stmt>),
    If(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Match(Box<Expr>, Vec<MatchArm>),
    Loop(Box<Expr>),
    While(Box<Expr>, Box<Expr>),
    For(String, Box<Expr>, Box<Expr>),
    Break(Option<Box<Expr>>),
    Continue,
    Return(Option<Box<Expr>>),
    Array(Vec<Expr>),
    Tuple(Vec<Expr>),
    StructLiteral(String, Vec<(String, Expr)>),
}

/// Literals
#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
    Bool(bool),
}

/// Binary operators
#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
}

/// Unary operators
#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Neg,    // -
    Not,    // !
    Deref,  // *
    Ref,    // &
}

/// Match arm
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

/// Patterns for match expressions
#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Ident(String),
    Tuple(Vec<Pattern>),
    Struct(String, Vec<(String, Pattern)>),
    Enum(String, Vec<Pattern>),
    Or(Vec<Pattern>),
}

/// Generic parameter
#[derive(Debug, Clone)]
pub struct GenericParam {
    pub name: String,
    pub bounds: Vec<String>,
}

/// Where clause
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
}

/// Where predicate
#[derive(Debug, Clone)]
pub struct WherePredicate {
    pub ty: Type,
    pub bounds: Vec<String>,
}

/// Trait declaration
#[derive(Debug, Clone)]
pub struct Trait {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub supertraits: Vec<String>,
    pub items: Vec<TraitItem>,
}

/// Trait item
#[derive(Debug, Clone)]
pub enum TraitItem {
    Function(String, Vec<Param>, Option<Type>),
    Type(String),
}

/// Implementation
#[derive(Debug, Clone)]
pub struct Impl {
    pub generics: Vec<GenericParam>,
    pub trait_name: Option<String>,
    pub self_ty: Type,
    pub where_clause: Option<WhereClause>,
    pub items: Vec<ImplItem>,
}

/// Implementation item
#[derive(Debug, Clone)]
pub enum ImplItem {
    Function(Function),
    Type(String, Type),
}

/// Use path
#[derive(Debug, Clone)]
pub struct UsePath {
    pub segments: Vec<String>,
    pub alias: Option<String>,
}
