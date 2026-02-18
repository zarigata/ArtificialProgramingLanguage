use std::fmt;

/// A type in the VeZ language
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// A named type, like `i32`, `String`, or a struct name.
    Named(String),

    /// A generic type, like `Vec<T>`.
    Generic(String, Vec<Type>),

    /// A reference to another type.
    Reference(Box<Type>),

    /// A mutable reference to another type.
    MutableReference(Box<Type>),

    /// An array of a specific type and size.
    Array(Box<Type>, u64),

    /// A tuple of types.
    Tuple(Vec<Type>),

    /// A function type.
    Function(Vec<Type>, Box<Type>),

    /// A type variable, used for type inference.
    TypeVar(u32),

    /// A trait object, like `dyn Display`.
    TraitObject(Vec<String>),

    /// The `self` type.
    SelfType,

    /// The unknown type, used when type inference fails.
    Unknown,
}

impl Type {
    /// Returns `true` if the type is a primitive type.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Type::Named(name) if matches!(
                name.as_str(),
                "bool" | "char" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "f32" | "f64"
            )
        )
    }

    /// Returns `true` if the type is a numeric type.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Type::Named(name) if matches!(
                name.as_str(),
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "f32" | "f64"
            )
        )
    }

    /// Returns `true` if the type is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::Named(name) if matches!(
                name.as_str(),
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
            )
        )
    }

    /// Returns `true` if the type is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Type::Named(name) if matches!(name.as_str(), "f32" | "f64")
        )
    }

    /// Returns `true` if the type is a boolean type.
    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Named(name) if name == "bool")
    }

    /// Returns `true` if the type is a char type.
    pub fn is_char(&self) -> bool {
        matches!(self, Type::Named(name) if name == "char")
    }

    /// Returns `true` if the type is a reference type.
    pub fn is_reference(&self) -> bool {
        matches!(self, Type::Reference(_))
    }

    /// Returns `true` if the type is a mutable reference type.
    pub fn is_mutable_reference(&self) -> bool {
        matches!(self, Type::MutableReference(_))
    }

    /// Returns `true` if the type is an array type.
    pub fn is_array(&self) -> bool {
        matches!(self, Type::Array(_, _))
    }

    /// Returns `true` if the type is a tuple type.
    pub fn is_tuple(&self) -> bool {
        matches!(self, Type::Tuple(_))
    }

    /// Returns `true` if the type is a function type.
    pub fn is_function(&self) -> bool {
        matches!(self, Type::Function(_, _))
    }

    /// Returns `true` if the type is a trait object type.
    pub fn is_trait_object(&self) -> bool {
        matches!(self, Type::TraitObject(_))
    }

    /// Returns `true` if the type is the `self` type.
    pub fn is_self(&self) -> bool {
        matches!(self, Type::SelfType)
    }

    /// Returns `true` if the type is the unknown type.
    pub fn is_unknown(&self) -> bool {
        matches!(self, Type::Unknown)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Named(name) => write!(f, "{}", name),
            Type::Generic(name, args) => {
                write!(f, "{}<", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            }
            Type::Reference(ty) => write!(f, "&{}", ty),
            Type::MutableReference(ty) => write!(f, "&mut {}", ty),
            Type::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            Type::Tuple(tys) => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Type::Function(params, ret) => {
                write!(f, "fn(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::TypeVar(id) => write!(f, "'{}", id),
            Type::TraitObject(traits) => {
                write!(f, "dyn {}", traits.join(" + "))
            }
            Type::SelfType => write!(f, "Self"),
            Type::Unknown => write!(f, "unknown"),
        }
    }
}
