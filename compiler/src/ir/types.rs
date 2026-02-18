//! IR type system

use std::fmt;

/// IR type representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    /// Void type (unit)
    Void,
    /// Boolean type
    Bool,
    /// Integer types
    I8,
    I16,
    I32,
    I64,
    I128,
    /// Unsigned integer types
    U8,
    U16,
    U32,
    U64,
    U128,
    /// Floating point types
    F16,
    BF16,
    F32,
    F64,
    /// Pointer type
    Pointer(Box<IrType>),
    /// Array type [T; N]
    Array(Box<IrType>, usize),
    /// Struct type
    Struct(Vec<IrType>),
    /// Function type
    Function(Vec<IrType>, Box<IrType>),
    /// SIMD vector types
    Vec2(Box<IrType>),
    Vec4(Box<IrType>),
    Vec8(Box<IrType>),
    Vec16(Box<IrType>),
    /// Generic vector type with arbitrary element count
    Vector(Box<IrType>, usize),
}

impl IrType {
    /// Get the size of a type in bytes
    pub fn size(&self) -> usize {
        match self {
            IrType::Void => 0,
            IrType::Bool => 1,
            IrType::I8 | IrType::U8 => 1,
            IrType::I16 | IrType::U16 | IrType::F16 | IrType::BF16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::I128 | IrType::U128 => 16,
            IrType::Pointer(_) => 8,
            IrType::Array(elem_ty, count) => elem_ty.size() * count,
            IrType::Struct(fields) => fields.iter().map(|f| f.size()).sum(),
            IrType::Function(_, _) => 8,
            IrType::Vec2(inner) => inner.size() * 2,
            IrType::Vec4(inner) => inner.size() * 4,
            IrType::Vec8(inner) => inner.size() * 8,
            IrType::Vec16(inner) => inner.size() * 16,
            IrType::Vector(inner, count) => inner.size() * count,
        }
    }
    
    /// Get the size of a type in bits
    pub fn size_in_bits(&self) -> usize {
        self.size() * 8
    }
    
    /// Get the alignment of a type in bytes
    pub fn alignment(&self) -> usize {
        match self {
            IrType::Void => 1,
            IrType::Bool => 1,
            IrType::I8 | IrType::U8 => 1,
            IrType::I16 | IrType::U16 | IrType::F16 | IrType::BF16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::I128 | IrType::U128 => 16,
            IrType::Pointer(_) => 8,
            IrType::Array(elem_ty, _) => elem_ty.alignment(),
            IrType::Struct(fields) => {
                fields.iter().map(|f| f.alignment()).max().unwrap_or(1)
            }
            IrType::Function(_, _) => 8,
            IrType::Vec2(inner) => inner.alignment().max(4),
            IrType::Vec4(inner) => inner.alignment().max(8),
            IrType::Vec8(inner) => inner.alignment().max(16),
            IrType::Vec16(inner) => inner.alignment().max(32),
            IrType::Vector(inner, count) => {
                let vec_bytes = inner.size() * count;
                if vec_bytes <= 16 { 16 }
                else if vec_bytes <= 32 { 32 }
                else if vec_bytes <= 64 { 64 }
                else { 64 }
            }
        }
    }
    
    /// Check if type is an integer
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128 |
            IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64 | IrType::U128
        )
    }
    
    /// Check if type is a float
    pub fn is_float(&self) -> bool {
        matches!(self, IrType::F32 | IrType::F64)
    }
    
    /// Check if type is a pointer
    pub fn is_pointer(&self) -> bool {
        matches!(self, IrType::Pointer(_))
    }
    
    /// Check if type is signed
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128
        )
    }
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Void => write!(f, "void"),
            IrType::Bool => write!(f, "bool"),
            IrType::I8 => write!(f, "i8"),
            IrType::I16 => write!(f, "i16"),
            IrType::I32 => write!(f, "i32"),
            IrType::I64 => write!(f, "i64"),
            IrType::I128 => write!(f, "i128"),
            IrType::U8 => write!(f, "u8"),
            IrType::U16 => write!(f, "u16"),
            IrType::U32 => write!(f, "u32"),
            IrType::U64 => write!(f, "u64"),
            IrType::U128 => write!(f, "u128"),
            IrType::F16 => write!(f, "f16"),
            IrType::BF16 => write!(f, "bf16"),
            IrType::F32 => write!(f, "f32"),
            IrType::F64 => write!(f, "f64"),
            IrType::Pointer(inner) => write!(f, "*{}", inner),
            IrType::Array(elem, count) => write!(f, "[{}; {}]", elem, count),
            IrType::Struct(fields) => {
                write!(f, "{{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field)?;
                }
                write!(f, "}}")
            }
            IrType::Function(params, ret) => {
                write!(f, "fn(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", ret)
            }
            IrType::Vec2(inner) => write!(f, "vec2<{}>", inner),
            IrType::Vec4(inner) => write!(f, "vec4<{}>", inner),
            IrType::Vec8(inner) => write!(f, "vec8<{}>", inner),
            IrType::Vec16(inner) => write!(f, "vec16<{}>", inner),
            IrType::Vector(inner, count) => write!(f, "vec{}<{}>", count, inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_sizes() {
        assert_eq!(IrType::Void.size(), 0);
        assert_eq!(IrType::Bool.size(), 1);
        assert_eq!(IrType::I32.size(), 4);
        assert_eq!(IrType::I64.size(), 8);
        assert_eq!(IrType::Pointer(Box::new(IrType::I32)).size(), 8);
    }

    #[test]
    fn test_type_alignment() {
        assert_eq!(IrType::I8.alignment(), 1);
        assert_eq!(IrType::I32.alignment(), 4);
        assert_eq!(IrType::I64.alignment(), 8);
    }

    #[test]
    fn test_type_predicates() {
        assert!(IrType::I32.is_integer());
        assert!(IrType::F64.is_float());
        assert!(IrType::Pointer(Box::new(IrType::I32)).is_pointer());
        assert!(IrType::I32.is_signed());
        assert!(!IrType::U32.is_signed());
    }

    #[test]
    fn test_array_type() {
        let arr_ty = IrType::Array(Box::new(IrType::I32), 10);
        assert_eq!(arr_ty.size(), 40);
    }

    #[test]
    fn test_struct_type() {
        let struct_ty = IrType::Struct(vec![IrType::I32, IrType::I64, IrType::Bool]);
        assert_eq!(struct_ty.size(), 13);
    }
}
