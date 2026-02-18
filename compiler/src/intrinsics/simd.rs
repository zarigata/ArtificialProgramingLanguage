//! SIMD Intrinsics for AI-optimized vector operations

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorWidth {
    Sse128,
    Avx256,
    Avx512,
    Neon128,
}

impl VectorWidth {
    pub fn bytes(&self) -> usize {
        match self {
            VectorWidth::Sse128 | VectorWidth::Neon128 => 16,
            VectorWidth::Avx256 => 32,
            VectorWidth::Avx512 => 64,
        }
    }
    
    pub fn f32_count(&self) -> usize {
        self.bytes() / 4
    }
    
    pub fn f64_count(&self) -> usize {
        self.bytes() / 8
    }
    
    pub fn i32_count(&self) -> usize {
        self.bytes() / 4
    }
    
    pub fn i64_count(&self) -> usize {
        self.bytes() / 8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    AddF32,
    AddF64,
    AddI32,
    AddI64,
    SubF32,
    SubF64,
    MulF32,
    MulF64,
    DivF32,
    DivF64,
    FmaF32,
    FmaF64,
    SqrtF32,
    SqrtF64,
    RsqrtF32,
    RecipF32,
    MaxF32,
    MaxF64,
    MinF32,
    MinF64,
    And,
    Or,
    Xor,
    Not,
    AndNot,
    ShiftLeft,
    ShiftRight,
    RotateLeft,
    RotateRight,
    Shuffle,
    Permute,
    Blend,
    CompareEq,
    CompareLt,
    CompareLe,
    CompareGt,
    CompareGe,
    CompareNe,
    Gather,
    Scatter,
    Load,
    Store,
    LoadAligned,
    StoreAligned,
    MaskLoad,
    MaskStore,
    ReduceAdd,
    ReduceMul,
    ReduceMax,
    ReduceMin,
    HorizontalAdd,
    DotProduct,
    MatrixMultiply,
}

impl fmt::Display for SimdOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdOp::AddF32 => write!(f, "vadd.f32"),
            SimdOp::AddF64 => write!(f, "vadd.f64"),
            SimdOp::AddI32 => write!(f, "vadd.i32"),
            SimdOp::AddI64 => write!(f, "vadd.i64"),
            SimdOp::SubF32 => write!(f, "vsub.f32"),
            SimdOp::SubF64 => write!(f, "vsub.f64"),
            SimdOp::MulF32 => write!(f, "vmul.f32"),
            SimdOp::MulF64 => write!(f, "vmul.f64"),
            SimdOp::DivF32 => write!(f, "vdiv.f32"),
            SimdOp::DivF64 => write!(f, "vdiv.f64"),
            SimdOp::FmaF32 => write!(f, "vfma.f32"),
            SimdOp::FmaF64 => write!(f, "vfma.f64"),
            SimdOp::SqrtF32 => write!(f, "vsqrt.f32"),
            SimdOp::SqrtF64 => write!(f, "vsqrt.f64"),
            SimdOp::RsqrtF32 => write!(f, "vrsqrt.f32"),
            SimdOp::RecipF32 => write!(f, "vrcp.f32"),
            SimdOp::MaxF32 => write!(f, "vmax.f32"),
            SimdOp::MaxF64 => write!(f, "vmax.f64"),
            SimdOp::MinF32 => write!(f, "vmin.f32"),
            SimdOp::MinF64 => write!(f, "vmin.f64"),
            SimdOp::Load => write!(f, "vload"),
            SimdOp::Store => write!(f, "vstore"),
            SimdOp::LoadAligned => write!(f, "vload.aligned"),
            SimdOp::StoreAligned => write!(f, "vstore.aligned"),
            SimdOp::DotProduct => write!(f, "vdot"),
            SimdOp::MatrixMultiply => write!(f, "vmatmul"),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub struct SimdIntrinsic {
    pub op: SimdOp,
    pub width: VectorWidth,
    pub inputs: usize,
    pub outputs: usize,
    pub latency: u32,
    pub throughput: u32,
}

impl SimdIntrinsic {
    pub fn new(op: SimdOp, width: VectorWidth) -> Self {
        let (inputs, outputs, latency, throughput) = Self::characteristics(&op);
        SimdIntrinsic { op, width, inputs, outputs, latency, throughput }
    }
    
    fn characteristics(op: &SimdOp) -> (usize, usize, u32, u32) {
        match op {
            SimdOp::AddF32 | SimdOp::AddF64 | SimdOp::AddI32 | SimdOp::AddI64 => (2, 1, 3, 1),
            SimdOp::SubF32 | SimdOp::SubF64 => (2, 1, 3, 1),
            SimdOp::MulF32 | SimdOp::MulF64 => (2, 1, 4, 1),
            SimdOp::DivF32 | SimdOp::DivF64 => (2, 1, 14, 5),
            SimdOp::FmaF32 | SimdOp::FmaF64 => (3, 1, 4, 1),
            SimdOp::SqrtF32 | SimdOp::SqrtF64 => (1, 1, 14, 7),
            SimdOp::Load | SimdOp::LoadAligned => (1, 1, 5, 1),
            SimdOp::Store | SimdOp::StoreAligned => (2, 1, 1, 1),
            SimdOp::DotProduct => (2, 1, 5, 1),
            SimdOp::MatrixMultiply => (3, 1, 15, 2),
            _ => (2, 1, 3, 1),
        }
    }
    
    pub fn to_llvm_intrinsic(&self) -> String {
        let prefix = match self.width {
            VectorWidth::Sse128 => "llvm.x86.sse",
            VectorWidth::Avx256 => "llvm.x86.avx",
            VectorWidth::Avx512 => "llvm.x86.avx512",
            VectorWidth::Neon128 => "llvm.aarch64.neon",
        };
        
        match self.op {
            SimdOp::AddF32 => format!("{}.add.ps", prefix),
            SimdOp::AddF64 => format!("{}.add.pd", prefix),
            SimdOp::MulF32 => format!("{}.mul.ps", prefix),
            SimdOp::MulF64 => format!("{}.mul.pd", prefix),
            SimdOp::FmaF32 => format!("{}.fmadd.ps", prefix),
            SimdOp::FmaF64 => format!("{}.fmadd.pd", prefix),
            SimdOp::SqrtF32 => format!("{}.sqrt.ps", prefix),
            SimdOp::SqrtF64 => format!("{}.sqrt.pd", prefix),
            _ => format!("{}.generic", prefix),
        }
    }
}

pub fn avx2_fma(a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = a[i] * b[i] + c[i];
    }
    result
}

pub fn avx512_dot(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn neon_load(ptr: *const f32) -> [f32; 4] {
    unsafe {
        let slice = std::slice::from_raw_parts(ptr, 4);
        [slice[0], slice[1], slice[2], slice[3]]
    }
}

pub fn neon_store(ptr: *mut f32, val: [f32; 4]) {
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr, 4);
        slice.copy_from_slice(&val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_width() {
        assert_eq!(VectorWidth::Avx256.bytes(), 32);
        assert_eq!(VectorWidth::Avx256.f32_count(), 8);
        assert_eq!(VectorWidth::Avx256.f64_count(), 4);
    }

    #[test]
    fn test_simd_intrinsic() {
        let intr = SimdIntrinsic::new(SimdOp::FmaF32, VectorWidth::Avx256);
        assert_eq!(intr.inputs, 3);
        assert_eq!(intr.outputs, 1);
    }

    #[test]
    fn test_fma() {
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];
        let c = [3.0f32; 8];
        let result = avx2_fma(&a, &b, &c);
        for v in result {
            assert_eq!(v, 5.0);
        }
    }
}
