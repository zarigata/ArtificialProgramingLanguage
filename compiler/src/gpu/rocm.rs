//! ROCm (AMD Open Compute) Backend
//! 
//! Provides HIP kernel generation and AMD GPU support for VeZ.
//! Supports AMD Instinct, Radeon Pro, and consumer Radeon GPUs.

use crate::ir::ssa::{self as ir, Function};
use crate::ir::types::IrType;
use crate::ir::instructions::BinaryOp;
use crate::error::Result;

/// AMD GPU architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdGpuArch {
    /// GCN 1.0 - Hawaii, Tahiti
    Gcn1,
    /// GCN 2.0 - Fiji, Tonga
    Gcn2,
    /// GCN 3.0 - Polaris (RX 400/500)
    Gcn3,
    /// GCN 4.0 - Vega
    Gcn4,
    /// GCN 5.0 - Vega 20, Arcturus
    Gcn5,
    /// RDNA 1 - Radeon RX 5000
    Rdna1,
    /// RDNA 2 - Radeon RX 6000
    Rdna2,
    /// RDNA 3 - Radeon RX 7000
    Rdna3,
    /// CDNA 1 - Instinct MI100
    Cdna1,
    /// CDNA 2 - Instinct MI200
    Cdna2,
    /// CDNA 3 - Instinct MI300
    Cdna3,
}

impl AmdGpuArch {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gcn1" | "hawaii" | "tahiti" => AmdGpuArch::Gcn1,
            "gcn2" | "fiji" | "tonga" => AmdGpuArch::Gcn2,
            "gcn3" | "polaris" | "rx400" | "rx500" => AmdGpuArch::Gcn3,
            "gcn4" | "vega" => AmdGpuArch::Gcn4,
            "gcn5" | "vega20" | "arcturus" => AmdGpuArch::Gcn5,
            "rdna1" | "rx5000" | "navi10" => AmdGpuArch::Rdna1,
            "rdna2" | "rx6000" | "navi21" => AmdGpuArch::Rdna2,
            "rdna3" | "rx7000" | "navi31" => AmdGpuArch::Rdna3,
            "cdna1" | "mi100" => AmdGpuArch::Cdna1,
            "cdna2" | "mi200" => AmdGpuArch::Cdna2,
            "cdna3" | "mi300" => AmdGpuArch::Cdna3,
            _ => AmdGpuArch::Rdna3,
        }
    }
    
    pub fn to_rocm_arch(&self) -> &'static str {
        match self {
            AmdGpuArch::Gcn1 => "gfx700",
            AmdGpuArch::Gcn2 => "gfx800",
            AmdGpuArch::Gcn3 => "gfx803",
            AmdGpuArch::Gcn4 => "gfx900",
            AmdGpuArch::Gcn5 => "gfx906",
            AmdGpuArch::Rdna1 => "gfx1010",
            AmdGpuArch::Rdna2 => "gfx1030",
            AmdGpuArch::Rdna3 => "gfx1100",
            AmdGpuArch::Cdna1 => "gfx908",
            AmdGpuArch::Cdna2 => "gfx90a",
            AmdGpuArch::Cdna3 => "gfx940",
        }
    }
    
    pub fn wavefront_size(&self) -> u32 {
        match self {
            AmdGpuArch::Gcn1 | AmdGpuArch::Gcn2 | AmdGpuArch::Gcn3 | 
            AmdGpuArch::Gcn4 | AmdGpuArch::Gcn5 | AmdGpuArch::Cdna1 | 
            AmdGpuArch::Cdna2 | AmdGpuArch::Cdna3 => 64,
            AmdGpuArch::Rdna1 | AmdGpuArch::Rdna2 | AmdGpuArch::Rdna3 => 32,
        }
    }
    
    pub fn supports_fp64(&self) -> bool {
        matches!(self, 
            AmdGpuArch::Gcn4 | AmdGpuArch::Gcn5 | 
            AmdGpuArch::Cdna1 | AmdGpuArch::Cdna2 | AmdGpuArch::Cdna3)
    }
    
    pub fn supports_bf16(&self) -> bool {
        matches!(self, 
            AmdGpuArch::Cdna1 | AmdGpuArch::Cdna2 | AmdGpuArch::Cdna3)
    }
    
    pub fn supports_matrix_cores(&self) -> bool {
        matches!(self, 
            AmdGpuArch::Cdna1 | AmdGpuArch::Cdna2 | AmdGpuArch::Cdna3)
    }
}

/// ROCm kernel configuration
#[derive(Debug, Clone)]
pub struct RocmKernelConfig {
    pub arch: AmdGpuArch,
    pub workgroup_size: (u32, u32, u32),
    pub grid_size: (u32, u32, u32),
    pub shared_memory: usize,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub max_waves_per_cu: u32,
    pub enable_wgp: bool,
    pub enable_cooperative_groups: bool,
}

impl Default for RocmKernelConfig {
    fn default() -> Self {
        RocmKernelConfig {
            arch: AmdGpuArch::Rdna3,
            workgroup_size: (256, 1, 1),
            grid_size: (1, 1, 1),
            shared_memory: 0,
            vgpr_count: 256,
            sgpr_count: 100,
            max_waves_per_cu: 8,
            enable_wgp: false,
            enable_cooperative_groups: false,
        }
    }
}

impl RocmKernelConfig {
    pub fn new(arch: AmdGpuArch) -> Self {
        RocmKernelConfig {
            arch,
            ..Default::default()
        }
    }
    
    pub fn with_workgroup(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup_size = (x, y, z);
        self
    }
    
    pub fn with_grid(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid_size = (x, y, z);
        self
    }
    
    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory = bytes;
        self
    }
    
    pub fn total_workgroup_threads(&self) -> u32 {
        self.workgroup_size.0 * self.workgroup_size.1 * self.workgroup_size.2
    }
    
    pub fn total_grid_threads(&self) -> u32 {
        self.grid_size.0 * self.grid_size.1 * self.grid_size.2
    }
}

/// HIP code generator for ROCm
pub struct HipCodegen {
    config: RocmKernelConfig,
    next_temp: usize,
    indent_level: usize,
}

impl HipCodegen {
    pub fn new(config: RocmKernelConfig) -> Self {
        HipCodegen {
            config,
            next_temp: 0,
            indent_level: 1,
        }
    }
    
    pub fn generate_kernel(&mut self, func: &ir::Function) -> Result<String> {
        let mut code = String::new();
        
        code.push_str(&self.generate_header());
        code.push_str("\n");
        code.push_str(&self.generate_kernel_signature(func));
        code.push_str(" {\n");
        
        self.indent_level = 1;
        code.push_str(&self.generate_thread_indexing());
        code.push_str(&self.generate_kernel_body(func));
        
        code.push_str("}\n");
        
        Ok(code)
    }
    
    fn generate_header(&self) -> String {
        let mut header = String::new();
        
        header.push_str("#include <hip/hip_runtime.h>\n");
        header.push_str("#include <hip/hip_fp16.h>\n");
        
        if self.config.arch.supports_bf16() {
            header.push_str("#include <hip/hip_bfloat16.h>\n");
        }
        
        if self.config.arch.supports_matrix_cores() {
            header.push_str("#include <rocblas/rocblas.h>\n");
        }
        
        header.push_str("\n");
        
        // AMD-specific intrinsics
        header.push_str("// AMD GPU intrinsics\n");
        header.push_str("#define __lds(x) __builtin_amdgcn_ds_bvh_stack_rtn(x)\n");
        header.push_str("#define __ballot(x) __builtin_amdgcn_ballot_w64(x)\n");
        header.push_str("#define __lanemask_gt() __builtin_amdgcn_lanemask_gt()\n");
        header.push_str("\n");
        
        // Wavefront operations
        let wf_size = self.config.arch.wavefront_size();
        header.push_str(&format!("#define WAVEFRONT_SIZE {}\n", wf_size));
        header.push_str("#define WARP_SIZE WAVEFRONT_SIZE\n");
        header.push_str("\n");
        
        header
    }
    
    fn generate_kernel_signature(&self, func: &ir::Function) -> String {
        let mut sig = String::new();
        
        // HIP kernel launch bounds
        let threads = self.config.total_workgroup_threads();
        sig.push_str(&format!(
            "__global__ __launch_bounds__({}) ",
            threads
        ));
        
        sig.push_str(&format!("void {}(", func.name));
        
        for (i, (name, ty)) in func.params.iter().enumerate() {
            if i > 0 {
                sig.push_str(", ");
            }
            sig.push_str(&self.hip_type(ty));
            sig.push_str(" ");
            sig.push_str(name);
        }
        
        sig.push_str(")");
        
        sig
    }
    
    fn generate_thread_indexing(&self) -> String {
        let mut code = String::new();
        let indent = "    ".repeat(self.indent_level);
        
        code.push_str(&format!("{}// Thread and block indexing\n", indent));
        code.push_str(&format!(
            "{}const int threadIdx_x = hipThreadIdx_x;\n", indent
        ));
        code.push_str(&format!(
            "{}const int threadIdx_y = hipThreadIdx_y;\n", indent
        ));
        code.push_str(&format!(
            "{}const int threadIdx_z = hipThreadIdx_z;\n", indent
        ));
        code.push_str(&format!(
            "{}const int blockIdx_x = hipBlockIdx_x;\n", indent
        ));
        code.push_str(&format!(
            "{}const int blockIdx_y = hipBlockIdx_y;\n", indent
        ));
        code.push_str(&format!(
            "{}const int blockIdx_z = hipBlockIdx_z;\n", indent
        ));
        code.push_str(&format!(
            "{}const int blockDim_x = hipBlockDim_x;\n", indent
        ));
        code.push_str(&format!(
            "{}const int blockDim_y = hipBlockDim_y;\n", indent
        ));
        code.push_str(&format!(
            "{}const int blockDim_z = hipBlockDim_z;\n", indent
        ));
        code.push_str("\n");
        
        code.push_str(&format!(
            "{}const int global_idx = blockIdx_x * blockDim_x + threadIdx_x;\n", indent
        ));
        code.push_str(&format!(
            "{}const int global_idy = blockIdx_y * blockDim_y + threadIdx_y;\n", indent
        ));
        code.push_str(&format!(
            "{}const int global_idz = blockIdx_z * blockDim_z + threadIdx_z;\n", indent
        ));
        code.push_str(&format!(
            "{}const int global_size = hipGridDim_x * blockDim_x;\n", indent
        ));
        code.push_str("\n");
        
        code.push_str(&format!(
            "{}const int local_idx = threadIdx_x;\n", indent
        ));
        code.push_str(&format!(
            "{}const int local_idy = threadIdx_y;\n", indent
        ));
        code.push_str(&format!(
            "{}const int local_idz = threadIdx_z;\n", indent
        ));
        code.push_str("\n");
        
        code
    }
    
    fn generate_kernel_body(&mut self, func: &ir::Function) -> String {
        let mut code = String::new();
        let indent = "    ".repeat(self.indent_level);
        
        for block in &func.blocks {
            for (value_id, instruction) in &block.instructions {
                code.push_str(&self.generate_instruction(&indent, value_id, instruction));
            }
        }
        
        code
    }
    
    fn generate_instruction(
        &mut self,
        indent: &str,
        value_id: &ir::ValueId,
        instruction: &crate::ir::instructions::Instruction,
    ) -> String {
        use crate::ir::instructions::Instruction;
        
        match instruction {
            Instruction::Binary { op, lhs, rhs, .. } => {
                let op_str = self.binary_op_str(op);
                format!("{}auto v{} = v{} {} v{};\n", indent, value_id.0, lhs.0, op_str, rhs.0)
            }
            Instruction::Return { value: Some(val) } => {
                format!("{}return v{};\n", indent, val.0)
            }
            Instruction::Return { value: None } => {
                format!("{}return;\n", indent)
            }
            Instruction::Load { ptr, .. } => {
                format!("{}auto v{} = *v{};\n", indent, value_id.0, ptr.0)
            }
            Instruction::Store { ptr, value } => {
                format!("{}*v{} = v{};\n", indent, ptr.0, value.0)
            }
            Instruction::Call { func, args, .. } => {
                let args_str: Vec<String> = args.iter().map(|a| format!("v{}", a.0)).collect();
                format!("{}auto v{} = v{}({});\n", indent, value_id.0, func.0, args_str.join(", "))
            }
            Instruction::Jump { target } => {
                format!("{}goto block_{};\n", indent, target)
            }
            Instruction::Branch { cond, then_block, else_block } => {
                format!(
                    "{}if (v{}) {{ goto block_{}; }} else {{ goto block_{}; }}\n",
                    indent, cond.0, then_block, else_block
                )
            }
            Instruction::Phi { incoming, .. } => {
                let _ = incoming;
                format!("{}// phi v{}\n", indent, value_id.0)
            }
            Instruction::Alloca { .. } => {
                format!("{}// alloca v{}\n", indent, value_id.0)
            }
            Instruction::GetElementPtr { ptr, indices, .. } => {
                let idx_str: Vec<String> = indices.iter().map(|i| format!("v{}", i.0)).collect();
                format!("{}auto v{} = v{}[{}];\n", indent, value_id.0, ptr.0, idx_str.join("]["))
            }
            Instruction::Unary { op, operand, .. } => {
                let op_str = match op {
                    crate::ir::instructions::UnaryOp::Neg => "-",
                    crate::ir::instructions::UnaryOp::Not => "!",
                };
                format!("{}auto v{} = {}v{};\n", indent, value_id.0, op_str, operand.0)
            }
            Instruction::Cast { value, to_ty, .. } => {
                let ty_str = self.hip_type(to_ty);
                format!("{}auto v{} = ({})v{};\n", indent, value_id.0, ty_str, value.0)
            }
            Instruction::Select { cond, true_val, false_val, .. } => {
                format!("{}auto v{} = v{} ? v{} : v{};\n", indent, value_id.0, cond.0, true_val.0, false_val.0)
            }
        }
    }
    
    fn binary_op_str(&self, op: &BinaryOp) -> &'static str {
        match op {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::Xor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
        }
    }
    
    fn hip_type(&self, ty: &IrType) -> String {
        match ty {
            IrType::I8 => "int8_t".to_string(),
            IrType::I16 => "int16_t".to_string(),
            IrType::I32 => "int32_t".to_string(),
            IrType::I64 => "int64_t".to_string(),
            IrType::I128 => "__int128_t".to_string(),
            IrType::U8 => "uint8_t".to_string(),
            IrType::U16 => "uint16_t".to_string(),
            IrType::U32 => "uint32_t".to_string(),
            IrType::U64 => "uint64_t".to_string(),
            IrType::U128 => "__uint128_t".to_string(),
            IrType::F16 => "hip_fp16".to_string(),
            IrType::BF16 => "hip_bfloat16".to_string(),
            IrType::F32 => "float".to_string(),
            IrType::F64 => {
                if self.config.arch.supports_fp64() {
                    "double".to_string()
                } else {
                    "float".to_string()
                }
            }
            IrType::Bool => "bool".to_string(),
            IrType::Pointer(inner) => format!("{}*", self.hip_type(inner)),
            IrType::Void => "void".to_string(),
            IrType::Array(inner, size) => format!("{}[{}]", self.hip_type(inner), size),
            IrType::Struct(_) => "void".to_string(),
            IrType::Function(_, _) => "void*".to_string(),
            IrType::Vec2(inner) => format!("{}2", self.hip_type_base(inner)),
            IrType::Vec4(inner) => format!("{}4", self.hip_type_base(inner)),
            IrType::Vec8(inner) => format!("{}8", self.hip_type_base(inner)),
            IrType::Vec16(inner) => format!("{}16", self.hip_type_base(inner)),
            IrType::Vector(inner, count) => format!("{}{}", self.hip_type_base(inner), count),
        }
    }
    
    fn hip_type_base(&self, ty: &IrType) -> String {
        match ty {
            IrType::I8 => "char".to_string(),
            IrType::I16 => "short".to_string(),
            IrType::I32 => "int".to_string(),
            IrType::I64 => "long".to_string(),
            IrType::U8 => "uchar".to_string(),
            IrType::U16 => "ushort".to_string(),
            IrType::U32 => "uint".to_string(),
            IrType::U64 => "ulong".to_string(),
            IrType::F16 => "half".to_string(),
            IrType::F32 => "float".to_string(),
            IrType::F64 => "double".to_string(),
            _ => "void".to_string(),
        }
    }
    
    pub fn generate_launcher(&self, func: &ir::Function) -> String {
        let mut code = String::new();
        
        // Host-side launcher function
        code.push_str(&format!(
            "extern \"C\" hipError_t launch_{}(\n",
            func.name
        ));
        
        // Parameters
        for (i, (name, ty)) in func.params.iter().enumerate() {
            code.push_str(&format!(
                "    {} {},\n",
                self.hip_type(ty),
                name
            ));
        }
        
        // Grid and block dimensions
        code.push_str("    dim3 grid_dim,\n");
        code.push_str("    dim3 block_dim,\n");
        code.push_str("    hipStream_t stream\n");
        code.push_str(") {\n");
        
        // Kernel launch
        code.push_str(&format!(
            "    hipLaunchKernelGGL({}, \n",
            func.name
        ));
        code.push_str("        grid_dim, block_dim,\n");
        code.push_str(&format!(
            "        {}, // shared memory\n",
            self.config.shared_memory
        ));
        code.push_str("        stream,\n");
        
        // Arguments
        for (name, _) in &func.params {
            code.push_str(&format!("        {},\n", name));
        }
        code.push_str("    );\n\n");
        
        code.push_str("    return hipSuccess;\n");
        code.push_str("}\n");
        
        code
    }
}

/// ROCm memory management
pub struct RocmMemory {
    allocations: Vec<RocmAllocation>,
}

#[derive(Debug, Clone)]
pub struct RocmAllocation {
    pub device_ptr: u64,
    pub host_ptr: Option<u64>,
    pub size: usize,
    pub device_id: u32,
    pub flags: RocmMemoryFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmMemoryFlags {
    pub read_only: bool,
    pub write_only: bool,
    pub coherent: bool,
    pub uncached: bool,
    pub fine_grained: bool,
}

impl Default for RocmMemoryFlags {
    fn default() -> Self {
        RocmMemoryFlags {
            read_only: false,
            write_only: false,
            coherent: true,
            uncached: false,
            fine_grained: false,
        }
    }
}

impl RocmMemory {
    pub fn new() -> Self {
        RocmMemory {
            allocations: Vec::new(),
        }
    }
    
    pub fn allocate(&mut self, size: usize, flags: RocmMemoryFlags) -> Result<RocmAllocation> {
        let alloc = RocmAllocation {
            device_ptr: 0,
            host_ptr: None,
            size,
            device_id: 0,
            flags,
        };
        
        self.allocations.push(alloc.clone());
        Ok(alloc)
    }
    
    pub fn allocate_host(&mut self, size: usize) -> Result<RocmAllocation> {
        let alloc = RocmAllocation {
            device_ptr: 0,
            host_ptr: Some(0),
            size,
            device_id: 0,
            flags: RocmMemoryFlags::default(),
        };
        
        self.allocations.push(alloc.clone());
        Ok(alloc)
    }
    
    pub fn copy_to_device(&self, _src: *const u8, _dst: &RocmAllocation, _size: usize) -> Result<()> {
        Ok(())
    }
    
    pub fn copy_to_host(&self, _src: &RocmAllocation, _dst: *mut u8, _size: usize) -> Result<()> {
        Ok(())
    }
    
    pub fn deallocate(&mut self, alloc: &RocmAllocation) -> Result<()> {
        self.allocations.retain(|a| a.device_ptr != alloc.device_ptr);
        Ok(())
    }
}

impl Default for RocmMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// AMD GPU device information
#[derive(Debug, Clone)]
pub struct AmdGpuDevice {
    pub device_id: u32,
    pub name: String,
    pub arch: AmdGpuArch,
    pub compute_units: u32,
    pub wavefront_size: u32,
    pub workgroup_max_size: u32,
    pub local_memory_size: usize,
    pub global_memory_size: usize,
    pub clock_frequency: u32,
    pub memory_clock: u32,
    pub memory_bus_width: u32,
    pub max_threads_per_cu: u32,
    pub l2_cache_size: usize,
    pub supports_fp64: bool,
    pub supports_bf16: bool,
    pub supports_matrix: bool,
}

impl AmdGpuDevice {
    pub fn mi300x() -> Self {
        AmdGpuDevice {
            device_id: 0,
            name: "AMD Instinct MI300X".to_string(),
            arch: AmdGpuArch::Cdna3,
            compute_units: 304,
            wavefront_size: 64,
            workgroup_max_size: 1024,
            local_memory_size: 64 * 1024,
            global_memory_size: 192 * 1024 * 1024 * 1024,
            clock_frequency: 2100,
            memory_clock: 1300,
            memory_bus_width: 8192,
            max_threads_per_cu: 256,
            l2_cache_size: 8 * 1024 * 1024,
            supports_fp64: true,
            supports_bf16: true,
            supports_matrix: true,
        }
    }
    
    pub fn rx7900xtx() -> Self {
        AmdGpuDevice {
            device_id: 0,
            name: "AMD Radeon RX 7900 XTX".to_string(),
            arch: AmdGpuArch::Rdna3,
            compute_units: 96,
            wavefront_size: 32,
            workgroup_max_size: 1024,
            local_memory_size: 64 * 1024,
            global_memory_size: 24 * 1024 * 1024 * 1024,
            clock_frequency: 2500,
            memory_clock: 2500,
            memory_bus_width: 384,
            max_threads_per_cu: 256,
            l2_cache_size: 6 * 1024 * 1024,
            supports_fp64: false,
            supports_bf16: false,
            supports_matrix: false,
        }
    }
    
    pub fn theoretical_flops(&self) -> f64 {
        let ops_per_cu_per_cycle = if self.supports_matrix { 128.0 } else { 64.0 };
        let cycles_per_second = self.clock_frequency as f64 * 1e6;
        ops_per_cu_per_cycle * self.compute_units as f64 * cycles_per_second * 2.0
    }
    
    pub fn memory_bandwidth(&self) -> f64 {
        (self.memory_bus_width as f64 / 8.0) * (self.memory_clock as f64 * 1e6 * 2.0) / 1e9
    }
}

/// ROCm runtime wrapper
pub struct RocmRuntime {
    devices: Vec<AmdGpuDevice>,
    current_device: u32,
}

impl RocmRuntime {
    pub fn new() -> Self {
        RocmRuntime {
            devices: vec![AmdGpuDevice::rx7900xtx()],
            current_device: 0,
        }
    }
    
    pub fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    pub fn get_device_count(&self) -> u32 {
        self.devices.len() as u32
    }
    
    pub fn get_device(&self, id: u32) -> Option<&AmdGpuDevice> {
        self.devices.get(id as usize)
    }
    
    pub fn set_device(&mut self, id: u32) -> Result<()> {
        if (id as usize) < self.devices.len() {
            self.current_device = id;
        }
        Ok(())
    }
    
    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

impl Default for RocmRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_amd_arch() {
        let arch = AmdGpuArch::from_string("mi300");
        assert_eq!(arch, AmdGpuArch::Cdna3);
        assert!(arch.supports_fp64());
        assert!(arch.supports_bf16());
    }
    
    #[test]
    fn test_rocm_config() {
        let config = RocmKernelConfig::new(AmdGpuArch::Cdna3)
            .with_workgroup(256, 1, 1)
            .with_grid(1024, 1, 1);
        
        assert_eq!(config.total_workgroup_threads(), 256);
        assert_eq!(config.total_grid_threads(), 1024);
    }
    
    #[test]
    fn test_mi300x_specs() {
        let gpu = AmdGpuDevice::mi300x();
        assert_eq!(gpu.compute_units, 304);
        assert!(gpu.global_memory_size > 100 * 1024 * 1024 * 1024);
    }
}
