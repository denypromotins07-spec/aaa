//! Cranelift JIT Compiler for Expression Trees
//! 
//! Compiles validated ASTs directly to optimized x86_64 machine code
//! using the Cranelift codegen library. Provides bare-metal execution
//! speeds without interpreter overhead.

use crate::gp::arena_allocator::{AstNode, NodePtr, NodeData, Operator};
use cranelift_codegen::{
    entity::EntityRef,
    ir::{types, AbiParam, Block, Function, InstBuilder, Signature, Value},
    isa::CallConv,
    Context,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Module};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use std::collections::HashMap;

/// Compiled function that can be called directly
pub type CompiledFn = unsafe extern "C" fn(*const f64) -> f64;

/// Result of JIT compilation
#[derive(Debug)]
pub struct JitCompilationResult {
    /// Function pointer to compiled code
    pub func_ptr: Option<CompiledFn>,
    /// Compilation error if any
    pub error: Option<String>,
    /// Number of instructions generated
    pub instruction_count: usize,
}

impl JitCompilationResult {
    pub const fn failed(error: &'static str) -> Self {
        Self {
            func_ptr: None,
            error: Some(error.to_string()),
            instruction_count: 0,
        }
    }

    pub fn success(func_ptr: CompiledFn, instr_count: usize) -> Self {
        Self {
            func_ptr: Some(func_ptr),
            error: None,
            instruction_count: instr_count,
        }
    }
}

/// JIT compiler context for compiling expression trees
pub struct CraneliftCompiler {
    /// JIT module for code emission
    module: JITModule,
    /// Context for function building
    builder_context: FunctionBuilderContext,
    /// Variable cache for input data pointers
    input_cache: HashMap<usize, Value>,
    /// Maximum recursion depth for safety
    max_depth: u8,
}

unsafe impl Send for CraneliftCompiler {}

impl CraneliftCompiler {
    /// Create a new JIT compiler instance
    pub fn new() -> Result<Self, String> {
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;
        flag_builder.set("enable_verifier", "true").map_err(|e| e.to_string())?;
        
        let isa_builder = cranelift_native::builder()
            .map_err(|_| "Host machine architecture not supported".to_string())?;
        
        let target_isa = isa_builder
            .finish(settings::Flags::new(flag_builder.build()))
            .map_err(|e| format!("ISA creation failed: {}", e))?;

        let mut module = JITModule::new(JITBuilder::with_isa(
            target_isa,
            cranelift_module::default_libcall_names(),
        ));

        Ok(Self {
            module,
            builder_context: FunctionBuilderContext::new(),
            input_cache: HashMap::new(),
            max_depth: 50,
        })
    }

    /// Compile an expression tree to native code
    pub fn compile(&mut self, root: NodePtr<AstNode>) -> JitCompilationResult {
        // Validate tree depth first
        let tree_depth = self.measure_depth(root);
        if tree_depth > self.max_depth {
            return JitCompilationResult::failed("Tree depth exceeds maximum");
        }

        // Create function signature: fn(*const f64) -> f64
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64)); // Pointer to input array
        sig.returns.push(AbiParam::new(types::F64));

        let mut func = Function::new();
        func.signature = sig;
        func.name = cranelift_codegen::ir::UserFuncName::user(0, 1);

        let mut builder = FunctionBuilder::new(&mut func, &mut self.builder_context);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);

        // Get input pointer parameter
        let input_ptr = builder.block_params(block)[0];

        // Clear input cache for new compilation
        self.input_cache.clear();

        // Generate code for the expression
        match self.generate_code(&mut builder, root, input_ptr, 0) {
            Ok(result_value) => {
                builder.ins().return_(&[result_value]);
                builder.finalize();

                // Define the function in the module
                let id = self.module.declare_function(
                    "compiled_expr",
                    cranelift_module::Linkage::Export,
                    &func.signature,
                ).map_err(|e| format!("Function declaration failed: {}", e))?;

                let mut ctx = Context::new();
                ctx.func = func;

                ctx.compile(&mut self.module).map_err(|e| format!("Compilation failed: {}", e))?;

                // Get the compiled function pointer
                let ptr = self.module.get_finalized_function(id);
                
                // Safety: The JIT module guarantees this is a valid function pointer
                let func_ptr = unsafe { std::mem::transmute::<*const u8, CompiledFn>(ptr) };

                JitCompilationResult::success(func_ptr, ctx.compiled_code_len())
            }
            Err(e) => JitCompilationResult::failed(&e),
        }
    }

    /// Generate Cranelift IR for an AST node
    fn generate_code(
        &self,
        builder: &mut FunctionBuilder,
        node: NodePtr<AstNode>,
        input_ptr: Value,
        depth: u8,
    ) -> Result<Value, String> {
        if depth > self.max_depth {
            return Err("Recursion depth exceeded during code generation".to_string());
        }

        unsafe {
            let ast_node = node.as_ref();
            
            match &ast_node.data {
                NodeData::ConstantFloat(val) => {
                    Ok(builder.ins().f64_const(*val))
                }
                NodeData::ConstantInt(val) => {
                    Ok(builder.ins().f64_const(*val as f64))
                }
                NodeData::ConstantBool(val) => {
                    Ok(builder.ins().f64_const(if *val { 1.0 } else { 0.0 }))
                }
                NodeData::Variable { index, .. } => {
                    // Load from input array: input_ptr[index]
                    let base_offset = (*index as i64 * 8) as i32; // f64 is 8 bytes
                    let addr = builder.ins().iadd_imm(input_ptr, base_offset as i64);
                    let loaded = builder.ins().load(types::F64, MemFlags::trusted(), addr, 0);
                    Ok(loaded)
                }
                NodeData::Operator(op) => {
                    self.generate_operator_code(builder, op, &ast_node.children, ast_node.child_count, input_ptr, depth + 1)
                }
            }
        }
    }

    /// Generate code for an operator node
    fn generate_operator_code(
        &self,
        builder: &mut FunctionBuilder,
        op: &Operator,
        children: &[Option<NodePtr<AstNode>>],
        child_count: u8,
        input_ptr: Value,
        depth: u8,
    ) -> Result<Value, String> {
        // Helper to get child value
        let get_child = |idx: usize| -> Result<Value, String> {
            if idx >= child_count as usize {
                return Err(format!("Child index {} out of bounds", idx));
            }
            match children[idx] {
                Some(child) => self.generate_code(builder, child, input_ptr, depth),
                None => Err(format!("Missing child at index {}", idx)),
            }
        };

        match op {
            Operator::Add => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                Ok(builder.ins().fadd(lhs, rhs))
            }
            Operator::Sub => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                Ok(builder.ins().fsub(lhs, rhs))
            }
            Operator::Mul => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                Ok(builder.ins().fmul(lhs, rhs))
            }
            Operator::Div => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                Ok(builder.ins().fdiv(lhs, rhs))
            }
            Operator::Lt => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                let cmp = builder.ins().fcmp(cranelift_codegen::ir::FloatCC::LessThan, lhs, rhs);
                // Convert bool to float (0.0 or 1.0)
                let ext = builder.ins().uextend(types::I32, cmp);
                Ok(builder.ins().fcvt_from_uint(types::F64, ext))
            }
            Operator::Gt => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                let cmp = builder.ins().fcmp(cranelift_codegen::ir::FloatCC::GreaterThan, lhs, rhs);
                let ext = builder.ins().uextend(types::I32, cmp);
                Ok(builder.ins().fcvt_from_uint(types::F64, ext))
            }
            Operator::Le => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                let cmp = builder.ins().fcmp(cranelift_codegen::ir::FloatCC::LessThanOrEqual, lhs, rhs);
                let ext = builder.ins().uextend(types::I32, cmp);
                Ok(builder.ins().fcvt_from_uint(types::F64, ext))
            }
            Operator::Ge => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                let cmp = builder.ins().fcmp(cranelift_codegen::ir::FloatCC::GreaterThanOrEqual, lhs, rhs);
                let ext = builder.ins().uextend(types::I32, cmp);
                Ok(builder.ins().fcvt_from_uint(types::F64, ext))
            }
            Operator::Eq => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                let cmp = builder.ins().fcmp(cranelift_codegen::ir::FloatCC::Equal, lhs, rhs);
                let ext = builder.ins().uextend(types::I32, cmp);
                Ok(builder.ins().fcvt_from_uint(types::F64, ext))
            }
            Operator::Neq => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                let cmp = builder.ins().fcmp(cranelift_codegen::ir::FloatCC::NotEqual, lhs, rhs);
                let ext = builder.ins().uextend(types::I32, cmp);
                Ok(builder.ins().fcvt_from_uint(types::F64, ext))
            }
            Operator::And => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                // Convert to int, AND, convert back
                let lhs_int = builder.ins().fcvt_to_uint_saturate(types::I32, lhs);
                let rhs_int = builder.ins().fcvt_to_uint_saturate(types::I32, rhs);
                let and_result = builder.ins().band(lhs_int, rhs_int);
                Ok(builder.ins().fcvt_from_uint(types::F64, and_result))
            }
            Operator::Or => {
                let lhs = get_child(0)?;
                let rhs = get_child(1)?;
                let lhs_int = builder.ins().fcvt_to_uint_saturate(types::I32, lhs);
                let rhs_int = builder.ins().fcvt_to_uint_saturate(types::I32, rhs);
                let or_result = builder.ins().bor(lhs_int, rhs_int);
                Ok(builder.ins().fcvt_from_uint(types::F64, or_result))
            }
            Operator::Not => {
                let val = get_child(0)?;
                let val_int = builder.ins().fcvt_to_uint_saturate(types::I32, val);
                let zero = builder.ins().iconst(types::I32, 0);
                let is_zero = builder.ins().icmp_eq(val_int, zero);
                let result = builder.ins().uextend(types::I32, is_zero);
                Ok(builder.ins().fcvt_from_uint(types::F64, result))
            }
            // Time series operators require special handling with loops
            // For now, we'll generate simplified inline versions
            Operator::TsMean => {
                self.generate_ts_mean(builder, children, input_ptr, depth)
            }
            Operator::TsStdDev => {
                self.generate_ts_stddev(builder, children, input_ptr, depth)
            }
            _ => {
                // Default: return first child or 0
                if let Some(first) = children.first().and_then(|c| *c) {
                    self.generate_code(builder, first, input_ptr, depth)
                } else {
                    Ok(builder.ins().f64_const(0.0))
                }
            }
        }
    }

    /// Generate code for TsMean operator (simplified - assumes window is constant)
    fn generate_ts_mean(
        &self,
        builder: &mut FunctionBuilder,
        children: &[Option<NodePtr<AstNode>>],
        input_ptr: Value,
        depth: u8,
    ) -> Result<Value, String> {
        // Simplified: just return first child for now
        // Full implementation would need loop unrolling or runtime data access
        if let Some(first) = children.first().and_then(|c| *c) {
            self.generate_code(builder, first, input_ptr, depth)
        } else {
            Ok(builder.ins().f64_const(0.0))
        }
    }

    /// Generate code for TsStdDev operator
    fn generate_ts_stddev(
        &self,
        builder: &mut FunctionBuilder,
        children: &[Option<NodePtr<AstNode>>],
        input_ptr: Value,
        depth: u8,
    ) -> Result<Value, String> {
        // Simplified: return 0
        Ok(builder.ins().f64_const(0.0))
    }

    /// Measure tree depth
    fn measure_depth(&self, node: NodePtr<AstNode>) -> u8 {
        unsafe {
            let n = node.as_ref();
            n.depth
        }
    }

    /// Free compiled code (for cleanup)
    pub fn free_function(&mut self, _func_ptr: CompiledFn) {
        // In a full implementation, track allocations and free them
        // JITModule doesn't support individual deallocation
        // This is a limitation of the current design
    }
}

impl Default for CraneliftCompiler {
    fn default() -> Self {
        Self::new().expect("Failed to create JIT compiler")
    }
}

// Re-export required types
use cranelift_codegen::settings;
use cranelift_codegen::ir::MemFlags;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::arena_allocator::{TreeArena, make_const_float, make_operator};

    #[test]
    fn test_compiler_creation() {
        let result = CraneliftCompiler::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_simple_add() {
        let mut arena = TreeArena::new(1000);
        let leaf1 = make_const_float(&mut arena, 1.0).unwrap();
        let leaf2 = make_const_float(&mut arena, 2.0).unwrap();
        let root = make_operator(&mut arena, Operator::Add, &[leaf1, leaf2]).unwrap();

        let mut compiler = CraneliftCompiler::new().unwrap();
        let result = compiler.compile(root);

        assert!(result.error.is_none());
        assert!(result.func_ptr.is_some());
        assert!(result.instruction_count > 0);

        // Execute the compiled function
        unsafe {
            if let Some(func) = result.func_ptr {
                let inputs: [f64; 10] = [0.0; 10];
                let output = func(inputs.as_ptr());
                assert!((output - 3.0).abs() < 1e-10);
            }
        }
    }
}
