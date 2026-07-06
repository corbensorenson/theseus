use symliquid_core::error::Result;
use symliquid_core::modules::vsa::VSAMemory;
use symliquid_core::tensor::Tensor;

#[cfg(feature = "cuda")]
const VSA_KERNELS: &str = include_str!("../kernels/vsa_kernels.cu");

pub fn bind_cpu_parity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    VSAMemory::bind(a, b)
}

pub fn bundle_cpu_parity(memory: &Tensor, x: &Tensor, decay: f32) -> Result<Tensor> {
    VSAMemory::bundle(memory, x, decay)
}

#[cfg(feature = "cuda")]
pub fn bind_cuda(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    a.ensure_same_shape(b, "CUDA VSA bind")?;
    let ctx = CudaContext::new(0).map_err(cuda_error)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(VSA_KERNELS).map_err(cuda_error)?;
    let module = ctx.load_module(ptx).map_err(cuda_error)?;
    let function = module
        .load_function("vsa_bind_kernel")
        .map_err(cuda_error)?;
    let a_dev = stream.clone_htod(&a.data).map_err(cuda_error)?;
    let b_dev = stream.clone_htod(&b.data).map_err(cuda_error)?;
    let mut out_dev = a_dev.clone();
    let n = a.len() as i32;

    let mut args = stream.launch_builder(&function);
    args.arg(&a_dev);
    args.arg(&b_dev);
    args.arg(&mut out_dev);
    args.arg(&n);
    unsafe {
        args.launch(LaunchConfig::for_num_elems(n as u32))
            .map_err(cuda_error)?;
    }
    let data = stream.clone_dtoh(&out_dev).map_err(cuda_error)?;
    Tensor::new(a.rows, a.cols, data)
}

#[cfg(feature = "cuda")]
pub fn bundle_cuda(memory: &Tensor, x: &Tensor, decay: f32) -> Result<Tensor> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    memory.ensure_same_shape(x, "CUDA VSA bundle")?;
    let ctx = CudaContext::new(0).map_err(cuda_error)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(VSA_KERNELS).map_err(cuda_error)?;
    let module = ctx.load_module(ptx).map_err(cuda_error)?;
    let function = module
        .load_function("vsa_bundle_kernel")
        .map_err(cuda_error)?;
    let memory_dev = stream.clone_htod(&memory.data).map_err(cuda_error)?;
    let x_dev = stream.clone_htod(&x.data).map_err(cuda_error)?;
    let mut out_dev = memory_dev.clone();
    let n = memory.len() as i32;

    let mut args = stream.launch_builder(&function);
    args.arg(&memory_dev);
    args.arg(&x_dev);
    args.arg(&mut out_dev);
    args.arg(&decay);
    args.arg(&n);
    unsafe {
        args.launch(LaunchConfig::for_num_elems(n as u32))
            .map_err(cuda_error)?;
    }
    let data = stream.clone_dtoh(&out_dev).map_err(cuda_error)?;
    Tensor::new(memory.rows, memory.cols, data)
}

#[cfg(feature = "cuda")]
pub fn permute_cuda(x: &Tensor, shift: isize) -> Result<Tensor> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    let ctx = CudaContext::new(0).map_err(cuda_error)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(VSA_KERNELS).map_err(cuda_error)?;
    let module = ctx.load_module(ptx).map_err(cuda_error)?;
    let function = module
        .load_function("vsa_permute_kernel")
        .map_err(cuda_error)?;
    let x_dev = stream.clone_htod(&x.data).map_err(cuda_error)?;
    let mut out_dev = x_dev.clone();
    let n = x.len() as i32;
    let shift = shift as i32;

    let mut args = stream.launch_builder(&function);
    args.arg(&x_dev);
    args.arg(&mut out_dev);
    args.arg(&n);
    args.arg(&shift);
    unsafe {
        args.launch(LaunchConfig::for_num_elems(n as u32))
            .map_err(cuda_error)?;
    }
    let data = stream.clone_dtoh(&out_dev).map_err(cuda_error)?;
    Tensor::new(x.rows, x.cols, data)
}

#[cfg(feature = "cuda")]
fn cuda_error(error: impl std::fmt::Display) -> symliquid_core::error::SymError {
    symliquid_core::error::SymError::InvalidArgument(format!("CUDA operation failed: {error}"))
}
