//! Optional CUDA backend surface.
//!
//! This crate currently exposes a correctness-first backend contract and an
//! emulated CUDA backend that delegates to the CPU reference operations. The
//! CUDA C kernel sources live in `kernels/` as the intended acceleration targets;
//! driver integration is deliberately isolated from the CPU crate.

use symliquid_core::error::Result;
use symliquid_core::modules::vsa::VSAMemory;
use symliquid_core::tensor::Tensor;

pub mod device;
pub mod fep_cuda;
pub mod kan_cuda;
pub mod kernels;
pub mod liquid_cuda;
pub mod readout_cuda;
pub mod reservoir_cuda;
pub mod rollout_cuda;
pub mod vsa_cuda;

#[derive(Debug, Clone, Default)]
pub struct CudaBackend {
    available: bool,
}

impl CudaBackend {
    pub fn new() -> Self {
        Self {
            available: cfg!(feature = "cuda"),
        }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn device_info(&self) -> device::DeviceInfo {
        if self.available {
            device::DeviceInfo::cuda_enabled()
        } else {
            device::DeviceInfo::cpu_emulated()
        }
    }

    pub fn vsa_bind(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            vsa_cuda::bind_cuda(a, b)
        }
        #[cfg(not(feature = "cuda"))]
        VSAMemory::bind(a, b)
    }

    pub fn vsa_bundle(&self, memory: &Tensor, x: &Tensor, decay: f32) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            vsa_cuda::bundle_cuda(memory, x, decay)
        }
        #[cfg(not(feature = "cuda"))]
        VSAMemory::bundle(memory, x, decay)
    }

    pub fn vsa_permute(&self, x: &Tensor, shift: isize) -> Tensor {
        #[cfg(feature = "cuda")]
        {
            vsa_cuda::permute_cuda(x, shift).unwrap_or_else(|_| VSAMemory::permute(x, shift))
        }
        #[cfg(not(feature = "cuda"))]
        VSAMemory::permute(x, shift)
    }

    pub fn cleanup_similarity(&self, query: &Tensor, table: &Tensor) -> Result<(usize, f32)> {
        VSAMemory::cleanup(query, table)
    }
}

impl symliquid_core::backend::Backend for CudaBackend {
    fn vsa_bind(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.vsa_bind(a, b)
    }

    fn vsa_bundle(&self, memory: &Tensor, x: &Tensor, decay: f32) -> Result<Tensor> {
        self.vsa_bundle(memory, x, decay)
    }

    fn vsa_permute(&self, x: &Tensor, shift: isize) -> Tensor {
        self.vsa_permute(x, shift)
    }

    fn cleanup_similarity(&self, query: &Tensor, symbol_table: &Tensor) -> Result<(usize, f32)> {
        self.cleanup_similarity(query, symbol_table)
    }

    fn expected_free_energy(
        &self,
        risk: &[f32],
        ambiguity: &[f32],
        info_gain: &[f32],
        epistemic_weight: f32,
    ) -> Result<Vec<f32>> {
        <symliquid_core::backend::CpuBackend as symliquid_core::backend::Backend>::expected_free_energy(
            &symliquid_core::backend::CpuBackend,
            risk,
            ambiguity,
            info_gain,
            epistemic_weight,
        )
    }
}

#[cfg(feature = "cuda")]
pub const CUDA_FEATURE_ENABLED: bool = true;

#[cfg(not(feature = "cuda"))]
pub const CUDA_FEATURE_ENABLED: bool = false;
