use crate::error::Result;
use crate::modules::vsa::VSAMemory;
use crate::tensor::Tensor;

pub trait Backend {
    fn vsa_bind(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn vsa_bundle(&self, memory: &Tensor, x: &Tensor, decay: f32) -> Result<Tensor>;
    fn vsa_permute(&self, x: &Tensor, shift: isize) -> Tensor;
    fn cleanup_similarity(&self, query: &Tensor, symbol_table: &Tensor) -> Result<(usize, f32)>;
    fn expected_free_energy(
        &self,
        risk: &[f32],
        ambiguity: &[f32],
        info_gain: &[f32],
        epistemic_weight: f32,
    ) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn vsa_bind(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        VSAMemory::bind(a, b)
    }

    fn vsa_bundle(&self, memory: &Tensor, x: &Tensor, decay: f32) -> Result<Tensor> {
        VSAMemory::bundle(memory, x, decay)
    }

    fn vsa_permute(&self, x: &Tensor, shift: isize) -> Tensor {
        VSAMemory::permute(x, shift)
    }

    fn cleanup_similarity(&self, query: &Tensor, symbol_table: &Tensor) -> Result<(usize, f32)> {
        VSAMemory::cleanup(query, symbol_table)
    }

    fn expected_free_energy(
        &self,
        risk: &[f32],
        ambiguity: &[f32],
        info_gain: &[f32],
        epistemic_weight: f32,
    ) -> Result<Vec<f32>> {
        if risk.len() != ambiguity.len() || risk.len() != info_gain.len() {
            return Err(crate::error::SymError::Shape(
                "expected risk, ambiguity, and info_gain arrays with equal length".to_string(),
            ));
        }
        Ok(risk
            .iter()
            .zip(ambiguity)
            .zip(info_gain)
            .map(|((r, a), i)| r + a - epistemic_weight * i)
            .collect())
    }
}
