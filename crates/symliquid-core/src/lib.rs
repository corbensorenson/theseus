//! CPU reference implementation for SymLiquid FEP-Net.
//!
//! The crate intentionally keeps the first prototype small: tensors are simple
//! row-major `Vec<f32>` buffers, gradients are explicit where needed, and the
//! synthetic tasks are built to exercise the architecture's interfaces.

pub mod ablations;
pub mod backend;
pub mod benchmarks;
pub mod cgs;
pub mod config;
pub mod error;
pub mod eval;
pub mod loop_closure;
pub mod modules;
pub mod tasks;
pub mod tensor;
pub mod token_superposition;
pub mod train;

pub use config::{AblationConfig, SymLiquidConfig};
pub use error::{Result, SymError};
pub use modules::model::{ModelOutput, ModelState, SymLiquidFEPNet};
pub use tensor::Tensor;
