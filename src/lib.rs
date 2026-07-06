//! Workspace-level re-export for the SymLiquid FEP-Net reference prototype.

pub use symliquid_core as core;

#[cfg(feature = "cuda")]
pub use symliquid_cuda as cuda;
