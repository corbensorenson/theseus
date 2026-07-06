#[derive(Debug, Clone, Copy)]
pub struct AblationConfig {
    pub use_kan: bool,
    pub use_liquid: bool,
    pub use_reservoir: bool,
    pub use_vsa: bool,
    pub use_fep: bool,
}

impl AblationConfig {
    pub fn full() -> Self {
        Self {
            use_kan: true,
            use_liquid: true,
            use_reservoir: true,
            use_vsa: true,
            use_fep: true,
        }
    }

    pub fn named(name: &str) -> Self {
        let mut cfg = Self::full();
        match name {
            "no_kan" => cfg.use_kan = false,
            "no_liquid" => cfg.use_liquid = false,
            "no_reservoir" => cfg.use_reservoir = false,
            "no_vsa" => cfg.use_vsa = false,
            "no_fep" => cfg.use_fep = false,
            "random_policy" => cfg.use_fep = false,
            _ => {}
        }
        cfg
    }
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self::full()
    }
}

#[derive(Debug, Clone)]
pub struct SymLiquidConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub reservoir_dim: usize,
    pub hv_dim: usize,
    pub output_dim: usize,
    pub latent_dim: usize,
    pub obs_dim: usize,
    pub action_dim: usize,
    pub kan_basis: usize,
    pub ablations: AblationConfig,
}

impl Default for SymLiquidConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            reservoir_dim: 64,
            hv_dim: 256,
            output_dim: 8,
            latent_dim: 8,
            obs_dim: 8,
            action_dim: 4,
            kan_basis: 9,
            ablations: AblationConfig::full(),
        }
    }
}
