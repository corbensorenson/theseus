use std::collections::BTreeMap;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub available: bool,
    pub name: String,
}

impl DeviceInfo {
    pub fn cpu_emulated() -> Self {
        Self {
            available: false,
            name: "cpu-emulated-cuda-surface".to_string(),
        }
    }

    pub fn cuda_enabled() -> Self {
        Self {
            available: true,
            name: "cuda-device-0".to_string(),
        }
    }
}

pub fn runtime_profile(cuda_feature_enabled: bool) -> BTreeMap<String, String> {
    let mut profile = BTreeMap::new();
    profile.insert(
        "cuda_feature_enabled".to_string(),
        cuda_feature_enabled.to_string(),
    );
    profile.insert(
        "cuda_fallback".to_string(),
        (!cuda_feature_enabled).to_string(),
    );

    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = text.lines().find(|line| !line.trim().is_empty()) {
                let parts = line
                    .split(',')
                    .map(|part| part.trim().to_string())
                    .collect::<Vec<_>>();
                if let Some(value) = parts.first() {
                    profile.insert("gpu_name".to_string(), value.clone());
                }
                if let Some(value) = parts.get(1) {
                    profile.insert("vram_total_mib".to_string(), value.clone());
                }
                if let Some(value) = parts.get(2) {
                    profile.insert("vram_free_mib".to_string(), value.clone());
                }
                if let Some(value) = parts.get(3) {
                    profile.insert("driver_version".to_string(), value.clone());
                }
                if let Some(value) = parts.get(4) {
                    profile.insert("compute_capability".to_string(), value.clone());
                }
                if let Some(value) = parts.get(5) {
                    profile.insert("gpu_utilization_percent".to_string(), value.clone());
                }
                profile.insert("nvidia_smi_available".to_string(), "true".to_string());
            }
        }
        Ok(output) => {
            profile.insert("nvidia_smi_available".to_string(), "false".to_string());
            profile.insert(
                "nvidia_smi_error".to_string(),
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            );
        }
        Err(error) => {
            profile.insert("nvidia_smi_available".to_string(), "false".to_string());
            profile.insert("nvidia_smi_error".to_string(), error.to_string());
        }
    }

    match Command::new("nvcc").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            let release = text
                .lines()
                .find(|line| line.contains("release"))
                .unwrap_or_else(|| text.lines().last().unwrap_or(""));
            profile.insert("cuda_toolkit".to_string(), release.trim().to_string());
            profile.insert("nvcc_available".to_string(), "true".to_string());
        }
        Ok(output) => {
            profile.insert("nvcc_available".to_string(), "false".to_string());
            profile.insert(
                "nvcc_error".to_string(),
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            );
        }
        Err(error) => {
            profile.insert("nvcc_available".to_string(), "false".to_string());
            profile.insert("nvcc_error".to_string(), error.to_string());
        }
    }

    profile
}
