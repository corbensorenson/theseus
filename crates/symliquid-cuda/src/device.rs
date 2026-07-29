use std::collections::BTreeMap;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub available: bool,
    pub name: String,
    pub compiled: bool,
    pub driver_visible: bool,
    pub device_visible: bool,
    pub initialized: bool,
    pub fallback: bool,
    pub state: String,
}

impl DeviceInfo {
    pub fn cpu_emulated() -> Self {
        Self {
            available: false,
            name: "cpu-emulated-cuda-surface".to_string(),
            compiled: false,
            driver_visible: false,
            device_visible: false,
            initialized: false,
            fallback: true,
            state: "not_compiled".to_string(),
        }
    }

    pub fn cuda_enabled() -> Self {
        let probe = probe_runtime(true);
        Self {
            available: probe.initialized,
            name: probe
                .gpu_name
                .clone()
                .unwrap_or_else(|| "cuda-device-0-unavailable".to_string()),
            compiled: probe.compiled,
            driver_visible: probe.driver_visible,
            device_visible: probe.device_visible,
            initialized: probe.initialized,
            fallback: !probe.initialized,
            state: probe.state,
        }
    }
}

pub fn runtime_profile(cuda_feature_enabled: bool) -> BTreeMap<String, String> {
    let mut profile = BTreeMap::new();
    let probe = probe_runtime(cuda_feature_enabled);
    profile.insert("cuda_compiled".to_string(), probe.compiled.to_string());
    profile.insert(
        "cuda_driver_visible".to_string(),
        probe.driver_visible.to_string(),
    );
    profile.insert(
        "cuda_device_visible".to_string(),
        probe.device_visible.to_string(),
    );
    profile.insert(
        "cuda_initialized".to_string(),
        probe.initialized.to_string(),
    );
    profile.insert("cuda_ready".to_string(), probe.initialized.to_string());
    profile.insert(
        "cuda_fallback".to_string(),
        (!probe.initialized).to_string(),
    );
    profile.insert("cuda_state".to_string(), probe.state.clone());
    profile.insert(
        "cuda_initialization_error".to_string(),
        probe.initialization_error.clone().unwrap_or_default(),
    );
    profile.insert(
        "cuda_feature_enabled".to_string(),
        cuda_feature_enabled.to_string(),
    );
    if let Some(name) = &probe.gpu_name {
        profile.insert("gpu_name".to_string(), name.clone());
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeProbe {
    compiled: bool,
    driver_visible: bool,
    device_visible: bool,
    initialized: bool,
    gpu_name: Option<String>,
    initialization_error: Option<String>,
    state: String,
}

fn probe_runtime(cuda_feature_enabled: bool) -> RuntimeProbe {
    if !cuda_feature_enabled {
        return RuntimeProbe {
            compiled: false,
            driver_visible: false,
            device_visible: false,
            initialized: false,
            gpu_name: None,
            initialization_error: None,
            state: "not_compiled".to_string(),
        };
    }
    let smi = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader,nounits"])
        .output();
    let (driver_visible, gpu_name) = match smi {
        Ok(output) if output.status.success() => {
            let name = String::from_utf8_lossy(&output.stdout)
                .lines()
                .find(|line| !line.trim().is_empty())
                .map(|line| line.trim().to_string());
            (true, name)
        }
        _ => (false, None),
    };
    let device_visible = gpu_name.is_some();
    let initialization = initialize_cuda_context();
    let (initialized, initialization_error) = match initialization {
        Ok(()) => (true, None),
        Err(error) => (false, Some(error)),
    };
    let state = if initialized {
        "ready"
    } else if !driver_visible {
        "driver_unavailable"
    } else if !device_visible {
        "no_device"
    } else {
        "initialization_failed"
    };
    RuntimeProbe {
        compiled: true,
        driver_visible,
        device_visible,
        initialized,
        gpu_name,
        initialization_error,
        state: state.to_string(),
    }
}

#[cfg(feature = "cuda")]
fn initialize_cuda_context() -> std::result::Result<(), String> {
    cudarc::driver::CudaContext::new(0)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[cfg(not(feature = "cuda"))]
fn initialize_cuda_context() -> std::result::Result<(), String> {
    Err("symliquid-cuda was not compiled with the cuda feature".to_string())
}

#[cfg(test)]
mod tests {
    use super::{runtime_profile, DeviceInfo};

    #[test]
    fn uncompiled_runtime_reports_each_state_without_claiming_availability() {
        let info = DeviceInfo::cpu_emulated();
        assert!(!info.available);
        assert!(!info.compiled);
        assert!(!info.driver_visible);
        assert!(!info.device_visible);
        assert!(!info.initialized);
        assert!(info.fallback);
        assert_eq!(info.state, "not_compiled");

        let profile = runtime_profile(false);
        assert_eq!(profile["cuda_compiled"], "false");
        assert_eq!(profile["cuda_initialized"], "false");
        assert_eq!(profile["cuda_fallback"], "true");
        assert_eq!(profile["cuda_state"], "not_compiled");
    }
}
