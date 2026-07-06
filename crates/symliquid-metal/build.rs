use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/rollout_state_update.metal");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let air = out_dir.join("rollout_state_update.air");
    let metallib = out_dir.join("rollout_state_update.metallib");

    let metal_status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "kernels/rollout_state_update.metal",
            "-o",
        ])
        .arg(&air)
        .status()
        .expect("xcrun metal should run on macOS");
    assert!(metal_status.success(), "xcrun metal failed");

    let metallib_status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&metallib)
        .status()
        .expect("xcrun metallib should run on macOS");
    assert!(metallib_status.success(), "xcrun metallib failed");
}
