use std::path::PathBuf;

fn main() {
    let out = out_arg();
    let report = symliquid_metal::rollout_metal_proof_report(
        &symliquid_metal::RolloutMetalProofConfig::default(),
    );
    let text = serde_json::to_string_pretty(&report).expect("report serializes") + "\n";
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create report parent");
        }
        std::fs::write(&path, text.as_bytes()).expect("write report");
    }
    print!("{text}");
    let ok = report
        .get("ok")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    std::process::exit(if ok { 0 } else { 2 });
}

fn out_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--out" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}
