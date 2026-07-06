use symliquid_core::eval::format_report;
use symliquid_core::tasks::active_classification::{run, ActiveClassificationConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run(ActiveClassificationConfig::default())?;
    println!("{}", format_report(&report));
    Ok(())
}
