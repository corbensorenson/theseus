use symliquid_core::eval::format_report;
use symliquid_core::tasks::delayed_recall::{run, DelayedRecallConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run(DelayedRecallConfig::default())?;
    println!("{}", format_report(&report));
    Ok(())
}
