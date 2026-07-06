use symliquid_core::eval::format_report;
use symliquid_core::tasks::active_gridworld::{run, ActiveGridworldConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run(ActiveGridworldConfig::default())?;
    println!("{}", format_report(&report));
    Ok(())
}
