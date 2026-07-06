use symliquid_core::eval::format_report;
use symliquid_core::tasks::role_filler::{run, RoleFillerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run(RoleFillerConfig::default())?;
    println!("{}", format_report(&report));
    Ok(())
}
