use symliquid_core::tasks::active_gridworld::{run, ActiveGridworldConfig};

#[test]
fn active_gridworld_runs_and_reports_cgs_terms() {
    let report = run(ActiveGridworldConfig {
        episodes: 16,
        ..ActiveGridworldConfig::default()
    })
    .unwrap();
    assert_eq!(report.task, "active_gridworld");
    assert!(report.cgs.total_cost() > 0.0);
    assert!(report.verification.checks > 0);
}
