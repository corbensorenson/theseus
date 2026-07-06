use symliquid_fep::core::tasks::active_classification::{
    run as run_active, ActiveClassificationConfig,
};
use symliquid_fep::core::tasks::active_gridworld::{run as run_gridworld, ActiveGridworldConfig};
use symliquid_fep::core::tasks::delayed_recall::{run as run_delayed, DelayedRecallConfig};
use symliquid_fep::core::tasks::role_filler::{run as run_role, RoleFillerConfig};

#[test]
fn toy_tasks_run_smoke_test() {
    let role = run_role(RoleFillerConfig {
        steps: 12,
        ..RoleFillerConfig::default()
    })
    .unwrap();
    assert_eq!(role.task, "role_filler");

    let delayed = run_delayed(DelayedRecallConfig {
        steps: 12,
        ..DelayedRecallConfig::default()
    })
    .unwrap();
    assert_eq!(delayed.task, "delayed_recall");

    let active = run_active(ActiveClassificationConfig {
        episodes: 12,
        ..ActiveClassificationConfig::default()
    })
    .unwrap();
    assert_eq!(active.task, "active_classification");

    let gridworld = run_gridworld(ActiveGridworldConfig {
        episodes: 12,
        ..ActiveGridworldConfig::default()
    })
    .unwrap();
    assert_eq!(gridworld.task, "active_gridworld");
}
