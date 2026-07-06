use crate::config::AblationConfig;
use crate::error::Result;
use crate::eval::TaskReport;
use crate::tasks::active_classification::{run as run_active, ActiveClassificationConfig};
use crate::tasks::active_gridworld::{run as run_gridworld, ActiveGridworldConfig};
use crate::tasks::delayed_recall::{run as run_delayed, DelayedRecallConfig};
use crate::tasks::role_filler::{run as run_role, RoleFillerConfig};

pub fn run_ablation_suite(task: &str, steps: usize, seed: u64) -> Result<Vec<TaskReport>> {
    let variants = variants_for(task);
    let mut reports = Vec::with_capacity(variants.len());
    for variant in variants {
        let ablations = AblationConfig::named(variant);
        let report = match task {
            "role_filler" | "role-filler" => run_role(RoleFillerConfig {
                steps,
                seed,
                ablations,
                variant: variant.to_string(),
                ..RoleFillerConfig::default()
            })?,
            "delayed_recall" | "delayed-recall" => run_delayed(DelayedRecallConfig {
                steps,
                seed,
                ablations,
                variant: variant.to_string(),
                ..DelayedRecallConfig::default()
            })?,
            "active_classification" | "active-classification" => {
                run_active(ActiveClassificationConfig {
                    episodes: steps,
                    seed,
                    ablations,
                    variant: variant.to_string(),
                    ..ActiveClassificationConfig::default()
                })?
            }
            "active_gridworld" | "active-gridworld" | "gridworld" => {
                run_gridworld(ActiveGridworldConfig {
                    episodes: steps,
                    seed,
                    ablations,
                    variant: variant.to_string(),
                    ..ActiveGridworldConfig::default()
                })?
            }
            other => {
                return Err(crate::error::SymError::InvalidArgument(format!(
                    "unknown ablation task {other}"
                )))
            }
        };
        reports.push(report);
    }
    Ok(reports)
}

fn variants_for(task: &str) -> Vec<&'static str> {
    match task {
        "role_filler" | "role-filler" => vec!["full", "no_vsa"],
        "delayed_recall" | "delayed-recall" => vec!["full", "no_reservoir", "no_vsa"],
        "active_classification" | "active-classification" => {
            vec!["full", "no_fep", "random_policy"]
        }
        "active_gridworld" | "active-gridworld" | "gridworld" => {
            vec!["full", "no_fep", "random_policy"]
        }
        _ => vec!["full"],
    }
}
