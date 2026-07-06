// Per-task parser/AST variant cache for candidate fanout hot paths.
// Variant generation can run parser completion and contract checks, so repeated
// bodies should pay that cost once per task.

use super::*;

#[derive(Default)]
pub(in crate::code_lm_closure::candidate_fanout) struct CandidateVariantCache {
    variants: HashMap<String, Vec<String>>,
    calls: usize,
    hits: usize,
}

impl CandidateVariantCache {
    pub(in crate::code_lm_closure::candidate_fanout) fn variants(
        &mut self,
        task: &CodeTask,
        body: &str,
    ) -> Vec<String> {
        self.calls = self.calls.saturating_add(1);
        let key = variant_cache_key(task, body);
        if let Some(cached) = self.variants.get(&key) {
            self.hits = self.hits.saturating_add(1);
            return cached.clone();
        }
        let variants = state_sequence_body_variants(task, body);
        self.variants.insert(key, variants.clone());
        variants
    }

    pub(in crate::code_lm_closure::candidate_fanout) fn entries(&self) -> usize {
        self.variants.len()
    }

    pub(in crate::code_lm_closure::candidate_fanout) fn hits(&self) -> usize {
        self.hits
    }

    pub(in crate::code_lm_closure::candidate_fanout) fn misses(&self) -> usize {
        self.calls.saturating_sub(self.hits)
    }
}

fn variant_cache_key(task: &CodeTask, body: &str) -> String {
    let template_free = template_free_student_candidates_enabled();
    format!(
        "{}\0{}\0{}\0{}",
        task.task_id,
        task.category,
        template_free,
        body.trim()
    )
}
