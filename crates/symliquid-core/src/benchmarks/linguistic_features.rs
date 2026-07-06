use super::{add_feature, add_text_features, tokenize};

mod lexicon;
use lexicon::*;

pub(super) fn sentence_features(sentence: &str, hv_dim: usize, stateful: bool) -> Vec<f32> {
    let mut features = vec![0.0; hv_dim];
    add_feature(&mut features, "bias", 0.25);
    add_text_features(&mut features, sentence, "sent");
    add_sentence_shape_features(&mut features, sentence);
    add_sentence_agreement_features(&mut features, sentence);
    add_blimp_linguistic_features(&mut features, sentence);
    if stateful {
        add_recurrent_number_state_features(&mut features, sentence);
        add_recurrent_grammar_slot_features(&mut features, sentence);
    }
    normalize_in_place(&mut features);
    features
}

pub(super) fn normalize_in_place(features: &mut [f32]) {
    let norm = features
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0);
    for value in features {
        *value /= norm;
    }
}

fn add_sentence_shape_features(features: &mut [f32], sentence: &str) {
    let tokens = tokenize(sentence);
    add_feature(features, &format!("len_bucket:{}", tokens.len() / 4), 0.3);
    for token in tokens {
        if token.len() >= 3 {
            let prefix = &token[..token.len().min(3)];
            let suffix = &token[token.len().saturating_sub(3)..];
            add_feature(features, &format!("prefix:{prefix}"), 0.15);
            add_feature(features, &format!("suffix:{suffix}"), 0.2);
        }
        if token.ends_with('s') {
            add_feature(features, "shape:ends_s", 0.15);
        }
    }
}

fn add_sentence_agreement_features(features: &mut [f32], sentence: &str) {
    let tokens = tokenize(sentence);
    for window in tokens.windows(2) {
        let left = &window[0];
        let right = &window[1];
        if matches!(
            left.as_str(),
            "this" | "that" | "a" | "an" | "he" | "she" | "it"
        ) && matches!(right.as_str(), "is" | "was" | "has" | "does")
        {
            add_feature(features, "agreement:singular_local", 0.8);
        }
        if matches!(left.as_str(), "these" | "those" | "they" | "we" | "you")
            && matches!(right.as_str(), "are" | "were" | "have" | "do")
        {
            add_feature(features, "agreement:plural_local", 0.8);
        }
        if left == "can" && !right.ends_with('s') {
            add_feature(features, "modal:base_form_after_can", 0.8);
        }
        if left == "can" && right.ends_with('s') {
            add_feature(features, "modal:bad_s_after_can", -0.8);
        }
    }
    for window in tokens.windows(4) {
        if window[0] == "this" && window.iter().any(|token| token == "are") {
            add_feature(features, "agreement:this_are_window", -0.8);
        }
        if window[0] == "these" && window.iter().any(|token| token == "is") {
            add_feature(features, "agreement:these_is_window", -0.8);
        }
    }
}

fn add_blimp_linguistic_features(features: &mut [f32], sentence: &str) {
    let tokens = tokenize(sentence);
    add_determiner_noun_state_features(features, &tokens);
    add_wh_gap_state_features(features, &tokens);
    add_tough_raising_state_features(features, &tokens);
    add_reflexive_binding_state_features(features, &tokens);
    add_ellipsis_state_features(features, &tokens);
    add_island_state_features(features, &tokens);
    add_subject_verb_binding_features(features, &tokens);
    add_passive_argument_state_features(features, &tokens);
    add_existential_there_state_features(features, &tokens);
    add_expletive_it_state_features(features, &tokens);
    add_animate_subject_state_features(features, &tokens);
    add_argument_structure_state_features(features, &tokens);
    add_left_branch_state_features(features, &tokens);

    let governance_score = linguistic_governance_score(&tokens);
    add_feature(
        features,
        "grammar:linguistic_governance_score",
        governance_score,
    );
    add_feature(
        features,
        "grammar:linguistic_governance_positive",
        governance_score.max(0.0),
    );
    add_feature(
        features,
        "grammar:linguistic_governance_negative",
        (-governance_score).max(0.0),
    );
}

fn add_determiner_noun_state_features(features: &mut [f32], tokens: &[String]) {
    for (idx, token) in tokens.iter().enumerate() {
        let Some(det_number) = determiner_number(token) else {
            continue;
        };
        for (noun_idx, noun) in tokens
            .iter()
            .enumerate()
            .take(tokens.len().min(idx + 5))
            .skip(idx + 1)
        {
            if is_functionish(noun) {
                break;
            }
            let noun_number = nominal_number_cue(noun);
            if noun_number == NumberState::Unknown {
                continue;
            }
            let distance = noun_idx - idx;
            add_feature(
                features,
                &format!(
                    "det_noun:{}:{}:dist{}",
                    det_number.as_str(),
                    noun_number.as_str(),
                    distance
                ),
                0.9,
            );
            if det_number == noun_number {
                add_feature(features, "det_noun:match", 1.0);
            } else {
                add_feature(features, "det_noun:mismatch", -1.0);
            }
            break;
        }
    }
}

fn add_wh_gap_state_features(features: &mut [f32], tokens: &[String]) {
    let has_wh = tokens
        .iter()
        .any(|token| matches!(token.as_str(), "what" | "who" | "whom" | "which"));
    let has_that = tokens.iter().any(|token| token == "that");
    let gap_like = ends_with_gap_like(tokens);
    if has_wh {
        add_feature(features, "wh_gap:has_wh_filler", 0.8);
    }
    if has_that && !has_wh {
        add_feature(features, "wh_gap:that_without_wh", 0.4);
    }
    if gap_like {
        add_feature(features, "wh_gap:gap_like_tail", 0.7);
    }
    match (has_wh, gap_like) {
        (true, true) => add_feature(features, "wh_gap:filler_gap_match", 1.2),
        (false, true) if has_that => add_feature(features, "wh_gap:that_gap_mismatch", -1.2),
        _ => {}
    }
    for window in tokens.windows(2) {
        if matches!(
            window[0].as_str(),
            "know" | "knows" | "remember" | "remembers" | "investigate" | "investigates"
        ) && matches!(window[1].as_str(), "what" | "who" | "that")
        {
            add_feature(
                features,
                &format!("wh_gap:embedding_verb:{}:{}", window[0], window[1]),
                0.8,
            );
        }
    }
}

fn add_tough_raising_state_features(features: &mut [f32], tokens: &[String]) {
    let gap_like = ends_with_gap_like(tokens);
    for window in tokens.windows(2) {
        if window[1] != "to" {
            continue;
        }
        if is_tough_adjective(&window[0]) {
            add_feature(features, "tough_raising:tough_adj_to", 0.9);
            if gap_like {
                add_feature(features, "tough_raising:tough_gap_match", 1.1);
            }
        }
        if is_raising_adjective(&window[0]) {
            add_feature(features, "tough_raising:raising_adj_to", 0.9);
            if gap_like {
                add_feature(features, "tough_raising:raising_gap_mismatch", -1.1);
            }
        }
    }
}

fn add_reflexive_binding_state_features(features: &mut [f32], tokens: &[String]) {
    for (idx, token) in tokens.iter().enumerate() {
        let Some(reflexive) = reflexive_profile(token) else {
            continue;
        };
        add_feature(
            features,
            &format!(
                "reflexive:{}:{}",
                reflexive.number.as_str(),
                reflexive.gender.as_str()
            ),
            0.8,
        );
        add_reflexive_lexical_binding_features(features, tokens, idx, token);

        if let Some(head) = relative_head_profile_before(tokens, idx) {
            add_profile_match_features(features, "reflexive:relative_head", head, reflexive, 2.0);
            add_feature(
                features,
                "reflexive:relative_head_score",
                profile_match_score(head, reflexive),
            );
        } else {
            if let Some(nearest) = nearest_entity_profile_before(tokens, idx) {
                add_profile_match_features(features, "reflexive:nearest", nearest, reflexive, 1.0);
            }
            if let Some(local_subject) = local_subject_profile_before(tokens, idx) {
                add_profile_match_features(
                    features,
                    "reflexive:local_subject",
                    local_subject,
                    reflexive,
                    1.15,
                );
            }
        }
        if let Some(reconstructed) = reconstruction_subject_profile_after(tokens, idx) {
            add_profile_match_features(
                features,
                "reflexive:reconstruction_subject",
                reconstructed,
                reflexive,
                1.9,
            );
            add_feature(
                features,
                "reflexive:reconstruction_score",
                profile_match_score(reconstructed, reflexive),
            );
        }
    }
}

fn add_reflexive_lexical_binding_features(
    features: &mut [f32],
    tokens: &[String],
    reflexive_idx: usize,
    reflexive_token: &str,
) {
    if let Some(nearest) = nearest_content_token_before(tokens, reflexive_idx) {
        add_feature(
            features,
            &format!("reflexive:nearest_token:{nearest}:{reflexive_token}"),
            1.2,
        );
    }
    if let Some(subject) = embedded_subject_token_before(tokens, reflexive_idx) {
        add_feature(
            features,
            &format!("reflexive:embedded_subject_token:{subject}:{reflexive_token}"),
            1.5,
        );
    }
    if let Some(matrix_subject) = matrix_subject_token_before_bridge(tokens, reflexive_idx) {
        add_feature(
            features,
            &format!("reflexive:matrix_subject_token:{matrix_subject}:{reflexive_token}"),
            0.55,
        );
    }
}

fn add_ellipsis_state_features(features: &mut [f32], tokens: &[String]) {
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "many" | "few" | "several"))
        || tokens
            .iter()
            .any(|token| number_word_value(token).is_some())
    {
        add_feature(features, "ellipsis:has_quantifier", 0.35);
    }

    if let Some(last) = tokens.last().map(String::as_str) {
        if is_common_modifier(last) {
            add_feature(features, "ellipsis:stranded_modifier_tail", -1.0);
        }
        if number_word_value(last).is_some() || matches!(last, "many" | "few" | "several") {
            add_feature(features, "ellipsis:bare_quantifier_tail", 0.8);
        }
    }

    for window in tokens.windows(3) {
        if matches!(window[0].as_str(), "as" | "so")
            && window[1] == "many"
            && is_common_modifier(&window[2])
        {
            add_feature(features, "ellipsis:comparative_stranded_modifier", -0.9);
        }
        if number_word_value(&window[0]).is_some() && is_common_modifier(&window[1]) {
            add_feature(features, "ellipsis:quantifier_modifier_noun_prefix", 0.5);
        }
    }
}

fn add_island_state_features(features: &mut [f32], tokens: &[String]) {
    let has_wh = tokens
        .iter()
        .any(|token| matches!(token.as_str(), "what" | "who" | "whom" | "which"));
    if !has_wh {
        return;
    }

    for (idx, token) in tokens.iter().enumerate() {
        if !matches!(
            token.as_str(),
            "after" | "before" | "while" | "without" | "because"
        ) {
            continue;
        }
        add_feature(features, &format!("island:adjunct_marker:{token}"), 0.45);
        let tail = &tokens[idx + 1..];
        if tail.is_empty() {
            continue;
        }
        let tail_has_object = tail.iter().skip(1).any(|tok| {
            matches!(tok.as_str(), "the" | "a" | "an" | "some" | "this" | "that")
                || is_entityish_token(tok)
        });
        if tail.last().is_some_and(|last| is_gap_tail_token(last)) && !tail_has_object {
            add_feature(features, "island:possible_gap_inside_adjunct", -1.25);
        }
        if tail_has_object {
            add_feature(features, "island:object_inside_adjunct", 0.85);
        }
    }

    let relative_marker_idx =
        tokens.iter().enumerate().skip(1).find_map(|(idx, token)| {
            matches!(token.as_str(), "who" | "that" | "which").then_some(idx)
        });
    let has_complex_np =
        tokens.windows(3).any(|window| window[0] == "of") || relative_marker_idx.is_some();
    if has_complex_np {
        add_feature(features, "island:complex_np_context", 0.55);
        if ends_with_gap_like(tokens) {
            add_feature(features, "island:complex_np_gap_tail", -0.65);
        }
        if let Some(marker_idx) = relative_marker_idx {
            let relative_clause_near_tail = tokens.len().saturating_sub(marker_idx) <= 5;
            if relative_clause_near_tail && ends_with_gap_like(tokens) {
                add_feature(features, "island:relative_clause_gap_tail", -1.3);
            }
            if !relative_clause_near_tail
                && tokens[marker_idx + 1..]
                    .iter()
                    .any(|token| is_entityish_token(token))
            {
                add_feature(features, "island:complex_np_then_matrix_object", 0.75);
            }
        }
    }
}

fn add_subject_verb_binding_features(features: &mut [f32], tokens: &[String]) {
    for idx in 0..tokens.len() {
        let Some(verb_number) = finite_verb_number(tokens, idx) else {
            continue;
        };
        if let Some(subject_number) = subject_head_number_before(tokens, idx) {
            add_feature(
                features,
                &format!(
                    "sva:head:{}:verb:{}",
                    subject_number.as_str(),
                    verb_number.as_str()
                ),
                0.7,
            );
            if subject_number == verb_number {
                add_feature(features, "sva:head_match", 1.1);
            } else {
                add_feature(features, "sva:head_mismatch", -1.1);
            }
        }
        if let Some(nearest_number) = nearest_nominal_number_before(tokens, idx) {
            add_feature(
                features,
                &format!(
                    "sva:nearest:{}:verb:{}",
                    nearest_number.as_str(),
                    verb_number.as_str()
                ),
                0.35,
            );
        }
    }
}

fn add_passive_argument_state_features(features: &mut [f32], tokens: &[String]) {
    for aux_idx in 0..tokens.len() {
        if !is_be_auxiliary(&tokens[aux_idx]) {
            continue;
        }
        let Some(participle_idx) = next_content_index(tokens, aux_idx + 1, 4) else {
            continue;
        };
        let participle = tokens[participle_idx].as_str();
        if passive_compatible_participle(participle) {
            add_feature(features, "passive:compatible_participle", 0.9);
            add_feature(
                features,
                &format!("passive:compatible_participle:{participle}"),
                0.7,
            );
        }
        if passive_incompatible_participle(participle) {
            add_feature(features, "passive:incompatible_participle", -1.1);
            add_feature(
                features,
                &format!("passive:incompatible_participle:{participle}"),
                -0.9,
            );
        }
        let Some(by_idx) = tokens[participle_idx + 1..]
            .iter()
            .position(|token| token == "by")
            .map(|offset| participle_idx + 1 + offset)
        else {
            continue;
        };
        if let Some(agent_idx) = next_content_index(tokens, by_idx + 1, 4) {
            match animacy_cue(&tokens[agent_idx]) {
                AnimacyState::Animate => {
                    add_feature(features, "passive:animate_by_agent", 0.85);
                }
                AnimacyState::Inanimate => {
                    add_feature(features, "passive:inanimate_by_agent", -0.85);
                }
                AnimacyState::Unknown => {}
            }
        }
    }
}

fn add_existential_there_state_features(features: &mut [f32], tokens: &[String]) {
    for (there_idx, token) in tokens.iter().enumerate() {
        if token != "there" {
            continue;
        }
        add_feature(features, "existential_there:present", 0.35);
        let Some(verb_idx) = governing_verb_before(tokens, there_idx) else {
            continue;
        };
        let verb = tokens[verb_idx].as_str();
        if existential_there_compatible_verb(verb) {
            add_feature(features, "existential_there:compatible_governor", 1.0);
            add_feature(
                features,
                &format!("existential_there:compatible_governor:{verb}"),
                0.8,
            );
        }
        if existential_there_incompatible_verb(verb) {
            add_feature(features, "existential_there:incompatible_governor", -1.0);
            add_feature(
                features,
                &format!("existential_there:incompatible_governor:{verb}"),
                -0.8,
            );
        }
        if tokens
            .get(there_idx + 1)
            .is_some_and(|next| matches!(next.as_str(), "to" | "be"))
        {
            add_feature(features, "existential_there:to_be_frame", 0.45);
        }
    }
}

fn add_expletive_it_state_features(features: &mut [f32], tokens: &[String]) {
    for it_idx in 0..tokens.len().saturating_sub(2) {
        if tokens[it_idx] != "it" || tokens.get(it_idx + 1).is_none_or(|token| token != "to") {
            continue;
        }
        add_feature(features, "expletive_it:it_to_frame", 0.35);
        let Some(verb_idx) = governing_verb_before(tokens, it_idx) else {
            continue;
        };
        let verb = tokens[verb_idx].as_str();
        if expletive_it_compatible_verb(verb) {
            add_feature(features, "expletive_it:compatible_governor", 1.0);
            add_feature(
                features,
                &format!("expletive_it:compatible_governor:{verb}"),
                0.8,
            );
        }
        if expletive_it_incompatible_verb(verb) {
            add_feature(features, "expletive_it:incompatible_governor", -1.0);
            add_feature(
                features,
                &format!("expletive_it:incompatible_governor:{verb}"),
                -0.8,
            );
        }
    }
}

fn add_animate_subject_state_features(features: &mut [f32], tokens: &[String]) {
    let Some(subject_idx) = first_subject_index(tokens) else {
        return;
    };
    let subject_animacy = animacy_cue(&tokens[subject_idx]);
    if subject_animacy == AnimacyState::Unknown {
        return;
    }
    if let Some(verb_idx) = first_predicate_index_after(tokens, subject_idx) {
        let verb = tokens[verb_idx].as_str();
        if animate_subject_required_verb(verb) {
            match subject_animacy {
                AnimacyState::Animate => {
                    add_feature(features, "animacy:required_subject_match", 1.0)
                }
                AnimacyState::Inanimate => {
                    add_feature(features, "animacy:required_subject_mismatch", -1.0)
                }
                AnimacyState::Unknown => {}
            }
            add_feature(
                features,
                &format!("animacy:required_subject_verb:{verb}"),
                0.5,
            );
        }
    }
}

fn add_argument_structure_state_features(features: &mut [f32], tokens: &[String]) {
    for (idx, token) in tokens.iter().enumerate() {
        if transitive_verb(token) {
            add_feature(features, &format!("argument:transitive_verb:{token}"), 0.35);
            if has_direct_object_after(tokens, idx) {
                add_feature(features, "argument:transitive_object_present", 0.75);
            } else if clause_boundary_after(tokens, idx) {
                add_feature(features, "argument:transitive_object_missing", -0.9);
            }
        }
        if intransitive_verb(token) {
            add_feature(
                features,
                &format!("argument:intransitive_verb:{token}"),
                0.35,
            );
            if has_direct_object_after(tokens, idx) {
                add_feature(features, "argument:intransitive_with_object", -0.85);
            } else {
                add_feature(features, "argument:intransitive_without_object", 0.45);
            }
        }
    }
}

fn add_left_branch_state_features(features: &mut [f32], tokens: &[String]) {
    for (idx, token) in tokens.iter().enumerate() {
        if !matches!(token.as_str(), "what" | "which" | "whose") {
            continue;
        }
        let wh_has_local_noun = tokens
            .get(idx + 1)
            .is_some_and(|next| nominal_number_cue(next) != NumberState::Unknown);
        if wh_has_local_noun {
            add_feature(features, "left_branch:wh_local_noun", 0.8);
        }
        if idx > 0 && wh_has_local_noun {
            add_feature(features, "left_branch:echo_wh_np", 1.0);
        }
        if idx == 0
            && tokens
                .get(1)
                .is_some_and(|next| is_auxiliary_or_modal(next))
            && !wh_has_local_noun
            && tail_contains_bare_nominal(tokens, 2)
        {
            add_feature(features, "left_branch:fronted_bare_wh_np_gap", -1.2);
        }
    }
    let adjunct_score = adjunct_island_score(tokens);
    if adjunct_score != 0.0 {
        add_feature(features, "island:adjunct_pairwise_score", adjunct_score);
    }
    let animacy_score = animate_subject_score(tokens);
    if animacy_score != 0.0 {
        add_feature(features, "animacy:subject_governance_score", animacy_score);
    }
}

pub(super) fn add_pairwise_contrast_features(
    features: &mut [f32],
    sentence_a: &str,
    sentence_b: &str,
) {
    let tokens_a = tokenize(sentence_a);
    let tokens_b = tokenize(sentence_b);
    let governance_delta =
        linguistic_governance_score(&tokens_a) - linguistic_governance_score(&tokens_b);
    add_feature(
        features,
        "contrast:linguistic_governance_delta",
        governance_delta,
    );
    if governance_delta > 0.0 {
        add_feature(features, "contrast:linguistic_governance_prefers_a", 1.0);
    } else if governance_delta < 0.0 {
        add_feature(features, "contrast:linguistic_governance_prefers_b", 1.0);
    }

    let max_len = tokens_a.len().max(tokens_b.len());
    for idx in 0..max_len {
        let token_a = tokens_a.get(idx).map(String::as_str).unwrap_or("<missing>");
        let token_b = tokens_b.get(idx).map(String::as_str).unwrap_or("<missing>");
        if token_a == token_b {
            continue;
        }

        let left = idx
            .checked_sub(1)
            .and_then(|left_idx| tokens_a.get(left_idx))
            .map(String::as_str)
            .unwrap_or("<start>");
        let right = tokens_a.get(idx + 1).map(String::as_str).unwrap_or("<end>");
        let cue = nearest_number_cue_before(&tokens_a, idx);

        add_feature(features, "contrast:changed_token", 0.6);
        add_feature(
            features,
            &format!("contrast:ordered:{token_a}>{token_b}"),
            1.0,
        );
        add_feature(
            features,
            &format!("contrast:left:{left}:{token_a}>{token_b}"),
            0.8,
        );
        add_feature(
            features,
            &format!("contrast:right:{token_a}>{token_b}:{right}"),
            0.8,
        );

        if let Some(cue_number) = cue {
            add_feature(
                features,
                &format!("contrast:cue:{}:{token_a}>{token_b}", cue_number.as_str()),
                1.2,
            );
            add_feature(
                features,
                "contrast:aux_cue_match_delta",
                number_match_delta(
                    auxiliary_number(token_a),
                    auxiliary_number(token_b),
                    cue_number,
                ),
            );
        }

        let right_number = nominal_number_cue(right);
        add_feature(
            features,
            "contrast:determiner_right_match_delta",
            number_match_delta(
                determiner_number(token_a),
                determiner_number(token_b),
                right_number,
            ),
        );

        if let (Some(aux_a), Some(aux_b)) = (auxiliary_number(token_a), auxiliary_number(token_b)) {
            add_feature(
                features,
                &format!("contrast:aux_pair:{}>{}", aux_a.as_str(), aux_b.as_str()),
                0.7,
            );
        }
        if let (Some(det_a), Some(det_b)) = (determiner_number(token_a), determiner_number(token_b))
        {
            add_feature(
                features,
                &format!("contrast:det_pair:{}>{}", det_a.as_str(), det_b.as_str()),
                0.7,
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberState {
    Unknown,
    Singular,
    Plural,
}

impl NumberState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Singular => "singular",
            Self::Plural => "plural",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenderState {
    Unknown,
    Male,
    Female,
    Neutral,
}

impl GenderState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Male => "male",
            Self::Female => "female",
            Self::Neutral => "neutral",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnimacyState {
    Unknown,
    Animate,
    Inanimate,
}

impl AnimacyState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Animate => "animate",
            Self::Inanimate => "inanimate",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EntityProfile {
    number: NumberState,
    gender: GenderState,
}

fn add_recurrent_number_state_features(features: &mut [f32], sentence: &str) {
    let tokens = tokenize(sentence);
    let mut state = NumberState::Unknown;
    let mut distance = 8usize;
    let mut saw_and_since_cue = false;

    for (idx, token) in tokens.iter().enumerate() {
        if token == "and" && state != NumberState::Unknown {
            saw_and_since_cue = true;
            add_feature(features, "number_state:coordination_seen", 0.35);
        }

        if let Some(cue) = number_cue(token) {
            state = if saw_and_since_cue {
                NumberState::Plural
            } else {
                cue
            };
            distance = 0;
            saw_and_since_cue = false;
            add_feature(
                features,
                &format!("number_state:set:{}:{}", state.as_str(), token),
                0.55,
            );
        }

        if let Some(aux_number) = auxiliary_number(token) {
            let strength = 1.25 / (1.0 + distance as f32 / 4.0);
            add_feature(
                features,
                &format!(
                    "number_state:{}:aux:{}:dist{}",
                    state.as_str(),
                    token,
                    distance.min(8)
                ),
                strength,
            );
            add_feature(
                features,
                &format!("aux:{}:position_bucket{}", token, idx / 4),
                0.35,
            );
            match (state, aux_number) {
                (NumberState::Singular, NumberState::Singular)
                | (NumberState::Plural, NumberState::Plural) => {
                    add_feature(features, "number_state:agreement_match", strength);
                    add_feature(
                        features,
                        &format!("number_state:match:{}:{}", state.as_str(), token),
                        strength,
                    );
                }
                (NumberState::Singular, NumberState::Plural)
                | (NumberState::Plural, NumberState::Singular) => {
                    add_feature(features, "number_state:agreement_mismatch", -strength);
                    add_feature(
                        features,
                        &format!("number_state:mismatch:{}:{}", state.as_str(), token),
                        -strength,
                    );
                }
                _ => {
                    add_feature(features, &format!("number_state:unknown_aux:{token}"), 0.2);
                }
            }
        }

        if !is_functionish(token) {
            distance = distance.saturating_add(1).min(12);
        }
    }

    let prior = agreement_prior_score(&tokens);
    add_feature(features, "grammar:agreement_prior", prior);
    add_feature(features, "grammar:agreement_prior_positive", prior.max(0.0));
    add_feature(
        features,
        "grammar:agreement_prior_negative",
        (-prior).max(0.0),
    );
}

pub(super) fn add_recurrent_grammar_slot_features(features: &mut [f32], sentence: &str) {
    let tokens = tokenize(sentence);
    let mut subject_number = NumberState::Unknown;
    let mut nearest_number = NumberState::Unknown;
    let mut subject_animacy = AnimacyState::Unknown;
    let mut subject_profile = None;
    let mut subject_idx = None;
    let mut binding_domain_start = 0usize;
    let mut wh_filler = false;
    let mut relative_depth = 0usize;

    for (idx, token) in tokens.iter().enumerate() {
        if matches!(token.as_str(), "that" | "who" | "which" | "whom") {
            binding_domain_start = idx + 1;
            relative_depth = relative_depth.saturating_add(1);
            add_feature(
                features,
                &format!(
                    "grammar_slot:domain_boundary:{token}:depth{}",
                    relative_depth.min(3)
                ),
                0.35,
            );
        }

        if matches!(token.as_str(), "what" | "who" | "whom" | "which") {
            wh_filler = true;
            add_feature(
                features,
                &format!("grammar_slot:dependency_open:{token}"),
                0.75,
            );
        }

        if let Some(profile) = entity_profile_at(&tokens, idx) {
            let animacy = animacy_cue(token);
            if subject_idx.is_none() && first_predicate_index_after(&tokens, idx).is_some() {
                subject_idx = Some(idx);
                subject_profile = Some(profile);
                subject_number = profile.number;
                subject_animacy = animacy;
                add_feature(
                    features,
                    &format!(
                        "grammar_slot:role_subject:{}:{}:{}",
                        profile.number.as_str(),
                        profile.gender.as_str(),
                        animacy.as_str()
                    ),
                    0.8,
                );
            }
            if profile.number != NumberState::Unknown {
                nearest_number = profile.number;
                add_feature(
                    features,
                    &format!("grammar_slot:nearest_number:{}", nearest_number.as_str()),
                    0.2,
                );
            }
        }

        if let Some(verb_number) = finite_verb_number(&tokens, idx) {
            add_feature(
                features,
                &format!(
                    "grammar_slot:agreement_query:{}:{}",
                    subject_number.as_str(),
                    verb_number.as_str()
                ),
                0.65,
            );
            match (subject_number, verb_number) {
                (NumberState::Singular, NumberState::Singular)
                | (NumberState::Plural, NumberState::Plural) => {
                    add_feature(features, "grammar_slot:agreement_match", 1.1);
                }
                (NumberState::Singular, NumberState::Plural)
                | (NumberState::Plural, NumberState::Singular) => {
                    add_feature(features, "grammar_slot:agreement_mismatch", -1.1);
                }
                _ => {}
            }
            if nearest_number != NumberState::Unknown
                && subject_number != NumberState::Unknown
                && nearest_number != subject_number
            {
                add_feature(
                    features,
                    &format!(
                        "grammar_slot:agreement_attractor:{}>{}",
                        subject_number.as_str(),
                        nearest_number.as_str()
                    ),
                    0.7,
                );
            }
        }

        if let Some(reflexive) = reflexive_profile(token) {
            let antecedent = (binding_domain_start..idx)
                .rev()
                .find_map(|candidate_idx| entity_profile_at(&tokens, candidate_idx))
                .or(subject_profile);
            if let Some(antecedent) = antecedent {
                add_profile_match_features(
                    features,
                    "grammar_slot:binding_domain",
                    antecedent,
                    reflexive,
                    1.25,
                );
                add_feature(
                    features,
                    "grammar_slot:binding_domain_score",
                    profile_match_score(antecedent, reflexive),
                );
            } else {
                add_feature(features, "grammar_slot:binding_unresolved", -0.5);
            }
            add_feature(
                features,
                &format!(
                    "grammar_slot:reflexive:{}:{}:dist{}",
                    reflexive.number.as_str(),
                    reflexive.gender.as_str(),
                    idx.saturating_sub(binding_domain_start).min(8)
                ),
                0.55,
            );
        }

        if transitive_verb(token) {
            let object_present = has_direct_object_after(&tokens, idx);
            add_feature(
                features,
                if object_present {
                    "grammar_slot:role_object_present"
                } else {
                    "grammar_slot:role_object_missing"
                },
                if object_present { 0.7 } else { -0.8 },
            );
        }

        if intransitive_verb(token) {
            let object_present = has_direct_object_after(&tokens, idx);
            if object_present {
                add_feature(
                    features,
                    "grammar_slot:intransitive_object_intrusion",
                    -0.75,
                );
            } else {
                add_feature(features, "grammar_slot:intransitive_no_object", 0.45);
            }
        }

        if animate_subject_required_verb(token) && subject_animacy != AnimacyState::Unknown {
            add_feature(
                features,
                &format!(
                    "grammar_slot:animacy_query:{}:{}",
                    subject_animacy.as_str(),
                    token
                ),
                0.55,
            );
            match subject_animacy {
                AnimacyState::Animate => {
                    add_feature(features, "grammar_slot:animacy_match", 0.8);
                }
                AnimacyState::Inanimate => {
                    add_feature(features, "grammar_slot:animacy_mismatch", -0.9);
                }
                AnimacyState::Unknown => {}
            }
        }
    }

    if wh_filler {
        if ends_with_gap_like(&tokens) {
            add_feature(features, "grammar_slot:dependency_resolved_gap", 1.0);
        } else {
            add_feature(features, "grammar_slot:dependency_requires_object", -0.35);
        }
    }

    if let Some(subject_idx) = subject_idx {
        add_feature(
            features,
            &format!("grammar_slot:subject_position_bucket{}", subject_idx / 4),
            0.3,
        );
    }
}

pub(super) fn sentence_quality_prior(sentence: &str) -> f32 {
    let tokens = tokenize(sentence);
    let mut score = linguistic_governance_score(&tokens);
    for window in tokens.windows(2) {
        if window[0] == "can" && !window[1].ends_with('s') {
            score += 0.8;
        }
        if window[0] == "can" && window[1].ends_with('s') {
            score -= 0.8;
        }
        if matches!(window[0].as_str(), "this" | "that") && !window[1].ends_with('s') {
            score += 0.25;
        }
        if matches!(window[0].as_str(), "these" | "those") && window[1].ends_with('s') {
            score += 0.25;
        }
    }
    score
}

fn linguistic_governance_score(tokens: &[String]) -> f32 {
    let mut score = agreement_prior_score(tokens);

    for (idx, token) in tokens.iter().enumerate() {
        if let Some(det_number) = determiner_number(token) {
            if let Some(noun_idx) = next_content_index(tokens, idx + 1, 5) {
                let noun_number =
                    nominal_number_at(tokens, noun_idx).unwrap_or(NumberState::Unknown);
                match (det_number, noun_number) {
                    (NumberState::Singular, NumberState::Singular)
                    | (NumberState::Plural, NumberState::Plural) => score += 0.45,
                    (NumberState::Singular, NumberState::Plural)
                    | (NumberState::Plural, NumberState::Singular) => score -= 0.65,
                    _ => {}
                }
            }
        }

        if let Some(reflexive) = reflexive_profile(token) {
            if let Some(reconstructed) = reconstruction_subject_profile_after(tokens, idx) {
                score += profile_match_score(reconstructed, reflexive);
            } else if let Some(head) = relative_head_profile_before(tokens, idx) {
                score += 1.2 * profile_match_score(head, reflexive);
            } else if let Some(local) = local_subject_profile_before(tokens, idx) {
                score += 0.8 * profile_match_score(local, reflexive);
            } else if let Some(nearest) = nearest_entity_profile_before(tokens, idx) {
                score += 0.5 * profile_match_score(nearest, reflexive);
            }
        }

        if transitive_verb(token) {
            if has_direct_object_after(tokens, idx) {
                score += 0.7;
            } else if clause_boundary_after(tokens, idx) {
                score -= 0.85;
            }
        }

        if intransitive_verb(token) {
            if has_direct_object_after(tokens, idx) {
                score -= 0.8;
            } else {
                score += 0.35;
            }
        }

        if token == "there" {
            if let Some(verb_idx) = governing_verb_before(tokens, idx) {
                let verb = tokens[verb_idx].as_str();
                if existential_there_compatible_verb(verb) {
                    score += 0.7;
                }
                if existential_there_incompatible_verb(verb) {
                    score -= 0.8;
                }
            }
        }

        if token == "it"
            && tokens
                .get(idx + 1)
                .is_some_and(|next| next.as_str() == "to")
        {
            if let Some(verb_idx) = governing_verb_before(tokens, idx) {
                let verb = tokens[verb_idx].as_str();
                if expletive_it_compatible_verb(verb) {
                    score += 0.65;
                }
                if expletive_it_incompatible_verb(verb) {
                    score -= 0.75;
                }
            }
        }
    }

    for aux_idx in 0..tokens.len() {
        if !is_be_auxiliary(&tokens[aux_idx]) {
            continue;
        }
        if let Some(participle_idx) = next_content_index(tokens, aux_idx + 1, 4) {
            let participle = tokens[participle_idx].as_str();
            if passive_compatible_participle(participle) {
                score += 0.5;
            }
            if passive_incompatible_participle(participle) {
                score -= 0.8;
            }
            if let Some(by_idx) = tokens[participle_idx + 1..]
                .iter()
                .position(|token| token == "by")
                .map(|offset| participle_idx + 1 + offset)
            {
                if let Some(agent_idx) = next_content_index(tokens, by_idx + 1, 4) {
                    match animacy_cue(&tokens[agent_idx]) {
                        AnimacyState::Animate => score += 0.45,
                        AnimacyState::Inanimate => score -= 0.45,
                        AnimacyState::Unknown => {}
                    }
                }
            }
        }
    }

    if tokens.first().is_some_and(|token| {
        matches!(token.as_str(), "what" | "which" | "whose")
            && tokens
                .get(1)
                .is_some_and(|next| is_auxiliary_or_modal(next))
            && tail_contains_bare_nominal(tokens, 2)
    }) {
        score -= 0.75;
    }
    score += adjunct_island_score(tokens);
    score += animate_subject_score(tokens);

    for idx in 1..tokens.len().saturating_sub(1) {
        if matches!(tokens[idx].as_str(), "what" | "which" | "whose")
            && nominal_number_cue(&tokens[idx + 1]) != NumberState::Unknown
        {
            score += 0.65;
        }
    }

    score
}

fn number_cue(token: &str) -> Option<NumberState> {
    match token {
        "this" | "that" | "a" | "an" | "each" | "every" | "he" | "she" | "it" | "him" | "her"
        | "himself" | "herself" | "itself" | "one" => Some(NumberState::Singular),
        "these" | "those" | "they" | "them" | "their" | "we" | "us" | "ourselves"
        | "themselves" | "both" | "many" | "several" => Some(NumberState::Plural),
        _ if token.len() > 3
            && token.ends_with('s')
            && !matches!(token, "is" | "was" | "has" | "does" | "this" | "his") =>
        {
            Some(NumberState::Plural)
        }
        _ => None,
    }
}

fn nominal_number_cue(token: &str) -> NumberState {
    match token {
        "alumni" | "analyses" | "cacti" | "children" | "criteria" | "crises" | "diagnoses"
        | "feet" | "fungi" | "geese" | "hypotheses" | "larvae" | "lice" | "men" | "mice"
        | "nuclei" | "oases" | "offspring" | "oxen" | "parentheses" | "people" | "phenomena"
        | "stimuli" | "synopses" | "teeth" | "theses" | "women" => NumberState::Plural,
        "alumnus" | "analysis" | "child" | "criterion" | "crisis" | "diagnosis" | "foot"
        | "goose" | "hypothesis" | "larva" | "louse" | "man" | "mouse" | "nucleus" | "oasis"
        | "ox" | "parenthesis" | "person" | "phenomenon" | "stimulus" | "synopsis" | "thesis"
        | "tooth" | "woman" => NumberState::Singular,
        "news" | "series" | "species" => NumberState::Unknown,
        _ => number_cue(token).unwrap_or(NumberState::Unknown),
    }
}

fn determiner_number(token: &str) -> Option<NumberState> {
    match token {
        "this" | "that" => Some(NumberState::Singular),
        "these" | "those" => Some(NumberState::Plural),
        _ => None,
    }
}

fn reflexive_profile(token: &str) -> Option<EntityProfile> {
    match token {
        "himself" => Some(EntityProfile {
            number: NumberState::Singular,
            gender: GenderState::Male,
        }),
        "herself" => Some(EntityProfile {
            number: NumberState::Singular,
            gender: GenderState::Female,
        }),
        "itself" => Some(EntityProfile {
            number: NumberState::Singular,
            gender: GenderState::Neutral,
        }),
        "themselves" => Some(EntityProfile {
            number: NumberState::Plural,
            gender: GenderState::Unknown,
        }),
        "myself" | "yourself" => Some(EntityProfile {
            number: NumberState::Singular,
            gender: GenderState::Unknown,
        }),
        "ourselves" | "yourselves" => Some(EntityProfile {
            number: NumberState::Plural,
            gender: GenderState::Unknown,
        }),
        _ => None,
    }
}

fn entity_profile_at(tokens: &[String], idx: usize) -> Option<EntityProfile> {
    let token = tokens.get(idx)?.as_str();
    let number = nominal_number_at(tokens, idx).unwrap_or_else(|| nominal_number_cue(token));
    let gender = entity_gender(token);
    if number == NumberState::Unknown && gender == GenderState::Unknown {
        None
    } else {
        Some(EntityProfile { number, gender })
    }
}

fn entity_gender(token: &str) -> GenderState {
    match token {
        "he" | "him" | "his" | "boy" | "boys" | "man" | "men" | "guy" | "guys" | "husband"
        | "father" | "son" | "grandson" | "boyfriend" | "waiter" | "actor" | "bruce" | "curtis"
        | "carl" | "donald" | "eric" | "gary" | "gerald" | "james" | "joel" | "jerry"
        | "kenneth" | "keith" | "kevin" | "larry" | "lawrence" | "liam" | "marcus" | "mark"
        | "patrick" | "paul" | "phillip" | "raymond" | "richard" | "robert" | "ronald"
        | "samuel" | "scott" | "stephen" | "steven" | "steve" | "thomas" | "timothy" | "travis"
        | "wayne" | "william" => GenderState::Male,
        "she" | "her" | "girl" | "girls" | "woman" | "women" | "lady" | "actress" | "waitress"
        | "mother" | "daughter" | "granddaughter" | "wife" | "girlfriend" | "alice" | "carla"
        | "caroline" | "christine" | "cindy" | "colleen" | "connie" | "dawn" | "deborah"
        | "diane" | "donna" | "elizabeth" | "eva" | "florence" | "holly" | "irene"
        | "jacqueline" | "jessica" | "karla" | "kayla" | "kimberley" | "laura" | "marie"
        | "melinda" | "monet" | "natalie" | "nicole" | "nina" | "rhonda" | "rose" | "sabrina"
        | "samantha" | "sara" | "sarah" | "sherry" | "stacey" | "stephanie" | "susan"
        | "tamara" | "tanya" | "tara" | "tracy" => GenderState::Female,
        "it" | "article" | "bicycle" | "bike" | "book" | "commentary" | "company" | "door"
        | "drawing" | "government" | "glacier" | "glaciers" | "hospital" | "hospitals"
        | "library" | "newspaper" | "play" | "popsicle" | "pork" | "river" | "screen" | "vase" => {
            GenderState::Neutral
        }
        _ => GenderState::Unknown,
    }
}

fn auxiliary_number(token: &str) -> Option<NumberState> {
    match token {
        "is" | "isn" | "was" | "wasn" | "has" | "hasn" | "does" | "doesn" => {
            Some(NumberState::Singular)
        }
        "are" | "aren" | "were" | "weren" | "have" | "haven" | "do" | "don" => {
            Some(NumberState::Plural)
        }
        _ => None,
    }
}

fn is_be_auxiliary(token: &str) -> bool {
    matches!(
        token,
        "am" | "are"
            | "aren"
            | "be"
            | "been"
            | "being"
            | "is"
            | "isn"
            | "was"
            | "wasn"
            | "were"
            | "weren"
    )
}

fn is_auxiliary_or_modal(token: &str) -> bool {
    auxiliary_number(token).is_some()
        || is_be_auxiliary(token)
        || matches!(
            token,
            "can"
                | "could"
                | "couldn"
                | "did"
                | "didn"
                | "had"
                | "hadn"
                | "may"
                | "might"
                | "must"
                | "shall"
                | "should"
                | "shouldn"
                | "will"
                | "won"
                | "would"
                | "wouldn"
        )
}

fn next_content_index(tokens: &[String], start: usize, max_distance: usize) -> Option<usize> {
    let end = (start + max_distance).min(tokens.len());
    (start..end).find(|&idx| {
        let token = tokens[idx].as_str();
        !is_functionish(token) && !matches!(token, "not" | "t" | "n")
    })
}

fn governing_verb_before(tokens: &[String], idx: usize) -> Option<usize> {
    let start = idx.saturating_sub(6);
    (start..idx).rev().find(|&token_idx| {
        let token = tokens[token_idx].as_str();
        existential_there_compatible_verb(token)
            || existential_there_incompatible_verb(token)
            || expletive_it_compatible_verb(token)
            || expletive_it_incompatible_verb(token)
    })
}

fn nearest_number_cue_before(tokens: &[String], idx: usize) -> Option<NumberState> {
    let end = idx.min(tokens.len());
    let start = end.saturating_sub(10);
    tokens[start..end]
        .iter()
        .rev()
        .find_map(|token| number_cue(token))
}

fn nearest_entity_profile_before(tokens: &[String], idx: usize) -> Option<EntityProfile> {
    let start = idx.saturating_sub(10);
    (start..idx)
        .rev()
        .find_map(|token_idx| entity_profile_at(tokens, token_idx))
}

fn local_subject_profile_before(tokens: &[String], idx: usize) -> Option<EntityProfile> {
    let start = tokens[..idx]
        .iter()
        .rposition(|token| matches!(token.as_str(), "that" | "who" | "whom" | "which"))
        .map_or(0, |pos| pos + 1);
    (start..idx)
        .rev()
        .find_map(|token_idx| entity_profile_at(tokens, token_idx))
}

fn relative_head_profile_before(tokens: &[String], idx: usize) -> Option<EntityProfile> {
    let marker = tokens[..idx]
        .iter()
        .enumerate()
        .rev()
        .find_map(|(token_idx, token)| {
            relative_marker_at(tokens, token_idx, token).then_some(token_idx)
        })?;
    (0..marker)
        .rev()
        .find_map(|token_idx| entity_profile_at(tokens, token_idx))
}

fn relative_marker_at(tokens: &[String], idx: usize, token: &str) -> bool {
    match token {
        "who" | "whom" | "which" => true,
        "that" => !is_complementizer_that_at(tokens, idx),
        _ => false,
    }
}

fn is_complementizer_that_at(tokens: &[String], idx: usize) -> bool {
    if tokens.get(idx).is_none_or(|token| token != "that") {
        return false;
    }
    previous_content_index(tokens, idx).is_some_and(|prev_idx| is_bridge_verb(&tokens[prev_idx]))
}

fn previous_content_index(tokens: &[String], idx: usize) -> Option<usize> {
    (0..idx).rev().find(|&token_idx| {
        let token = tokens[token_idx].as_str();
        !is_functionish(token) && !matches!(token, "not" | "t" | "n")
    })
}

fn nearest_content_token_before(tokens: &[String], idx: usize) -> Option<&str> {
    tokens[..idx]
        .iter()
        .rev()
        .find(|token| is_entityish_token(token))
        .map(String::as_str)
}

fn embedded_subject_token_before(tokens: &[String], idx: usize) -> Option<&str> {
    let bridge_idx = tokens[..idx]
        .iter()
        .rposition(|token| is_bridge_verb(token))?;
    tokens[bridge_idx + 1..idx]
        .iter()
        .find(|token| is_entityish_token(token))
        .map(String::as_str)
}

fn matrix_subject_token_before_bridge(tokens: &[String], idx: usize) -> Option<&str> {
    let bridge_idx = tokens[..idx]
        .iter()
        .rposition(|token| is_bridge_verb(token))?;
    tokens[..bridge_idx]
        .iter()
        .rev()
        .find(|token| is_entityish_token(token))
        .map(String::as_str)
}

fn reconstruction_subject_profile_after(tokens: &[String], idx: usize) -> Option<EntityProfile> {
    let marker_idx = tokens[idx + 1..]
        .iter()
        .position(|token| matches!(token.as_str(), "that" | "who" | "which"))
        .map(|offset| idx + 1 + offset)?;
    let search_start = marker_idx + 1;
    let search_end = (search_start + 6).min(tokens.len());
    for token_idx in search_start..search_end {
        let token = tokens[token_idx].as_str();
        if is_auxiliary_or_modal(token) || finite_verb_number(tokens, token_idx).is_some() {
            break;
        }
        if let Some(profile) = entity_profile_at(tokens, token_idx) {
            return Some(profile);
        }
    }
    None
}

fn is_bridge_verb(token: &str) -> bool {
    matches!(
        token,
        "explain"
            | "explained"
            | "explaining"
            | "explains"
            | "forget"
            | "forgot"
            | "imagine"
            | "imagined"
            | "imagines"
            | "imagining"
            | "remember"
            | "remembered"
            | "remembers"
            | "realize"
            | "realized"
            | "realizes"
            | "reveal"
            | "revealed"
            | "reveals"
            | "said"
            | "say"
            | "saying"
            | "says"
            | "think"
            | "thinking"
            | "thinks"
            | "thought"
    )
}

fn is_entityish_token(token: &str) -> bool {
    !is_functionish(token)
        && auxiliary_number(token).is_none()
        && !is_bridge_verb(token)
        && !matches!(
            token,
            "can"
                | "could"
                | "couldn"
                | "did"
                | "didn"
                | "had"
                | "hadn"
                | "has"
                | "hasn"
                | "might"
                | "should"
                | "shouldn"
                | "will"
                | "won"
                | "would"
                | "wouldn"
                | "like"
                | "about"
                | "from"
        )
}

fn animacy_cue(token: &str) -> AnimacyState {
    match token {
        "actor" | "actress" | "adult" | "adults" | "alumni" | "boy" | "boys" | "cashier"
        | "cashiers" | "child" | "children" | "customer" | "customers" | "dancer" | "dancers"
        | "doctor" | "doctors" | "driver" | "drivers" | "girl" | "girls" | "guest" | "guests"
        | "guy" | "guys" | "lady" | "ladies" | "man" | "men" | "natalie" | "offspring"
        | "patient" | "patients" | "pedestrian" | "pedestrians" | "person" | "people"
        | "samanta" | "samantha" | "senator" | "senators" | "student" | "students" | "teacher"
        | "teachers" | "teenager" | "teenagers" | "waiter" | "waiters" | "waitress"
        | "waitresses" | "woman" | "women" => AnimacyState::Animate,
        "bike" | "bicycle" | "bicycles" | "blouse" | "blouses" | "book" | "books" | "committee"
        | "committees" | "computer" | "computers" | "convertible" | "convertibles"
        | "diagnosis" | "fish" | "glass" | "glasses" | "jacket" | "jackets" | "ladder"
        | "movie" | "movies" | "newspaper" | "newspapers" | "nuclei" | "parenthesis" | "pasta"
        | "pie" | "plate" | "projector" | "projectors" | "screen" | "screens" | "sock"
        | "socks" | "synopses" | "theses" | "truck" | "trucks" | "turtle" | "window"
        | "windows" => AnimacyState::Inanimate,
        _ => match entity_gender(token) {
            GenderState::Male | GenderState::Female => AnimacyState::Animate,
            GenderState::Neutral => AnimacyState::Inanimate,
            GenderState::Unknown => AnimacyState::Unknown,
        },
    }
}

fn has_direct_object_after(tokens: &[String], verb_idx: usize) -> bool {
    let end = (verb_idx + 5).min(tokens.len());
    for idx in verb_idx + 1..end {
        let token = tokens[idx].as_str();
        if matches!(
            token,
            "by" | "for" | "from" | "in" | "into" | "of" | "on" | "to" | "with"
        ) {
            return false;
        }
        if matches!(token, "not" | "t") || is_auxiliary_or_modal(token) {
            continue;
        }
        if determiner_number(token).is_some()
            || nominal_number_at(tokens, idx).is_some()
            || entity_gender(token) != GenderState::Unknown
        {
            return true;
        }
    }
    false
}

fn first_subject_index(tokens: &[String]) -> Option<usize> {
    let stop = tokens
        .iter()
        .enumerate()
        .find_map(|(idx, token)| {
            (is_auxiliary_or_modal(token)
                || finite_verb_number(tokens, idx).is_some()
                || transitive_verb(token)
                || intransitive_verb(token))
            .then_some(idx)
        })
        .unwrap_or(tokens.len());
    (0..stop.min(8)).rev().find(|&idx| {
        entity_profile_at(tokens, idx).is_some()
            || animacy_cue(&tokens[idx]) != AnimacyState::Unknown
    })
}

fn first_predicate_index_after(tokens: &[String], subject_idx: usize) -> Option<usize> {
    for idx in subject_idx + 1..tokens.len() {
        let token = tokens[idx].as_str();
        if is_auxiliary_or_modal(token) || matches!(token, "not" | "t") {
            continue;
        }
        if transitive_verb(token)
            || intransitive_verb(token)
            || finite_verb_number(tokens, idx).is_some()
        {
            return Some(idx);
        }
    }
    None
}

fn clause_boundary_after(tokens: &[String], verb_idx: usize) -> bool {
    tokens[verb_idx + 1..]
        .iter()
        .take(4)
        .all(|token| is_functionish(token) || matches!(token.as_str(), "not" | "t"))
}

fn tail_contains_bare_nominal(tokens: &[String], start: usize) -> bool {
    (start..tokens.len()).any(|idx| {
        let token = tokens[idx].as_str();
        !is_functionish(token)
            && !is_auxiliary_or_modal(token)
            && nominal_number_at(tokens, idx).is_some()
            && tokens
                .get(idx.saturating_sub(1))
                .is_none_or(|prev| determiner_number(prev).is_none())
    })
}

fn adjunct_island_score(tokens: &[String]) -> f32 {
    if !tokens
        .first()
        .is_some_and(|token| matches!(token.as_str(), "what" | "who" | "whom" | "which"))
    {
        return 0.0;
    }
    let mut score = 0.0;
    for (marker_idx, marker) in tokens.iter().enumerate() {
        if !matches!(
            marker.as_str(),
            "after" | "before" | "while" | "without" | "because"
        ) {
            continue;
        }
        let before_has_object =
            tokens[..marker_idx]
                .iter()
                .enumerate()
                .skip(2)
                .any(|(idx, token)| {
                    !is_functionish(token)
                        && !is_auxiliary_or_modal(token)
                        && nominal_number_at(tokens, idx).is_some()
                });
        let tail = &tokens[marker_idx + 1..];
        let tail_has_object = tail.iter().enumerate().skip(1).any(|(offset, token)| {
            let idx = marker_idx + 1 + offset;
            !is_functionish(token)
                && !is_auxiliary_or_modal(token)
                && nominal_number_at(tokens, idx).is_some()
        });
        let tail_gap = tail
            .last()
            .is_some_and(|last| is_gap_tail_token(last) || transitive_verb(last));
        if tail_has_object {
            score += 1.1;
        }
        if before_has_object && tail_gap {
            score -= 1.35;
        }
    }
    score
}

fn animate_subject_score(tokens: &[String]) -> f32 {
    let Some(subject_idx) = first_subject_index(tokens) else {
        return 0.0;
    };
    let Some(verb_idx) = first_predicate_index_after(tokens, subject_idx) else {
        return 0.0;
    };
    if !animate_subject_required_verb(&tokens[verb_idx]) {
        return 0.0;
    }
    match animacy_cue(&tokens[subject_idx]) {
        AnimacyState::Animate => 0.85,
        AnimacyState::Inanimate => -0.95,
        AnimacyState::Unknown => 0.0,
    }
}

fn add_profile_match_features(
    features: &mut [f32],
    prefix: &str,
    antecedent: EntityProfile,
    reflexive: EntityProfile,
    weight: f32,
) {
    add_feature(
        features,
        &format!(
            "{prefix}:antecedent:{}:{}:reflexive:{}:{}",
            antecedent.number.as_str(),
            antecedent.gender.as_str(),
            reflexive.number.as_str(),
            reflexive.gender.as_str()
        ),
        0.45 * weight,
    );

    if antecedent.number != NumberState::Unknown {
        if antecedent.number == reflexive.number {
            add_feature(features, &format!("{prefix}:number_match"), weight);
        } else {
            add_feature(features, &format!("{prefix}:number_mismatch"), -weight);
        }
    }

    if antecedent.gender != GenderState::Unknown && reflexive.gender != GenderState::Unknown {
        if antecedent.gender == reflexive.gender {
            add_feature(features, &format!("{prefix}:gender_match"), weight);
        } else {
            add_feature(features, &format!("{prefix}:gender_mismatch"), -weight);
        }
    }
}

fn profile_match_score(antecedent: EntityProfile, reflexive: EntityProfile) -> f32 {
    let mut score = 0.0;
    if antecedent.number != NumberState::Unknown {
        if antecedent.number == reflexive.number {
            score += 0.8;
        } else {
            score -= 0.8;
        }
    }
    if antecedent.gender != GenderState::Unknown && reflexive.gender != GenderState::Unknown {
        if antecedent.gender == reflexive.gender {
            score += 0.8;
        } else {
            score -= 0.8;
        }
    }
    score
}

fn finite_verb_number(tokens: &[String], idx: usize) -> Option<NumberState> {
    let token = tokens.get(idx)?.as_str();
    if idx > 0
        && matches!(
            tokens[idx - 1].as_str(),
            "to" | "can" | "could" | "should" | "would" | "might" | "will"
        )
    {
        return None;
    }
    if let Some(aux) = auxiliary_number(token) {
        return Some(aux);
    }
    if singular_present_verb(token) {
        return Some(NumberState::Singular);
    }
    if base_present_verb(token) {
        return Some(NumberState::Plural);
    }
    None
}

fn subject_head_number_before(tokens: &[String], verb_idx: usize) -> Option<NumberState> {
    let mut end = verb_idx;
    if let Some(relative_idx) = tokens[..verb_idx]
        .iter()
        .enumerate()
        .find_map(|(idx, token)| relative_marker_at(tokens, idx, token).then_some(idx))
    {
        end = relative_idx;
    }
    if let Some(boundary_idx) = tokens[..end]
        .iter()
        .position(|token| is_subject_pp_boundary(token))
    {
        end = boundary_idx;
    }
    (0..end)
        .rev()
        .find_map(|idx| nominal_number_at(tokens, idx))
}

fn is_subject_pp_boundary(token: &str) -> bool {
    matches!(
        token,
        "about"
            | "above"
            | "after"
            | "around"
            | "behind"
            | "beside"
            | "between"
            | "by"
            | "in"
            | "inside"
            | "near"
            | "of"
            | "on"
            | "under"
            | "with"
            | "to"
    )
}

fn nearest_nominal_number_before(tokens: &[String], idx: usize) -> Option<NumberState> {
    let start = idx.saturating_sub(8);
    (start..idx)
        .rev()
        .find_map(|token_idx| nominal_number_at(tokens, token_idx))
}

fn nominal_number_at(tokens: &[String], idx: usize) -> Option<NumberState> {
    let token = tokens.get(idx)?.as_str();
    if is_functionish(token) || auxiliary_number(token).is_some() {
        return None;
    }
    let cue = nominal_number_cue(token);
    if cue != NumberState::Unknown {
        return Some(cue);
    }
    let prev = idx
        .checked_sub(1)
        .and_then(|prev_idx| tokens.get(prev_idx))
        .map(String::as_str);
    match prev {
        Some("a" | "an" | "each" | "every" | "this" | "that" | "the") => {
            Some(NumberState::Singular)
        }
        Some("all" | "both" | "many" | "most" | "several" | "these" | "those") => {
            Some(NumberState::Plural)
        }
        _ if token.ends_with('s') && !singular_present_verb(token) => Some(NumberState::Plural),
        _ if entity_gender(token) != GenderState::Unknown => Some(NumberState::Singular),
        _ => None,
    }
}

fn number_match_delta(
    number_a: Option<NumberState>,
    number_b: Option<NumberState>,
    cue: NumberState,
) -> f32 {
    if cue == NumberState::Unknown {
        return 0.0;
    }
    let a_score = if number_a == Some(cue) { 1.0 } else { 0.0 };
    let b_score = if number_b == Some(cue) { 1.0 } else { 0.0 };
    a_score - b_score
}

fn agreement_prior_score(tokens: &[String]) -> f32 {
    let mut score = 0.0;
    for (idx, token) in tokens.iter().enumerate() {
        let Some(aux) = auxiliary_number(token) else {
            continue;
        };
        let cue = subject_head_number_before(tokens, idx).unwrap_or_else(|| {
            let start = idx.saturating_sub(8);
            tokens[start..idx]
                .iter()
                .rev()
                .find_map(|prev| number_cue(prev))
                .unwrap_or(NumberState::Unknown)
        });
        match (cue, aux) {
            (NumberState::Singular, NumberState::Singular)
            | (NumberState::Plural, NumberState::Plural) => score += 0.6,
            (NumberState::Singular, NumberState::Plural)
            | (NumberState::Plural, NumberState::Singular) => score -= 0.6,
            _ => {}
        }
        if idx > 0
            && tokens[idx - 1] == "you"
            && matches!(token.as_str(), "are" | "were" | "have" | "do")
        {
            score += 0.8;
        }
        if idx > 0
            && tokens[idx - 1] == "you"
            && matches!(token.as_str(), "is" | "was" | "has" | "does")
        {
            score -= 0.8;
        }
    }
    score
}
