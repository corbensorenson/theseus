//! Exact corpus primitives for the frozen Theseus neural-seed lineage.
//!
//! The Python tokenizer remains the executable reference. This crate must earn
//! canonical routing through differential parity and end-to-end evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const TARGET_BYTE_BEGIN: &str = "<target_token_bytes>";
pub const TARGET_BYTE_END: &str = "</target_token_bytes>";
pub const MAX_TOKEN_BYTES: usize = 512;
pub const KERC_KERNEL_OBJECTIVES: [&str; 2] = [
    "surface_to_kernel_program_v1",
    "kernel_program_to_answer_packet_v1",
];
const KERC_COMPACT_TOKEN_PREFIXES: [&str; 41] = [
    "VERSION:",
    "SERIALIZATION:",
    "NODE_",
    "OP:",
    "MOD:",
    "POL:",
    "QUANT:",
    "CONF:",
    "DERIV:",
    "SPANS:",
    "ROLE:",
    "HANDLE:",
    "CONCEPT:",
    "NUMBER:",
    "QUANTITY:",
    "TEMPORAL:",
    "TEXT:",
    "SYMBOL:",
    "NODE_REF:",
    "LIST_",
    "AMBIG_",
    "PROB:",
    "EVIDENCE:",
    "BYTE:",
    "BOOL:",
    "NULL",
    "ROOT:",
    "PROGRAM_END",
    "ANSWER_VERSION:",
    "CLAIM_",
    "PRED:",
    "DECISION_",
    "DISPOSITION:",
    "UNCERTAINTY:",
    "CONTROLLING:",
    "AMBIGUITY_ID:",
    "REQUIRED_TERM:",
    "REQUIRED_CAVEAT:",
    "STYLE:",
    "ANSWER_END",
    "MACRO:",
];

const CATEGORIES: [&str; 6] = [
    "english_conversation_instruction",
    "english_broad",
    "python",
    "javascript_typescript",
    "html_css",
    "rust",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncodingReceipt {
    pub policy: String,
    pub profile: String,
    pub logical_token_count: usize,
    pub encoded_token_count: usize,
    pub fallback_token_count: usize,
    pub fallback_byte_count: usize,
    pub unknown_token_count: usize,
    pub exact_text_equal: bool,
    pub failure_behavior: String,
    pub fallback_return_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncodedDocument {
    pub logical_tokens: Vec<String>,
    pub ids: Vec<i32>,
    pub receipt: EncodingReceipt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum KercCodeSpace {
    #[serde(rename = "V_K")]
    Kernel,
    #[serde(rename = "V_P")]
    Pointer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KercCodeToken {
    pub text: String,
    pub space: KercCodeSpace,
}

#[derive(Debug, Clone, Serialize)]
pub struct KercEncodingReceipt {
    pub policy: String,
    pub logical_token_count: usize,
    pub encoded_token_count: usize,
    pub encoded_tokens_by_space: BTreeMap<String, usize>,
    pub fallback_token_count: usize,
    pub fallback_byte_count: usize,
    pub unknown_token_count: usize,
    pub exact_text_equal: bool,
    pub failure_behavior: String,
    pub fallback_return_count: usize,
}

#[derive(Debug, Clone)]
struct Piece {
    raw: Vec<u8>,
    token: String,
    id: i32,
}

#[derive(Debug, Clone)]
pub struct ExactVocabulary {
    ids: HashMap<String, i32>,
    inverse: HashMap<i32, String>,
    begin_id: Option<i32>,
    end_id: Option<i32>,
    byte_ids: Vec<Option<i32>>,
    pieces_by_first: Vec<Vec<Piece>>,
    unknown_id: i32,
    fallback_active: bool,
}

impl ExactVocabulary {
    pub fn from_json_path(path: &Path) -> Result<Self, String> {
        let mut raw = String::new();
        File::open(path)
            .map_err(|error| format!("open vocabulary {}: {error}", path.display()))?
            .read_to_string(&mut raw)
            .map_err(|error| format!("read vocabulary {}: {error}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| format!("parse vocabulary {}: {error}", path.display()))?;
        let object = value
            .get("target_vocab")
            .and_then(serde_json::Value::as_object)
            .or_else(|| value.as_object())
            .ok_or_else(|| "target vocabulary must be a JSON object".to_string())?;
        let mut ordered = Vec::with_capacity(object.len());
        for (token, value) in object {
            let id = value
                .as_i64()
                .ok_or_else(|| format!("non-integer vocabulary id for {token:?}"))?;
            let id = i32::try_from(id).map_err(|_| format!("vocabulary id out of range: {id}"))?;
            ordered.push((token.clone(), id));
        }
        ordered.sort_by_key(|row| row.1);
        Self::from_ordered_tokens(ordered)
    }

    pub fn from_json_path_key(path: &Path, key: &str) -> Result<Self, String> {
        let payload: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?,
        )
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
        let map = payload
            .get(key)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{key} missing from {}", path.display()))?;
        let mut ordered = Vec::with_capacity(map.len());
        for (token, value) in map {
            let id = value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| format!("invalid token id for {token:?}"))?;
            ordered.push((token.clone(), id));
        }
        ordered.sort_by_key(|row| row.1);
        Self::from_ordered_tokens(ordered)
    }

    pub fn from_ordered_tokens(tokens: Vec<(String, i32)>) -> Result<Self, String> {
        let mut ids = HashMap::with_capacity(tokens.len());
        let mut inverse = HashMap::with_capacity(tokens.len());
        for (token, id) in &tokens {
            if ids.insert(token.clone(), *id).is_some()
                || inverse.insert(*id, token.clone()).is_some()
            {
                return Err("duplicate token or id in vocabulary".to_string());
            }
        }
        let begin_id = ids.get(TARGET_BYTE_BEGIN).copied();
        let end_id = ids.get(TARGET_BYTE_END).copied();
        let mut byte_ids = vec![None; 256];
        let mut pieces: BTreeMap<Vec<u8>, Piece> = BTreeMap::new();
        for (value, byte_id) in byte_ids.iter_mut().enumerate() {
            let token = format!("<byte:{value:02x}>");
            if let Some(id) = ids.get(&token).copied() {
                *byte_id = Some(id);
                pieces.insert(
                    vec![value as u8],
                    Piece {
                        raw: vec![value as u8],
                        token,
                        id,
                    },
                );
            }
        }
        // Python constructs this inventory in vocabulary insertion order. Sorting
        // by ID reconstructs that order even when the JSON parser reorders keys.
        for (token, id) in &tokens {
            if let Some(raw) = parse_byte_piece(token) {
                if !raw.is_empty() {
                    pieces.insert(
                        raw.clone(),
                        Piece {
                            raw,
                            token: token.clone(),
                            id: *id,
                        },
                    );
                }
            }
        }
        let mut pieces_by_first = vec![Vec::new(); 256];
        for piece in pieces.into_values() {
            pieces_by_first[piece.raw[0] as usize].push(piece);
        }
        for rows in &mut pieces_by_first {
            rows.sort_by(|left, right| {
                right
                    .raw
                    .len()
                    .cmp(&left.raw.len())
                    .then_with(|| left.raw.cmp(&right.raw))
            });
        }
        let fallback_active =
            begin_id.is_some() && end_id.is_some() && byte_ids.iter().all(Option::is_some);
        Ok(Self {
            unknown_id: ids.get("<unk>").copied().unwrap_or(1),
            ids,
            inverse,
            begin_id,
            end_id,
            byte_ids,
            pieces_by_first,
            fallback_active,
        })
    }

    pub fn encode_document(&self, text: &str, category: &str) -> Result<EncodedDocument, String> {
        if !CATEGORIES.contains(&category) {
            return Err(format!("unknown MoECOT tokenizer category: {category}"));
        }
        let logical = exact_text_tokens(text);
        let owned = logical
            .iter()
            .map(|token| (*token).to_string())
            .collect::<Vec<_>>();
        self.encode_logical_tokens(&owned, text)
    }

    pub fn encode_logical_tokens(
        &self,
        logical: &[String],
        expected_text: &str,
    ) -> Result<EncodedDocument, String> {
        let mut ids = Vec::with_capacity(logical.len());
        let mut fallback_token_count = 0usize;
        let mut fallback_byte_count = 0usize;
        let mut unknown_token_count = 0usize;
        for token in logical {
            if let Some(id) = self.ids.get(token.as_str()).copied() {
                ids.push(id);
                continue;
            }
            let payload = token.as_bytes();
            if self.fallback_active && !payload.is_empty() && payload.len() <= MAX_TOKEN_BYTES {
                ids.push(self.begin_id.expect("fallback boundary was validated"));
                ids.extend(self.encode_byte_pieces(payload));
                ids.push(self.end_id.expect("fallback boundary was validated"));
                fallback_token_count += 1;
                fallback_byte_count += payload.len();
            } else {
                ids.push(self.unknown_id);
                unknown_token_count += 1;
            }
        }
        let reconstructed = self.decode_ids(&ids)?;
        let exact = reconstructed == expected_text;
        let logical_token_count = logical.len();
        Ok(EncodedDocument {
            logical_tokens: logical.to_vec(),
            receipt: EncodingReceipt {
                policy: "project_theseus_moecot_language_tokenizer_rust_v1".to_string(),
                profile: "exact_text_v1".to_string(),
                logical_token_count,
                encoded_token_count: ids.len(),
                fallback_token_count,
                fallback_byte_count,
                unknown_token_count,
                exact_text_equal: exact,
                failure_behavior: "reject_without_fallback".to_string(),
                fallback_return_count: 0,
            },
            ids,
        })
    }

    fn encode_byte_pieces(&self, payload: &[u8]) -> Vec<i32> {
        let length = payload.len();
        let mut best: Vec<Option<Vec<&Piece>>> = vec![None; length + 1];
        best[length] = Some(Vec::new());
        for index in (0..length).rev() {
            let mut winner: Option<Vec<&Piece>> = None;
            for piece in &self.pieces_by_first[payload[index] as usize] {
                let next = index + piece.raw.len();
                if next > length || &payload[index..next] != piece.raw.as_slice() {
                    continue;
                }
                let Some(tail) = &best[next] else { continue };
                let mut candidate = Vec::with_capacity(tail.len() + 1);
                candidate.push(piece);
                candidate.extend(tail.iter().copied());
                if winner
                    .as_ref()
                    .map(|current| piece_sequence_cmp(&candidate, current) == Ordering::Less)
                    .unwrap_or(true)
                {
                    winner = Some(candidate);
                }
            }
            best[index] = winner;
        }
        best[0]
            .as_ref()
            .map(|pieces| pieces.iter().map(|piece| piece.id).collect())
            .unwrap_or_else(|| {
                payload
                    .iter()
                    .map(|value| self.byte_ids[*value as usize].expect("byte fallback active"))
                    .collect()
            })
    }

    pub fn decode_ids(&self, ids: &[i32]) -> Result<String, String> {
        let mut output = String::new();
        let mut payload = Vec::new();
        let mut active = false;
        for id in ids {
            let token = self
                .inverse
                .get(id)
                .ok_or_else(|| format!("target token id is absent from vocabulary: {id}"))?;
            if !active {
                if token == TARGET_BYTE_BEGIN {
                    active = true;
                    payload.clear();
                } else if token == TARGET_BYTE_END || parse_any_byte_piece(token).is_some() {
                    return Err(format!("byte fallback boundary fault at token {token:?}"));
                } else {
                    output.push_str(token);
                }
                continue;
            }
            if token == TARGET_BYTE_END {
                if payload.is_empty() {
                    return Err("empty byte fallback span".to_string());
                }
                output.push_str(
                    std::str::from_utf8(&payload)
                        .map_err(|error| format!("invalid UTF-8 fallback span: {error}"))?,
                );
                active = false;
                payload.clear();
            } else if let Some(raw) = parse_any_byte_piece(token) {
                payload.extend_from_slice(&raw);
                if payload.len() > MAX_TOKEN_BYTES {
                    return Err("byte fallback span exceeds bound".to_string());
                }
            } else {
                return Err(format!("byte token expected, observed {token:?}"));
            }
        }
        if active {
            return Err("truncated byte fallback span".to_string());
        }
        Ok(output)
    }
}

pub fn encode_kerc_global_target(
    text: &str,
    kernel_vocab: &ExactVocabulary,
    pointer_vocab: &ExactVocabulary,
    kernel_offset: i32,
    pointer_offset: i32,
) -> Result<(Vec<i32>, KercEncodingReceipt), String> {
    if pointer_offset <= kernel_offset {
        return Err("pointer_offset must exceed kernel_offset".to_string());
    }
    let tokens = kerc_code_tokens(text)?;
    let mut ids = Vec::new();
    let mut by_space = BTreeMap::from([("V_K".to_string(), 0usize), ("V_P".to_string(), 0usize)]);
    let mut fallback_tokens = 0usize;
    let mut fallback_bytes = 0usize;
    let mut unknown = 0usize;
    for token in &tokens {
        let (vocab, offset, key) = match token.space {
            KercCodeSpace::Kernel => (kernel_vocab, kernel_offset, "V_K"),
            KercCodeSpace::Pointer => (pointer_vocab, pointer_offset, "V_P"),
        };
        let encoded =
            vocab.encode_logical_tokens(std::slice::from_ref(&token.text), &token.text)?;
        for value in encoded.ids {
            ids.push(
                value
                    .checked_add(offset)
                    .ok_or_else(|| "KERC global token id overflow".to_string())?,
            );
        }
        *by_space.get_mut(key).expect("typed space initialized") +=
            encoded.receipt.encoded_token_count;
        fallback_tokens += encoded.receipt.fallback_token_count;
        fallback_bytes += encoded.receipt.fallback_byte_count;
        unknown += encoded.receipt.unknown_token_count;
    }
    let reconstructed = decode_kerc_global_target(
        &ids,
        kernel_vocab,
        pointer_vocab,
        kernel_offset,
        pointer_offset,
    )?;
    Ok((
        ids,
        KercEncodingReceipt {
            policy: "project_theseus_kerc_global_dual_code_encoding_rust_v1".to_string(),
            logical_token_count: tokens.len(),
            encoded_token_count: by_space.values().sum(),
            encoded_tokens_by_space: by_space,
            fallback_token_count: fallback_tokens,
            fallback_byte_count: fallback_bytes,
            unknown_token_count: unknown,
            exact_text_equal: reconstructed == text,
            failure_behavior: "reject_without_surface_or_template_fallback".to_string(),
            fallback_return_count: 0,
        },
    ))
}

pub fn decode_kerc_global_target(
    ids: &[i32],
    kernel_vocab: &ExactVocabulary,
    pointer_vocab: &ExactVocabulary,
    kernel_offset: i32,
    pointer_offset: i32,
) -> Result<String, String> {
    let mut output = String::new();
    let mut index = 0usize;
    while index < ids.len() {
        let (vocab, offset) = if ids[index] >= pointer_offset {
            (pointer_vocab, pointer_offset)
        } else if ids[index] >= kernel_offset {
            (kernel_vocab, kernel_offset)
        } else {
            return Err(format!(
                "KERC global token below declared spaces: {}",
                ids[index]
            ));
        };
        let mut local = Vec::new();
        while index < ids.len() {
            let same_space = if offset == pointer_offset {
                ids[index] >= pointer_offset
            } else {
                ids[index] >= kernel_offset && ids[index] < pointer_offset
            };
            if !same_space {
                break;
            }
            local.push(ids[index] - offset);
            index += 1;
        }
        output.push_str(&vocab.decode_ids(&local)?);
    }
    Ok(output)
}

pub fn kerc_code_tokens(text: &str) -> Result<Vec<KercCodeToken>, String> {
    let mut raw = Vec::new();
    let bytes = text.as_bytes();
    let mut outside_start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += text[index..]
                .chars()
                .next()
                .expect("valid UTF-8 boundary")
                .len_utf8();
            continue;
        }
        if outside_start < index {
            append_exact_kerc_tokens(&mut raw, &text[outside_start..index]);
        }
        let string_start = index;
        index += 1;
        let mut escaped = false;
        let mut terminated = false;
        while index < bytes.len() {
            let character = bytes[index];
            index += 1;
            if escaped {
                escaped = false;
            } else if character == b'\\' {
                escaped = true;
            } else if character == b'"' {
                terminated = true;
                break;
            }
        }
        if !terminated {
            return Err("KERC code tokenizer encountered unterminated JSON string".to_string());
        }
        append_bounded_kerc_token(&mut raw, &text[string_start..index]);
        outside_start = index;
    }
    if outside_start < text.len() {
        append_exact_kerc_tokens(&mut raw, &text[outside_start..]);
    }
    let mut tokens = Vec::with_capacity(raw.len());
    let mut cursor = 0usize;
    while cursor < raw.len() {
        if raw[cursor].text == "@"
            && cursor + 1 < raw.len()
            && raw[cursor + 1]
                .text
                .chars()
                .filter(|character| *character != '_')
                .all(char::is_alphanumeric)
            && raw[cursor + 1]
                .text
                .chars()
                .any(|character| character != '_')
        {
            tokens.push(KercCodeToken {
                text: format!("@{}", raw[cursor + 1].text),
                space: KercCodeSpace::Pointer,
            });
            cursor += 2;
        } else {
            tokens.push(raw[cursor].clone());
            cursor += 1;
        }
    }
    if tokens
        .iter()
        .map(|token| token.text.as_str())
        .collect::<String>()
        != text
    {
        return Err("KERC code tokenizer failed exact reconstruction".to_string());
    }
    Ok(tokens)
}

fn append_exact_kerc_tokens(output: &mut Vec<KercCodeToken>, text: &str) {
    for token in exact_text_tokens(text) {
        append_bounded_kerc_token(output, token);
    }
}

fn append_bounded_kerc_token(output: &mut Vec<KercCodeToken>, token: &str) {
    let space = kerc_code_space(token);
    if token.len() <= MAX_TOKEN_BYTES {
        output.push(KercCodeToken {
            text: token.to_string(),
            space,
        });
        return;
    }
    let mut start = 0usize;
    let mut bytes = 0usize;
    for (index, character) in token.char_indices() {
        let width = character.len_utf8();
        if index > start && bytes + width > MAX_TOKEN_BYTES {
            output.push(KercCodeToken {
                text: token[start..index].to_string(),
                space,
            });
            start = index;
            bytes = 0;
        }
        bytes += width;
    }
    if start < token.len() {
        output.push(KercCodeToken {
            text: token[start..].to_string(),
            space,
        });
    }
}

fn kerc_code_space(token: &str) -> KercCodeSpace {
    if token.len() >= 3 && token.starts_with('"') && token.ends_with('"') {
        if let Ok(decoded) = serde_json::from_str::<String>(token) {
            let mut characters = decoded.chars();
            if let Some(prefix) = characters.next() {
                let rest = characters.as_str();
                if matches!(prefix, 'K' | 'P' | 'S')
                    && KERC_COMPACT_TOKEN_PREFIXES
                        .iter()
                        .any(|candidate| rest.starts_with(candidate))
                {
                    return match prefix {
                        'P' => KercCodeSpace::Pointer,
                        _ => KercCodeSpace::Kernel,
                    };
                }
            }
        }
    }
    if matches!(
        token,
        "{" | "}" | "[" | "]" | "(" | ")" | ":" | "," | "\"" | "\\" | " " | "\n" | "\r" | "\t"
    ) || token.chars().all(char::is_whitespace)
        || kerc_pointer_atom(token)
    {
        KercCodeSpace::Pointer
    } else {
        KercCodeSpace::Kernel
    }
}

fn kerc_pointer_atom(token: &str) -> bool {
    if let Some(rest) = token.strip_prefix('@') {
        let mut chars = rest.chars();
        return chars.next().is_some_and(|value| value.is_ascii_uppercase())
            && chars.all(|value| value.is_ascii_alphanumeric() || value == '_');
    }
    let bytes = token.as_bytes();
    let mut index = usize::from(bytes.first() == Some(&b'-'));
    let integer_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index == integer_start {
        return false;
    }
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        let fraction_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index == fraction_start {
            return false;
        }
    }
    if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
        index += 1;
        if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
            index += 1;
        }
        let exponent_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index == exponent_start {
            return false;
        }
    }
    index == bytes.len()
}

fn piece_sequence_cmp(left: &[&Piece], right: &[&Piece]) -> Ordering {
    left.len().cmp(&right.len()).then_with(|| {
        left.iter()
            .map(|piece| piece.token.as_str())
            .cmp(right.iter().map(|piece| piece.token.as_str()))
    })
}

fn parse_byte_piece(token: &str) -> Option<Vec<u8>> {
    token
        .strip_prefix("<bytes:")
        .and_then(|value| value.strip_suffix('>'))
        .and_then(parse_hex)
}

fn parse_any_byte_piece(token: &str) -> Option<Vec<u8>> {
    if let Some(value) = token
        .strip_prefix("<byte:")
        .and_then(|value| value.strip_suffix('>'))
    {
        return parse_hex(value);
    }
    parse_byte_piece(token)
}

fn parse_hex(value: &str) -> Option<Vec<u8>> {
    if value.is_empty()
        || !value.len().is_multiple_of(2)
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return None;
    }
    (0..value.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&value[index..index + 2], 16).ok())
        .collect()
}

#[derive(Debug, Clone, Copy)]
enum AsciiRunClass {
    IdentifierContinue,
    DecimalDigit,
    HexDigitOrUnderscore,
    BinaryDigitOrUnderscore,
}

/// Canonical scanner. On Apple Silicon it uses bounded NEON blocks for the
/// ASCII runs that dominate source corpora, then finishes with the scalar
/// reference. Token boundaries remain defined by `exact_text_tokens_scalar`.
pub fn exact_text_tokens(text: &str) -> Vec<&str> {
    exact_text_tokens_impl(text, true)
}

/// Independent scalar oracle used by differential tests and qualifications.
pub fn exact_text_tokens_scalar(text: &str) -> Vec<&str> {
    exact_text_tokens_impl(text, false)
}

fn exact_text_tokens_impl(text: &str, accelerated: bool) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        let start = index;
        if bytes[index] == b'\r' {
            index += 1;
            if index < bytes.len() && bytes[index] == b'\n' {
                index += 1;
            }
        } else if bytes[index] == b'\n' {
            index += 1;
        } else if is_non_newline_whitespace(text, index) {
            index = advance_while(text, index, |character| {
                character != '\r' && character != '\n' && character.is_whitespace()
            });
        } else if is_identifier_start(bytes[index]) {
            index += 1;
            index = scan_ascii_run(bytes, index, AsciiRunClass::IdentifierContinue, accelerated);
        } else if bytes[index] == b'0'
            && index + 2 < bytes.len()
            && matches!(bytes[index + 1], b'x' | b'X')
            && is_hex_digit_or_underscore(bytes[index + 2])
        {
            index += 3;
            index = scan_ascii_run(
                bytes,
                index,
                AsciiRunClass::HexDigitOrUnderscore,
                accelerated,
            );
        } else if bytes[index] == b'0'
            && index + 2 < bytes.len()
            && matches!(bytes[index + 1], b'b' | b'B')
            && matches!(bytes[index + 2], b'0' | b'1' | b'_')
        {
            index += 3;
            index = scan_ascii_run(
                bytes,
                index,
                AsciiRunClass::BinaryDigitOrUnderscore,
                accelerated,
            );
        } else if bytes[index].is_ascii_digit() {
            index += 1;
            index = scan_ascii_run(bytes, index, AsciiRunClass::DecimalDigit, accelerated);
            if index + 1 < bytes.len() && bytes[index] == b'.' && bytes[index + 1].is_ascii_digit()
            {
                index += 2;
                index = scan_ascii_run(bytes, index, AsciiRunClass::DecimalDigit, accelerated);
            }
            if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
                let exponent = index;
                let mut cursor = index + 1;
                if cursor < bytes.len() && matches!(bytes[cursor], b'+' | b'-') {
                    cursor += 1;
                }
                let digits = cursor;
                cursor = scan_ascii_run(bytes, cursor, AsciiRunClass::DecimalDigit, accelerated);
                if cursor > digits {
                    index = cursor;
                } else {
                    index = exponent;
                }
            }
        } else {
            index += text[index..]
                .chars()
                .next()
                .expect("valid string boundary")
                .len_utf8();
        }
        tokens.push(&text[start..index]);
    }
    tokens
}

fn scan_ascii_run(
    bytes: &[u8],
    mut index: usize,
    class: AsciiRunClass,
    accelerated: bool,
) -> usize {
    #[cfg(target_arch = "aarch64")]
    if accelerated {
        while index + 16 <= bytes.len() {
            // SAFETY: the length check above proves that the 16-byte unaligned
            // load is in bounds. AArch64 guarantees NEON availability.
            if !unsafe { neon_all_match(bytes.as_ptr().add(index), class) } {
                break;
            }
            index += 16;
        }
    }
    let _ = accelerated;
    while index < bytes.len() && ascii_class_matches(bytes[index], class) {
        index += 1;
    }
    index
}

fn ascii_class_matches(byte: u8, class: AsciiRunClass) -> bool {
    match class {
        AsciiRunClass::IdentifierContinue => is_identifier_continue(byte),
        AsciiRunClass::DecimalDigit => byte.is_ascii_digit(),
        AsciiRunClass::HexDigitOrUnderscore => is_hex_digit_or_underscore(byte),
        AsciiRunClass::BinaryDigitOrUnderscore => matches!(byte, b'0' | b'1' | b'_'),
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_all_match(pointer: *const u8, class: AsciiRunClass) -> bool {
    use std::arch::aarch64::{
        vandq_u8, vcgeq_u8, vcleq_u8, vdupq_n_u8, vld1q_u8, vminvq_u8, vorrq_u8,
    };

    let value = vld1q_u8(pointer);
    let range = |lower: u8, upper: u8| {
        vandq_u8(
            vcgeq_u8(value, vdupq_n_u8(lower)),
            vcleq_u8(value, vdupq_n_u8(upper)),
        )
    };
    let equal = |expected: u8| {
        vandq_u8(
            vcgeq_u8(value, vdupq_n_u8(expected)),
            vcleq_u8(value, vdupq_n_u8(expected)),
        )
    };
    let valid = match class {
        AsciiRunClass::IdentifierContinue => vorrq_u8(
            vorrq_u8(range(b'a', b'z'), range(b'A', b'Z')),
            vorrq_u8(range(b'0', b'9'), vorrq_u8(equal(b'_'), equal(b'$'))),
        ),
        AsciiRunClass::DecimalDigit => range(b'0', b'9'),
        AsciiRunClass::HexDigitOrUnderscore => vorrq_u8(
            vorrq_u8(range(b'0', b'9'), range(b'a', b'f')),
            vorrq_u8(range(b'A', b'F'), equal(b'_')),
        ),
        AsciiRunClass::BinaryDigitOrUnderscore => {
            vorrq_u8(vorrq_u8(equal(b'0'), equal(b'1')), equal(b'_'))
        }
    };
    vminvq_u8(valid) == u8::MAX
}

fn is_non_newline_whitespace(text: &str, index: usize) -> bool {
    text[index..]
        .chars()
        .next()
        .map(|character| character != '\r' && character != '\n' && character.is_whitespace())
        .unwrap_or(false)
}

fn advance_while<F>(text: &str, mut index: usize, predicate: F) -> usize
where
    F: Fn(char) -> bool,
{
    while index < text.len() {
        let character = text[index..].chars().next().expect("valid string boundary");
        if !predicate(character) {
            break;
        }
        index += character.len_utf8();
    }
    index
}

fn is_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$')
}

fn is_identifier_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$')
}

fn is_hex_digit_or_underscore(byte: u8) -> bool {
    byte.is_ascii_hexdigit() || byte == b'_'
}

pub fn ids_sha256(ids: &[i32]) -> String {
    let mut digest = Sha256::new();
    for value in ids {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocabulary() -> ExactVocabulary {
        let mut tokens = vec![
            ("<unk>".to_string(), 1),
            (TARGET_BYTE_BEGIN.to_string(), 2),
            (TARGET_BYTE_END.to_string(), 3),
            ("hello".to_string(), 4),
            (" ".to_string(), 5),
        ];
        for value in 0..256usize {
            tokens.push((format!("<byte:{value:02x}>"), 6 + value as i32));
        }
        tokens.push(("<bytes:776f>".to_string(), 262));
        ExactVocabulary::from_ordered_tokens(tokens).unwrap()
    }

    #[test]
    fn exact_scanner_covers_edge_classes_losslessly() {
        let text = "hello\r\n  $x 0x1_f 0b10_ 12.5e-2 . λ\t\u{2003}";
        let tokens = exact_text_tokens(text);
        assert_eq!(tokens.concat(), text);
        assert_eq!(
            tokens,
            vec![
                "hello",
                "\r\n",
                "  ",
                "$x",
                " ",
                "0x1_f",
                " ",
                "0b10_",
                " ",
                "12.5e-2",
                " ",
                ".",
                " ",
                "λ",
                "\t\u{2003}"
            ]
        );
    }

    #[test]
    fn accelerated_scanner_matches_scalar_oracle_on_adversarial_text() {
        let fragments = [
            "identifier_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$tail",
            "  \t\u{2003}\r\n",
            "0x0123456789abcdefABCDEF____ 0b01010101____ 1234567890.123e-99",
            "fn f(x: &str) -> String { format!(\"{x}🙂\") }\n",
            "<div data-long=\"alpha_beta_123\">héllo</div>",
        ];
        let mut state = 0x9e37_79b9_u32;
        for length in 0..512usize {
            let mut text = String::new();
            while text.len() < length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                text.push_str(fragments[state as usize % fragments.len()]);
            }
            let mut boundary = length.min(text.len());
            while !text.is_char_boundary(boundary) {
                boundary -= 1;
            }
            text.truncate(boundary);
            assert_eq!(exact_text_tokens(&text), exact_text_tokens_scalar(&text));
        }
    }

    #[test]
    fn byte_piece_encoding_roundtrips() {
        let vocab = vocabulary();
        let encoded = vocab
            .encode_document("hello world", "english_broad")
            .unwrap();
        assert_eq!(vocab.decode_ids(&encoded.ids).unwrap(), "hello world");
        assert_eq!(encoded.receipt.fallback_token_count, 1);
        assert_eq!(encoded.receipt.unknown_token_count, 0);
        assert!(encoded.receipt.exact_text_equal);
    }

    #[test]
    fn oversized_unknown_is_explicit_and_not_a_fallback_return() {
        let vocab = vocabulary();
        let text = "x".repeat(MAX_TOKEN_BYTES + 1);
        let encoded = vocab.encode_document(&text, "python").unwrap();
        assert_eq!(encoded.receipt.unknown_token_count, 1);
        assert_eq!(encoded.receipt.fallback_return_count, 0);
        assert!(!encoded.receipt.exact_text_equal);
    }
}
