use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use serde::Serialize;

const LOG_ZERO: f32 = f32::NEG_INFINITY;

#[derive(Clone, Debug, PartialEq)]
pub struct PrefixBeamHypothesis {
    pub token_ids: Vec<u32>,
    pub score: f32,
    pub blank_score: f32,
    pub non_blank_score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PredictionRecord {
    pub utt_id: String,
    pub token_ids: Vec<u32>,
    pub score: f32,
    pub alignments: Vec<TokenAlignmentRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenAlignmentRecord {
    pub token_id: u32,
    pub start_encoder_t: usize,
    pub end_encoder_t: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_ms: f32,
    pub end_ms: f32,
}

pub fn load_logits_and_lengths(
    tensors_path: &Path,
    logits_key: &str,
    lengths_key: &str,
) -> Result<(Vec<Vec<Vec<f32>>>, Vec<usize>), String> {
    let tensors = candle_core::safetensors::load(tensors_path, &Device::Cpu)
        .map_err(|err| format!("Failed to load {}: {err}", tensors_path.display()))?;
    let logits = tensors
        .get(logits_key)
        .ok_or_else(|| format!("Missing tensor {logits_key:?} in {}.", tensors_path.display()))?;
    let lengths = tensors
        .get(lengths_key)
        .ok_or_else(|| format!("Missing tensor {lengths_key:?} in {}.", tensors_path.display()))?;

    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|err| format!("Failed to cast logits tensor to f32: {err}"))?;
    let lengths = lengths
        .flatten_all()
        .map_err(|err| format!("Failed to flatten lengths tensor: {err}"))?
        .to_dtype(DType::U32)
        .map_err(|err| format!("Failed to cast lengths tensor to u32: {err}"))?;

    let logits = tensor_to_vec3_f32(&logits)?;
    let lengths = tensor_to_vec1_u32(&lengths)?
        .into_iter()
        .map(|value| value as usize)
        .collect::<Vec<_>>();

    if logits.len() != lengths.len() {
        return Err(format!(
            "Batch mismatch between logits ({}) and lengths ({}).",
            logits.len(),
            lengths.len()
        ));
    }
    Ok((logits, lengths))
}

pub fn load_utt_ids(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read utt ids from {}: {err}", path.display()))?;
    let mut utt_ids = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('{') {
            let value: serde_json::Value =
                serde_json::from_str(trimmed).map_err(|err| format!("Invalid JSONL utt id line: {err}"))?;
            let utt_id = value
                .get("utt_id")
                .or_else(|| value.get("id"))
                .or_else(|| value.get("audio_id"))
                .and_then(|value| value.as_str())
                .ok_or_else(|| String::from("JSONL utt id line must contain utt_id, id, or audio_id."))?;
            utt_ids.push(utt_id.to_string());
            continue;
        }
        utt_ids.push(trimmed.to_string());
    }
    Ok(utt_ids)
}

pub fn predict_from_logits(
    utt_ids: &[String],
    logits: &[Vec<Vec<f32>>],
    lengths: &[usize],
    blank_id: u32,
    beam_size: usize,
    token_prune_topk: Option<usize>,
    subsampling_rate: usize,
    right_context: usize,
    frame_shift_ms: f32,
) -> Result<Vec<PredictionRecord>, String> {
    if logits.len() != utt_ids.len() || logits.len() != lengths.len() {
        return Err(format!(
            "Input length mismatch: utt_ids={}, logits={}, lengths={}.",
            utt_ids.len(),
            logits.len(),
            lengths.len()
        ));
    }

    let mut predictions = Vec::with_capacity(utt_ids.len());
    for batch_idx in 0..utt_ids.len() {
        let sequence = &logits[batch_idx];
        let length = lengths[batch_idx];
        if length > sequence.len() {
            return Err(format!(
                "lengths[{batch_idx}]={} exceeds logits time dimension {}.",
                length,
                sequence.len()
            ));
        }
        let log_probs = log_softmax_sequence(&sequence[..length]);
        let hypotheses = prefix_beam_search(&log_probs, blank_id, beam_size, token_prune_topk)?;
        let best = hypotheses
            .first()
            .ok_or_else(|| format!("No beam-search hypothesis generated for batch index {batch_idx}."))?;
        let spans = ctc_forced_align(&log_probs, &best.token_ids, blank_id)?;
        let alignments = build_token_alignments(
            &best.token_ids,
            &spans,
            subsampling_rate,
            right_context,
            frame_shift_ms,
        );
        predictions.push(PredictionRecord {
            utt_id: utt_ids[batch_idx].clone(),
            token_ids: best.token_ids.clone(),
            score: best.score,
            alignments,
        });
    }
    Ok(predictions)
}

pub fn write_predictions_jsonl(path: &Path, predictions: &[PredictionRecord]) -> Result<(), String> {
    let mut output = String::new();
    for prediction in predictions {
        let line = serde_json::to_string(prediction)
            .map_err(|err| format!("Failed to serialize prediction for {}: {err}", prediction.utt_id))?;
        output.push_str(&line);
        output.push('\n');
    }
    fs::write(path, output).map_err(|err| format!("Failed to write {}: {err}", path.display()))
}

pub fn prefix_beam_search(
    log_probs: &[Vec<f32>],
    blank_id: u32,
    beam_size: usize,
    token_prune_topk: Option<usize>,
) -> Result<Vec<PrefixBeamHypothesis>, String> {
    if beam_size == 0 {
        return Err(String::from("beam_size must be >= 1."));
    }
    if log_probs.is_empty() {
        return Ok(vec![PrefixBeamHypothesis {
            token_ids: Vec::new(),
            score: 0.0,
            blank_score: 0.0,
            non_blank_score: LOG_ZERO,
        }]);
    }
    let vocab_size = log_probs[0].len();
    if vocab_size == 0 {
        return Err(String::from("logits vocabulary dimension must be positive."));
    }
    let blank_index = blank_id as usize;
    if blank_index >= vocab_size {
        return Err(format!("blank_id {blank_id} is out of range for vocab size {vocab_size}."));
    }

    let mut beams = vec![(Vec::<u32>::new(), 0.0_f32, LOG_ZERO)];
    for frame in log_probs {
        if frame.len() != vocab_size {
            return Err(String::from("All frames must have the same vocabulary size."));
        }
        let candidates = frame_candidates(frame, blank_index, token_prune_topk);
        let mut next_beams: Vec<(Vec<u32>, f32, f32)> = Vec::new();
        for (prefix, prefix_blank, prefix_non_blank) in &beams {
            let prefix_total = log_addexp(*prefix_blank, *prefix_non_blank);
            for (token_index, token_logp) in &candidates {
                if *token_index == blank_index {
                    let entry = get_or_insert_beam(&mut next_beams, prefix);
                    entry.1 = log_addexp(entry.1, prefix_total + *token_logp);
                    continue;
                }

                let token_id = *token_index as u32;
                let last_token = prefix.last().copied();
                let mut extended = prefix.clone();
                extended.push(token_id);
                if last_token == Some(token_id) {
                    let same_entry = get_or_insert_beam(&mut next_beams, prefix);
                    same_entry.2 = log_addexp(same_entry.2, *prefix_non_blank + *token_logp);

                    let extended_entry = get_or_insert_beam(&mut next_beams, &extended);
                    extended_entry.2 = log_addexp(extended_entry.2, *prefix_blank + *token_logp);
                    continue;
                }

                let extended_entry = get_or_insert_beam(&mut next_beams, &extended);
                extended_entry.2 = log_addexp(extended_entry.2, prefix_total + *token_logp);
            }
        }

        next_beams.sort_by(|left, right| {
            let left_score = log_addexp(left.1, left.2);
            let right_score = log_addexp(right.1, right.2);
            right_score.partial_cmp(&left_score).unwrap_or(Ordering::Equal)
        });
        next_beams.truncate(beam_size);
        beams = next_beams;
    }

    Ok(beams
        .into_iter()
        .map(|(token_ids, blank_score, non_blank_score)| PrefixBeamHypothesis {
            score: log_addexp(blank_score, non_blank_score),
            token_ids,
            blank_score,
            non_blank_score,
        })
        .collect())
}

pub fn ctc_forced_align(
    log_probs: &[Vec<f32>],
    token_ids: &[u32],
    blank_id: u32,
) -> Result<Vec<(usize, usize)>, String> {
    if token_ids.is_empty() {
        return Ok(Vec::new());
    }
    if log_probs.is_empty() {
        return Err(String::from("Cannot align an empty log-prob sequence."));
    }
    let vocab_size = log_probs[0].len();
    let blank_index = blank_id as usize;
    if blank_index >= vocab_size {
        return Err(format!("blank_id {blank_id} is out of range for vocab size {vocab_size}."));
    }

    let mut extended = Vec::with_capacity(token_ids.len() * 2 + 1);
    extended.push(blank_id);
    for token_id in token_ids {
        extended.push(*token_id);
        extended.push(blank_id);
    }

    let time_steps = log_probs.len();
    let num_states = extended.len();
    let mut scores = vec![vec![LOG_ZERO; num_states]; time_steps];
    let mut backpointers = vec![vec![usize::MAX; num_states]; time_steps];
    scores[0][0] = log_probs[0][blank_index];
    if num_states > 1 {
        let token_index = extended[1] as usize;
        scores[0][1] = log_probs[0][token_index];
    }

    for time_idx in 1..time_steps {
        for state_idx in 0..num_states {
            let token_id = extended[state_idx] as usize;
            let mut best_score = scores[time_idx - 1][state_idx];
            let mut best_prev = state_idx;

            if state_idx > 0 && scores[time_idx - 1][state_idx - 1] > best_score {
                best_score = scores[time_idx - 1][state_idx - 1];
                best_prev = state_idx - 1;
            }
            if state_idx > 1
                && extended[state_idx] != blank_id
                && extended[state_idx] != extended[state_idx - 2]
                && scores[time_idx - 1][state_idx - 2] > best_score
            {
                best_score = scores[time_idx - 1][state_idx - 2];
                best_prev = state_idx - 2;
            }
            if best_score == LOG_ZERO {
                continue;
            }

            scores[time_idx][state_idx] = best_score + log_probs[time_idx][token_id];
            backpointers[time_idx][state_idx] = best_prev;
        }
    }

    let mut final_state = num_states - 1;
    if num_states > 1 && scores[time_steps - 1][num_states - 2] > scores[time_steps - 1][num_states - 1] {
        final_state = num_states - 2;
    }
    if scores[time_steps - 1][final_state] == LOG_ZERO {
        return Err(String::from("Unable to compute a valid CTC alignment path."));
    }

    let mut path_states = vec![final_state; time_steps];
    let mut state = final_state;
    for time_idx in (1..time_steps).rev() {
        let prev_state = backpointers[time_idx][state];
        if prev_state == usize::MAX {
            return Err(String::from("CTC alignment backtracking failed."));
        }
        state = prev_state;
        path_states[time_idx - 1] = state;
    }

    let mut spans = Vec::with_capacity(token_ids.len());
    for token_index in 0..token_ids.len() {
        let state_index = token_index * 2 + 1;
        let positions = path_states
            .iter()
            .enumerate()
            .filter_map(|(time_idx, state)| if *state == state_index { Some(time_idx) } else { None })
            .collect::<Vec<_>>();
        if positions.is_empty() {
            return Err(format!("Token at index {token_index} received no alignment span."));
        }
        spans.push((positions[0], *positions.last().unwrap()));
    }
    Ok(spans)
}

pub fn build_token_alignments(
    token_ids: &[u32],
    spans: &[(usize, usize)],
    subsampling_rate: usize,
    right_context: usize,
    frame_shift_ms: f32,
) -> Vec<TokenAlignmentRecord> {
    token_ids
        .iter()
        .zip(spans.iter())
        .map(|(token_id, (start_encoder_t, end_encoder_t))| {
            let start_frame = subsampling_rate * *start_encoder_t;
            let end_frame = subsampling_rate * *end_encoder_t + right_context;
            TokenAlignmentRecord {
                token_id: *token_id,
                start_encoder_t: *start_encoder_t,
                end_encoder_t: *end_encoder_t,
                start_frame,
                end_frame,
                start_ms: start_frame as f32 * frame_shift_ms,
                end_ms: (end_frame as f32 + 1.0) * frame_shift_ms,
            }
        })
        .collect()
}

fn tensor_to_vec3_f32(tensor: &Tensor) -> Result<Vec<Vec<Vec<f32>>>, String> {
    tensor
        .to_vec3::<f32>()
        .map_err(|err| format!("Expected [B, T, V] tensor, got {:?}: {err}", tensor.dims()))
}

fn tensor_to_vec1_u32(tensor: &Tensor) -> Result<Vec<u32>, String> {
    tensor
        .to_vec1::<u32>()
        .map_err(|err| format!("Expected [B] tensor, got {:?}: {err}", tensor.dims()))
}

fn log_softmax_sequence(sequence: &[Vec<f32>]) -> Vec<Vec<f32>> {
    sequence.iter().map(|frame| log_softmax(frame)).collect()
}

fn log_softmax(values: &[f32]) -> Vec<f32> {
    let max_value = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let sum_exp = values.iter().map(|value| (*value - max_value).exp()).sum::<f32>();
    let log_sum_exp = max_value + sum_exp.ln();
    values.iter().map(|value| *value - log_sum_exp).collect()
}

fn frame_candidates(frame: &[f32], blank_index: usize, token_prune_topk: Option<usize>) -> Vec<(usize, f32)> {
    match token_prune_topk {
        Some(topk) if topk > 0 && topk < frame.len() => {
            let mut indexed = frame
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<(usize, f32)>>();
            indexed.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
            indexed.truncate(topk);
            if !indexed.iter().any(|(index, _)| *index == blank_index) {
                indexed.push((blank_index, frame[blank_index]));
            }
            indexed
        }
        _ => frame.iter().copied().enumerate().collect(),
    }
}

fn get_or_insert_beam<'a>(
    beams: &'a mut Vec<(Vec<u32>, f32, f32)>,
    prefix: &[u32],
) -> &'a mut (Vec<u32>, f32, f32) {
    if let Some(position) = beams.iter().position(|(tokens, _, _)| tokens == prefix) {
        return &mut beams[position];
    }
    beams.push((prefix.to_vec(), LOG_ZERO, LOG_ZERO));
    let index = beams.len() - 1;
    &mut beams[index]
}

fn log_addexp(a: f32, b: f32) -> f32 {
    if a == LOG_ZERO {
        return b;
    }
    if b == LOG_ZERO {
        return a;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::{ctc_forced_align, prefix_beam_search, predict_from_logits};

    #[test]
    fn prefix_beam_search_can_beat_greedy_when_paths_merge() {
        let log_probs = vec![vec![0.6_f32.ln(), 0.4_f32.ln()], vec![0.6_f32.ln(), 0.4_f32.ln()]];
        let beam1 = prefix_beam_search(&log_probs, 0, 1, None).unwrap();
        let beam2 = prefix_beam_search(&log_probs, 0, 2, None).unwrap();

        assert!(beam1[0].token_ids.is_empty());
        assert_eq!(beam2[0].token_ids, vec![1]);
    }

    #[test]
    fn predict_from_logits_emits_json_ready_records() {
        let utt_ids = vec![String::from("utt-0"), String::from("utt-1")];
        let logits = vec![
            vec![vec![1.0, 5.0], vec![1.0, 5.0], vec![1.0, 5.0]],
            vec![vec![1.0, 5.0], vec![1.0, 5.0], vec![1.0, 5.0]],
        ];
        let lengths = vec![3, 3];
        let predictions = predict_from_logits(&utt_ids, &logits, &lengths, 0, 2, None, 1, 0, 10.0).unwrap();

        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].utt_id, "utt-0");
        assert!(!predictions[0].token_ids.is_empty());
        assert!(predictions[0].token_ids.iter().all(|token_id| *token_id == 1));
        assert_eq!(predictions[0].alignments.len(), predictions[0].token_ids.len());
    }

    #[test]
    fn ctc_forced_align_returns_monotonic_token_spans() {
        let log_probs = vec![
            vec![0.9_f32.ln(), 0.1_f32.ln(), 1.0e-6_f32.ln()],
            vec![0.1_f32.ln(), 0.8_f32.ln(), 0.1_f32.ln()],
            vec![0.1_f32.ln(), 0.8_f32.ln(), 0.1_f32.ln()],
            vec![0.8_f32.ln(), 0.1_f32.ln(), 0.1_f32.ln()],
            vec![0.1_f32.ln(), 0.1_f32.ln(), 0.8_f32.ln()],
        ];

        let spans = ctc_forced_align(&log_probs, &[1, 2], 0).unwrap();
        assert_eq!(spans, vec![(1, 2), (4, 4)]);
    }
}
