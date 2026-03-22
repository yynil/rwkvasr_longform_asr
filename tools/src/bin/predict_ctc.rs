use std::env;
use std::path::{Path, PathBuf};

use rwkvasr_tools::predict_ctc::{
    load_logits_and_lengths, load_utt_ids, predict_from_logits, write_predictions_jsonl,
};

#[derive(Debug)]
struct Config {
    tensors_path: PathBuf,
    utt_ids_path: PathBuf,
    output_path: PathBuf,
    logits_key: String,
    lengths_key: String,
    blank_id: u32,
    beam_size: usize,
    token_prune_topk: Option<usize>,
    subsampling_rate: usize,
    right_context: usize,
    frame_shift_ms: f32,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[rwkvasr] error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args()?;
    let (logits, lengths) = load_logits_and_lengths(
        &config.tensors_path,
        &config.logits_key,
        &config.lengths_key,
    )?;
    let utt_ids = load_utt_ids(&config.utt_ids_path)?;
    let predictions = predict_from_logits(
        &utt_ids,
        &logits,
        &lengths,
        config.blank_id,
        config.beam_size,
        config.token_prune_topk,
        config.subsampling_rate,
        config.right_context,
        config.frame_shift_ms,
    )?;
    if let Some(parent) = config.output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create output directory {}: {err}",
                parent.display()
            )
        })?;
    }
    write_predictions_jsonl(&config.output_path, &predictions)?;
    println!(
        "predict_ctc wrote {} prediction(s) to {}",
        predictions.len(),
        config.output_path.display()
    );
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut tensors_path: Option<PathBuf> = None;
    let mut utt_ids_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut logits_key = String::from("logits");
    let mut lengths_key = String::from("lengths");
    let mut blank_id = 0_u32;
    let mut beam_size = 8_usize;
    let mut token_prune_topk: Option<usize> = None;
    let mut subsampling_rate = 6_usize;
    let mut right_context = 10_usize;
    let mut frame_shift_ms = 10.0_f32;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tensors-path" => tensors_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--utt-ids-path" => utt_ids_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--output-path" => output_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--logits-key" => logits_key = next_value(&mut args, &arg)?,
            "--lengths-key" => lengths_key = next_value(&mut args, &arg)?,
            "--blank-id" => {
                blank_id = next_value(&mut args, &arg)?
                    .parse::<u32>()
                    .map_err(|err| format!("Invalid --blank-id: {err}"))?;
            }
            "--beam-size" => {
                beam_size = next_value(&mut args, &arg)?
                    .parse::<usize>()
                    .map_err(|err| format!("Invalid --beam-size: {err}"))?;
            }
            "--token-prune-topk" => {
                token_prune_topk = Some(
                    next_value(&mut args, &arg)?
                        .parse::<usize>()
                        .map_err(|err| format!("Invalid --token-prune-topk: {err}"))?,
                );
            }
            "--subsampling-rate" => {
                subsampling_rate = next_value(&mut args, &arg)?
                    .parse::<usize>()
                    .map_err(|err| format!("Invalid --subsampling-rate: {err}"))?;
            }
            "--right-context" => {
                right_context = next_value(&mut args, &arg)?
                    .parse::<usize>()
                    .map_err(|err| format!("Invalid --right-context: {err}"))?;
            }
            "--frame-shift-ms" => {
                frame_shift_ms = next_value(&mut args, &arg)?
                    .parse::<f32>()
                    .map_err(|err| format!("Invalid --frame-shift-ms: {err}"))?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument {other:?}. Use --help for usage.")),
        }
    }

    let tensors_path = tensors_path.ok_or_else(|| String::from("--tensors-path is required."))?;
    let utt_ids_path = utt_ids_path.ok_or_else(|| String::from("--utt-ids-path is required."))?;
    let output_path = output_path.unwrap_or_else(|| default_output_path(&tensors_path));
    if beam_size == 0 {
        return Err(String::from("--beam-size must be >= 1."));
    }

    Ok(Config {
        tensors_path,
        utt_ids_path,
        output_path,
        logits_key,
        lengths_key,
        blank_id,
        beam_size,
        token_prune_topk,
        subsampling_rate,
        right_context,
        frame_shift_ms,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("Expected a value after {flag}."))
}

fn default_output_path(tensors_path: &Path) -> PathBuf {
    let file_stem = tensors_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("ctc_logits");
    tensors_path.with_file_name(format!("{file_stem}.predictions.jsonl"))
}

fn print_help() {
    println!(
        "\
Rust Candle CTC prefix-beam predictor\n\
\n\
Required:\n\
  --tensors-path <PATH>     Safetensors file containing [B, T, V] logits and [B] lengths\n\
  --utt-ids-path <PATH>     Text or JSONL file containing one utterance id per sample\n\
\n\
Optional:\n\
  --output-path <PATH>      JSONL output path (default: <tensors>.predictions.jsonl)\n\
  --logits-key <NAME>       Tensor key for logits (default: logits)\n\
  --lengths-key <NAME>      Tensor key for lengths (default: lengths)\n\
  --blank-id <INT>          CTC blank id (default: 0)\n\
  --beam-size <INT>         Prefix beam size (default: 8)\n\
  --token-prune-topk <INT>  Optional per-frame token pruning\n\
  --subsampling-rate <INT>  Encoder subsampling rate for timestamp projection (default: 6)\n\
  --right-context <INT>     Encoder right-context in input frames (default: 10)\n\
  --frame-shift-ms <FLOAT>  Input frame shift in milliseconds (default: 10)\n\
"
    );
}
