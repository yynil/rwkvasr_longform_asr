use arrow_array::{
    Array, ArrayRef, BinaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeBinaryArray, LargeStringArray, RecordBatch, StringArray, StructArray, UInt32Array,
    UInt64Array,
};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};
use tar::{Archive, Builder, Header};

const TEXT_KEYS: &[&str] = &[
    "text",
    "sentence",
    "normalized_text",
    "transcription",
    "transcript",
];
const ID_KEYS: &[&str] = &["id", "sid", "utt_id", "utterance_id", "segment_id", "key"];
const AUDIO_SUFFIXES: &[&str] = &["wav", "mp3", "flac"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    GigaSpeechParquet,
    WenetSpeechLhotse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextNormalization {
    None,
    Runtime,
}

impl TextNormalization {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Runtime => "runtime",
        }
    }
}

#[derive(Clone, Debug)]
struct Config {
    command: Command,
    input_root: PathBuf,
    input_glob: String,
    output_root: PathBuf,
    staging_root: Option<PathBuf>,
    staging_max_bytes: Option<u64>,
    language: String,
    shard_prefix: String,
    samples_per_shard: usize,
    max_input_shards: usize,
    max_samples: usize,
    progress_every: usize,
    parquet_batch_size: usize,
    threads: usize,
    skip_missing: bool,
    overwrite: bool,
    resume: bool,
    adopt_existing: bool,
    text_normalization: TextNormalization,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
struct ShardStats {
    input_shards: usize,
    converted_shards: usize,
    resumed_shards: usize,
    samples: usize,
    skipped_samples: usize,
    missing_audio: usize,
    missing_text: usize,
    unsupported_audio: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DoneMarker {
    version: u32,
    command: String,
    input_index: usize,
    input_name: String,
    input_size: u64,
    input_mtime_unix: u64,
    shard_prefix: String,
    #[serde(default)]
    text_normalization: Option<String>,
    output_files: Vec<String>,
    stats: ShardStats,
}

enum WorkerMessage {
    Done {
        input_index: usize,
        input_name: String,
        stats: ShardStats,
    },
    Err(String),
}

struct Sample {
    key: String,
    audio_suffix: String,
    audio_bytes: Vec<u8>,
    metadata: Value,
}

struct ShardedTarWriter {
    output_root: PathBuf,
    staging_root: Option<PathBuf>,
    staging_max_bytes: Option<u64>,
    prefix: String,
    input_index: usize,
    samples_per_shard: usize,
    part_index: usize,
    samples_in_part: usize,
    builder: Option<Builder<File>>,
    current_path: Option<PathBuf>,
    current_final_path: Option<PathBuf>,
    output_files: Vec<String>,
}

impl ShardedTarWriter {
    fn new(
        output_root: PathBuf,
        staging_root: Option<PathBuf>,
        staging_max_bytes: Option<u64>,
        prefix: String,
        input_index: usize,
        samples_per_shard: usize,
    ) -> Self {
        Self {
            output_root,
            staging_root,
            staging_max_bytes,
            prefix,
            input_index,
            samples_per_shard,
            part_index: 0,
            samples_in_part: 0,
            builder: None,
            current_path: None,
            current_final_path: None,
            output_files: Vec::new(),
        }
    }

    fn add_sample(&mut self, sample: Sample) -> Result<(), String> {
        if self.builder.is_none() || self.samples_in_part >= self.samples_per_shard {
            self.open_next_part()?;
        }
        let metadata_bytes = serde_json::to_vec(&sample.metadata)
            .map_err(|err| format!("Failed to serialize metadata for {}: {err}", sample.key))?;
        let audio_name = format!("{}.{}", sample.key, sample.audio_suffix);
        let json_name = format!("{}.json", sample.key);
        self.append_bytes(&audio_name, &sample.audio_bytes)?;
        self.append_bytes(&json_name, &metadata_bytes)?;
        self.samples_in_part += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        if let Some(mut builder) = self.builder.take() {
            builder
                .finish()
                .map_err(|err| format!("Failed to finish output tar: {err}"))?;
        }
        let Some(current_path) = self.current_path.take() else {
            return Ok(());
        };
        let Some(final_path) = self.current_final_path.take() else {
            return Ok(());
        };
        if current_path != final_path {
            publish_staged_file(&current_path, &final_path)?;
        }
        Ok(())
    }

    fn open_next_part(&mut self) -> Result<(), String> {
        self.finish()?;
        fs::create_dir_all(&self.output_root).map_err(|err| {
            format!(
                "Failed to create output root {}: {err}",
                self.output_root.display()
            )
        })?;
        let file_name = format!(
            "{}-{:06}-{:04}.tar",
            self.prefix, self.input_index, self.part_index
        );
        let final_path = self.output_root.join(&file_name);
        let path = match &self.staging_root {
            Some(staging_root) => {
                ensure_staging_capacity(staging_root, self.staging_max_bytes)?;
                let dir = staging_root.join(&self.prefix);
                fs::create_dir_all(&dir).map_err(|err| {
                    format!("Failed to create staging dir {}: {err}", dir.display())
                })?;
                dir.join(format!("{file_name}.part"))
            }
            None => self.output_root.join(format!("{file_name}.part")),
        };
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|err| format!("Failed to remove stale {}: {err}", path.display()))?;
        }
        let file = File::create(&path)
            .map_err(|err| format!("Failed to create output shard {}: {err}", path.display()))?;
        self.builder = Some(Builder::new(file));
        self.current_path = Some(path);
        self.current_final_path = Some(final_path);
        self.output_files.push(file_name);
        self.part_index += 1;
        self.samples_in_part = 0;
        Ok(())
    }

    fn append_bytes(&mut self, path: &str, bytes: &[u8]) -> Result<(), String> {
        let builder = self
            .builder
            .as_mut()
            .ok_or_else(|| String::from("Output tar writer is not open."))?;
        let mut header = Header::new_gnu();
        header
            .set_path(path)
            .map_err(|err| format!("Invalid tar path {path}: {err}"))?;
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_cksum();
        builder
            .append(&header, Cursor::new(bytes))
            .map_err(|err| format!("Failed writing {path}: {err}"))?;
        Ok(())
    }
}

impl Drop for ShardedTarWriter {
    fn drop(&mut self) {
        let _ = self.builder.take();
        if let Some(path) = self.current_path.take() {
            if path.exists() {
                let _ = fs::remove_file(path);
            }
        }
    }
}

fn ensure_staging_capacity(staging_root: &Path, max_bytes: Option<u64>) -> Result<(), String> {
    let Some(max_bytes) = max_bytes else {
        return Ok(());
    };
    if max_bytes == 0 || !staging_root.exists() {
        return Ok(());
    }
    let used = directory_size(staging_root)?;
    if used > max_bytes {
        return Err(format!(
            "Staging root {} uses {} bytes, exceeding --staging-max-bytes {}.",
            staging_root.display(),
            used,
            max_bytes
        ));
    }
    Ok(())
}

fn directory_size(path: &Path) -> Result<u64, String> {
    let mut total = 0_u64;
    for entry in
        fs::read_dir(path).map_err(|err| format!("Failed to read {}: {err}", path.display()))?
    {
        let entry = entry.map_err(|err| format!("Failed to read staging entry: {err}"))?;
        let metadata = entry
            .metadata()
            .map_err(|err| format!("Failed to stat {}: {err}", entry.path().display()))?;
        if metadata.is_dir() {
            total += directory_size(&entry.path())?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}

fn publish_staged_file(staged_path: &Path, final_path: &Path) -> Result<(), String> {
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Failed to create output dir {}: {err}", parent.display()))?;
    }
    if final_path.exists() {
        fs::remove_file(final_path)
            .map_err(|err| format!("Failed to remove stale {}: {err}", final_path.display()))?;
    }
    match fs::rename(staged_path, final_path) {
        Ok(()) => Ok(()),
        Err(rename_err) => {
            let expected_len = fs::metadata(staged_path)
                .map_err(|err| format!("Failed to stat staged {}: {err}", staged_path.display()))?
                .len();
            let copied = fs::copy(staged_path, final_path).map_err(|copy_err| {
                format!(
                    "Failed to publish staged file {} -> {}: rename error: {}; copy error: {}",
                    staged_path.display(),
                    final_path.display(),
                    rename_err,
                    copy_err
                )
            })?;
            if copied != expected_len {
                return Err(format!(
                    "Short copy publishing staged file {} -> {}: copied {} expected {}",
                    staged_path.display(),
                    final_path.display(),
                    copied,
                    expected_len
                ));
            }
            fs::remove_file(staged_path).map_err(|err| {
                format!("Failed to remove staged {}: {err}", staged_path.display())
            })?;
            Ok(())
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[rwkvasr] error: {err}");
        std::process::exit(1);
    }
}

fn new_progress_bar(total_shards: usize) -> Result<ProgressBar, String> {
    let progress = ProgressBar::new(total_shards as u64);
    progress.set_draw_target(ProgressDrawTarget::stderr_with_hz(2));
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {elapsed_precise} [{wide_bar:.cyan/blue}] {pos}/{len} {percent}% eta={eta_precise} {msg}",
        )
        .map_err(|err| format!("Invalid progress template: {err}"))?
        .progress_chars("=>-"),
    );
    progress.enable_steady_tick(Duration::from_millis(1000));
    progress.tick();
    Ok(progress)
}

fn print_progress_line(progress: &ProgressBar, message: String) {
    progress.suspend(|| {
        println!("{message}");
    });
}

fn run() -> Result<(), String> {
    let config = parse_args()?;
    let mut inputs = list_inputs(&config)?;
    if config.max_input_shards > 0 && inputs.len() > config.max_input_shards {
        inputs.truncate(config.max_input_shards);
    }
    if inputs.is_empty() {
        return Err(format!(
            "No input files matching {:?} under {}.",
            config.input_glob,
            config.input_root.display()
        ));
    }
    prepare_output_root(&config)?;

    let threads = if config.max_samples > 0 {
        1
    } else {
        config.threads.min(inputs.len()).max(1)
    };
    println!(
        "[rwkvasr] Rust converter command={:?} input_shards={} threads={} output={}",
        config.command,
        inputs.len(),
        threads,
        config.output_root.display()
    );

    let start = Instant::now();
    let progress = new_progress_bar(inputs.len())?;
    progress.set_message(format!(
        "command={:?} samples=0 skipped=0 converted=0 resumed=0",
        config.command
    ));
    let config = Arc::new(config);
    let inputs = Arc::new(inputs);
    let next_index = Arc::new(AtomicUsize::new(0));
    let global_samples = Arc::new(AtomicUsize::new(0));
    let completed_shards = Arc::new(AtomicUsize::new(0));
    let progress_done = Arc::new(AtomicBool::new(false));
    let heartbeat = {
        let global_samples = Arc::clone(&global_samples);
        let completed_shards = Arc::clone(&completed_shards);
        let progress_done = Arc::clone(&progress_done);
        let progress = progress.clone();
        let total_shards = inputs.len();
        thread::spawn(move || {
            while !progress_done.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(15));
                if progress_done.load(Ordering::Relaxed) {
                    break;
                }
                print_progress_line(
                    &progress,
                    format!(
                        "[rwkvasr] Convert heartbeat: completed={}/{} samples={} elapsed={:.1}s",
                        completed_shards.load(Ordering::Relaxed),
                        total_shards,
                        global_samples.load(Ordering::Relaxed),
                        start.elapsed().as_secs_f32(),
                    ),
                );
            }
        })
    };
    let (tx, rx) = mpsc::channel::<WorkerMessage>();
    let mut handles = Vec::with_capacity(threads);

    for _ in 0..threads {
        let config = Arc::clone(&config);
        let inputs = Arc::clone(&inputs);
        let next_index = Arc::clone(&next_index);
        let global_samples = Arc::clone(&global_samples);
        let progress = progress.clone();
        let tx = tx.clone();
        handles.push(thread::spawn(move || loop {
            if config.max_samples > 0
                && global_samples.load(Ordering::Relaxed) >= config.max_samples
            {
                break;
            }
            let input_index = next_index.fetch_add(1, Ordering::Relaxed);
            if input_index >= inputs.len() {
                break;
            }
            let input_path = &inputs[input_index];
            let input_name = input_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("<unknown>")
                .to_string();
            progress.set_message(format!(
                "checking {}/{} {} total_samples={}",
                input_index + 1,
                inputs.len(),
                input_name,
                global_samples.load(Ordering::Relaxed)
            ));
            match maybe_resume_input(input_index, input_path, &config) {
                Ok(Some(stats)) => {
                    global_samples.fetch_add(stats.samples, Ordering::Relaxed);
                    progress.set_message(format!(
                        "resume skip {}/{} {} samples={} total_samples={}",
                        input_index + 1,
                        inputs.len(),
                        input_name,
                        stats.samples,
                        global_samples.load(Ordering::Relaxed)
                    ));
                    if tx
                        .send(WorkerMessage::Done {
                            input_index,
                            input_name,
                            stats,
                        })
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
                Ok(None) => {}
                Err(err) => {
                    let _ = tx.send(WorkerMessage::Err(err));
                    break;
                }
            }
            progress.set_message(format!(
                "start {}/{} {} total_samples={}",
                input_index + 1,
                inputs.len(),
                input_name,
                global_samples.load(Ordering::Relaxed)
            ));
            let result = match config.command {
                Command::GigaSpeechParquet => convert_gigaspeech_shard(
                    input_index,
                    input_path,
                    &config,
                    &global_samples,
                    &progress,
                ),
                Command::WenetSpeechLhotse => convert_wenetspeech_shard(
                    input_index,
                    input_path,
                    &config,
                    &global_samples,
                    &progress,
                ),
            };
            let message = match result {
                Ok(stats) => WorkerMessage::Done {
                    input_index,
                    input_name,
                    stats,
                },
                Err(err) => WorkerMessage::Err(err),
            };
            if tx.send(message).is_err() {
                break;
            }
        }));
    }
    drop(tx);

    let mut total = ShardStats::default();
    let mut completed = 0_usize;
    while let Ok(message) = rx.recv() {
        match message {
            WorkerMessage::Done {
                input_index,
                input_name,
                stats,
            } => {
                completed += 1;
                completed_shards.store(completed, Ordering::Relaxed);
                total.input_shards += stats.input_shards;
                total.converted_shards += stats.converted_shards;
                total.resumed_shards += stats.resumed_shards;
                total.samples += stats.samples;
                total.skipped_samples += stats.skipped_samples;
                total.missing_audio += stats.missing_audio;
                total.missing_text += stats.missing_text;
                total.unsupported_audio += stats.unsupported_audio;
                progress.inc(1);
                progress.set_message(format!(
                    "done idx={} current={} samples={} skipped={} converted={} resumed={}",
                    input_index,
                    input_name,
                    total.samples,
                    total.skipped_samples,
                    total.converted_shards,
                    total.resumed_shards,
                ));
                if completed == 1 || completed % 10 == 0 || completed == inputs.len() {
                    print_progress_line(&progress, format!(
                        "[rwkvasr] Convert progress: completed={}/{} samples={} skipped={} elapsed={:.1}s",
                    completed,
                    inputs.len(),
                    total.samples,
                    total.skipped_samples,
                    start.elapsed().as_secs_f32(),
                    ));
                }
            }
            WorkerMessage::Err(err) => {
                progress_done.store(true, Ordering::Relaxed);
                let _ = heartbeat.join();
                progress.abandon_with_message(format!("failed: {err}"));
                for handle in handles {
                    let _ = handle.join();
                }
                return Err(err);
            }
        }
    }

    for handle in handles {
        handle
            .join()
            .map_err(|_| String::from("A converter worker thread panicked."))?;
    }
    progress_done.store(true, Ordering::Relaxed);
    heartbeat
        .join()
        .map_err(|_| String::from("Progress heartbeat thread panicked."))?;
    write_conversion_summary(&config, &total, start.elapsed().as_secs_f32())?;
    progress.finish_with_message(format!(
        "done samples={} skipped={} converted={} resumed={} elapsed={:.1}s",
        total.samples,
        total.skipped_samples,
        total.converted_shards,
        total.resumed_shards,
        start.elapsed().as_secs_f32(),
    ));
    println!(
        "convert_asr_corpus samples={} skipped={} output={}",
        total.samples,
        total.skipped_samples,
        config.output_root.display()
    );
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut args = env::args().skip(1);
    let command = match args.next().as_deref() {
        Some("gigaspeech-parquet") => Command::GigaSpeechParquet,
        Some("wenetspeech-lhotse") => Command::WenetSpeechLhotse,
        Some("--help") | Some("-h") | None => {
            print_help();
            std::process::exit(0);
        }
        Some(other) => return Err(format!("Unknown command {other:?}. Use --help.")),
    };

    let mut input_root: Option<PathBuf> = None;
    let mut output_root: Option<PathBuf> = None;
    let mut staging_root: Option<PathBuf> = None;
    let mut staging_max_bytes: Option<u64> = None;
    let mut language: Option<String> = None;
    let mut shard_prefix: Option<String> = None;
    let mut input_glob = match command {
        Command::GigaSpeechParquet => String::from("*.parquet"),
        Command::WenetSpeechLhotse => String::from("cuts_*.jsonl.gz"),
    };
    let mut samples_per_shard = 5_000_usize;
    let mut max_input_shards = 0_usize;
    let mut max_samples = 0_usize;
    let mut progress_every = 10_000_usize;
    let mut parquet_batch_size = 128_usize;
    let mut threads = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4);
    let mut skip_missing = false;
    let mut overwrite = false;
    let mut resume = false;
    let mut adopt_existing = false;
    let mut text_normalization = TextNormalization::None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input-root" => input_root = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--input-glob" => input_glob = next_value(&mut args, &arg)?,
            "--output-root" => output_root = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--staging-root" => staging_root = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--staging-max-bytes" => staging_max_bytes = Some(parse_u64(&mut args, &arg)?),
            "--language" => language = Some(next_value(&mut args, &arg)?),
            "--shard-prefix" => shard_prefix = Some(next_value(&mut args, &arg)?),
            "--samples-per-shard" => samples_per_shard = parse_usize(&mut args, &arg)?,
            "--max-input-shards" => max_input_shards = parse_usize(&mut args, &arg)?,
            "--max-samples" => max_samples = parse_usize(&mut args, &arg)?,
            "--progress-every" => progress_every = parse_usize(&mut args, &arg)?,
            "--parquet-batch-size" => parquet_batch_size = parse_usize(&mut args, &arg)?,
            "--threads" => threads = parse_usize(&mut args, &arg)?,
            "--skip-missing" => skip_missing = true,
            "--overwrite" => overwrite = true,
            "--resume" => resume = true,
            "--adopt-existing" => adopt_existing = true,
            "--text-normalization" => {
                text_normalization = parse_text_normalization(&next_value(&mut args, &arg)?)?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument {other:?}. Use --help.")),
        }
    }

    let input_root = input_root.ok_or_else(|| String::from("--input-root is required."))?;
    let output_root = output_root.ok_or_else(|| String::from("--output-root is required."))?;
    let language = language.ok_or_else(|| String::from("--language is required."))?;
    let shard_prefix = shard_prefix.ok_or_else(|| String::from("--shard-prefix is required."))?;
    if samples_per_shard == 0 {
        return Err(String::from("--samples-per-shard must be positive."));
    }
    if parquet_batch_size == 0 {
        return Err(String::from("--parquet-batch-size must be positive."));
    }
    if threads == 0 {
        return Err(String::from("--threads must be positive."));
    }
    if overwrite && resume {
        return Err(String::from(
            "--overwrite and --resume are mutually exclusive; use --overwrite for a clean rebuild or --resume to continue existing output.",
        ));
    }
    if adopt_existing && !resume {
        return Err(String::from("--adopt-existing requires --resume."));
    }
    if let Some(staging_root) = &staging_root {
        if staging_root == &output_root {
            return Err(String::from(
                "--staging-root must be different from --output-root.",
            ));
        }
    }

    Ok(Config {
        command,
        input_root,
        input_glob,
        output_root,
        staging_root,
        staging_max_bytes,
        language,
        shard_prefix,
        samples_per_shard,
        max_input_shards,
        max_samples,
        progress_every,
        parquet_batch_size,
        threads,
        skip_missing,
        overwrite,
        resume,
        adopt_existing,
        text_normalization,
    })
}

fn print_help() {
    println!("Convert ASR corpora into RWKV-ASR canonical WebDataset tar shards.");
    println!();
    println!("Commands:");
    println!("  gigaspeech-parquet");
    println!("  wenetspeech-lhotse");
    println!();
    println!("Required:");
    println!("  --input-root PATH");
    println!("  --output-root PATH");
    println!("  --language LANG");
    println!("  --shard-prefix PREFIX");
    println!();
    println!("Common optional:");
    println!("  --input-glob GLOB");
    println!("  --samples-per-shard INT     default: 5000");
    println!("  --max-input-shards INT      default: 0 (no cap)");
    println!("  --max-samples INT           default: 0 (no cap; forces threads=1)");
    println!("  --progress-every INT        default: 10000");
    println!("  --threads INT               default: available_parallelism()");
    println!("  --staging-root PATH         optional fast local scratch for output tar parts");
    println!("  --staging-max-bytes INT     optional scratch usage guard");
    println!("  --skip-missing");
    println!("  --overwrite");
    println!("  --resume                    skip completed input shards using done markers");
    println!(
        "  --adopt-existing            with --resume, verify and adopt old per-shard tar output"
    );
    println!("  --text-normalization MODE   none|runtime, default: none");
    println!();
    println!("GigaSpeech optional:");
    println!("  --parquet-batch-size INT    default: 128");
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("Missing value for {flag}."))
}

fn parse_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    next_value(args, flag)?
        .parse::<usize>()
        .map_err(|err| format!("Invalid {flag}: {err}"))
}

fn parse_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, String> {
    next_value(args, flag)?
        .parse::<u64>()
        .map_err(|err| format!("Invalid {flag}: {err}"))
}

fn parse_text_normalization(value: &str) -> Result<TextNormalization, String> {
    match value.to_ascii_lowercase().as_str() {
        "none" => Ok(TextNormalization::None),
        "runtime" => Ok(TextNormalization::Runtime),
        other => Err(format!(
            "Invalid --text-normalization {other:?}; expected none or runtime."
        )),
    }
}

fn normalize_transcript_text(text: &str, language: &str, mode: TextNormalization) -> String {
    match mode {
        TextNormalization::None => text.to_string(),
        TextNormalization::Runtime => normalize_runtime_text(text, language),
    }
}

fn normalize_runtime_text(text: &str, language: &str) -> String {
    let mut normalized = replace_markup_tags(text);
    if language == "en" || language.starts_with("en-") || language.starts_with("en_") {
        normalized = lowercase_outside_markup(&normalized);
    }
    cleanup_punctuation_spacing(&normalized)
}

fn lowercase_outside_markup(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0_usize;
    while index < chars.len() {
        if chars[index] != '<' {
            output.push(chars[index].to_ascii_lowercase());
            index += 1;
            continue;
        }
        let mut end = index + 1;
        while end < chars.len() && chars[end] != '>' {
            end += 1;
        }
        if end >= chars.len() {
            output.push(chars[index].to_ascii_lowercase());
            index += 1;
            continue;
        }
        for ch in &chars[index..=end] {
            output.push(*ch);
        }
        index = end + 1;
    }
    output
}

fn replace_markup_tags(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0_usize;
    while index < chars.len() {
        if chars[index] != '<' {
            output.push(chars[index]);
            index += 1;
            continue;
        }
        let mut end = index + 1;
        while end < chars.len() && chars[end] != '>' {
            end += 1;
        }
        if end >= chars.len() {
            output.push(chars[index]);
            index += 1;
            continue;
        }
        let raw_tag: String = chars[index + 1..end].iter().collect();
        let tag = raw_tag
            .chars()
            .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
            .collect::<String>()
            .to_ascii_uppercase();
        match tag.as_str() {
            "COMMA" => output.push(','),
            "PERIOD" | "FULLSTOP" | "DOT" => output.push('.'),
            "QUESTION" | "QUESTIONMARK" => output.push('?'),
            "EXCLAMATION" | "EXCLAMATIONMARK" | "EXCLAMATIONPOINT" => output.push('!'),
            "COLON" => output.push(':'),
            "SEMICOLON" => output.push(';'),
            "DASH" | "HYPHEN" => output.push('-'),
            "APOSTROPHE" => output.push('\''),
            _ => output.push(' '),
        }
        index = end + 1;
    }
    output
}

fn cleanup_punctuation_spacing(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if matches!(ch, ',' | '.' | '?' | '!' | ':' | ';') {
            while output.ends_with(' ') {
                output.pop();
            }
            output.push(ch);
            output.push(' ');
            pending_space = false;
            continue;
        }
        let previous = output.chars().next_back();
        if pending_space
            && !output.is_empty()
            && !output.ends_with(' ')
            && !(previous.is_some_and(is_cjk_unified) && is_cjk_unified(ch))
        {
            output.push(' ');
        }
        output.push(ch);
        pending_space = false;
    }
    output.trim().to_string()
}

fn is_cjk_unified(ch: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&ch)
}

fn list_inputs(config: &Config) -> Result<Vec<PathBuf>, String> {
    let mut inputs = Vec::new();
    for entry in fs::read_dir(&config.input_root).map_err(|err| {
        format!(
            "Failed to read input root {}: {err}",
            config.input_root.display()
        )
    })? {
        let entry = entry.map_err(|err| format!("Failed to read input entry: {err}"))?;
        let path = entry.path();
        if path.is_file() && matches_glob_name(&path, &config.input_glob) {
            inputs.push(path);
        }
    }
    inputs.sort();
    Ok(inputs)
}

fn matches_glob_name(path: &Path, pattern: &str) -> bool {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    if pattern == "*" {
        return true;
    }
    if let Some((prefix, suffix)) = pattern.split_once('*') {
        return name.starts_with(prefix) && name.ends_with(suffix);
    }
    name == pattern
}

fn prepare_output_root(config: &Config) -> Result<(), String> {
    fs::create_dir_all(&config.output_root).map_err(|err| {
        format!(
            "Failed to create output root {}: {err}",
            config.output_root.display()
        )
    })?;
    let prefix = format!("{}-", config.shard_prefix);
    let mut existing = Vec::new();
    for entry in fs::read_dir(&config.output_root).map_err(|err| {
        format!(
            "Failed to inspect output root {}: {err}",
            config.output_root.display()
        )
    })? {
        let path = entry
            .map_err(|err| format!("Failed to inspect output entry: {err}"))?
            .path();
        if path.extension().and_then(|value| value.to_str()) == Some("tar")
            && path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|name| name.starts_with(&prefix))
                .unwrap_or(false)
        {
            existing.push(path);
        }
    }
    if !existing.is_empty() && !config.overwrite && !config.resume {
        return Err(format!(
            "{} already contains {}*.tar; pass --overwrite to replace them or --resume to continue.",
            config.output_root.display(),
            config.shard_prefix
        ));
    }
    if config.overwrite {
        for path in existing {
            fs::remove_file(&path)
                .map_err(|err| format!("Failed to remove {}: {err}", path.display()))?;
        }
        let state_dir = conversion_state_dir(config);
        if state_dir.exists() {
            fs::remove_dir_all(&state_dir)
                .map_err(|err| format!("Failed to remove {}: {err}", state_dir.display()))?;
        }
    }
    Ok(())
}

fn maybe_resume_input(
    input_index: usize,
    input_path: &Path,
    config: &Config,
) -> Result<Option<ShardStats>, String> {
    if !config.resume {
        return Ok(None);
    }
    if let Some(stats) = load_done_marker(input_index, input_path, config)? {
        return Ok(Some(stats));
    }
    if config.adopt_existing {
        if let Some(stats) = try_adopt_existing_output(input_index, input_path, config)? {
            return Ok(Some(stats));
        }
    }
    cleanup_output_for_input(input_index, config)?;
    Ok(None)
}

fn conversion_state_dir(config: &Config) -> PathBuf {
    config.output_root.join("_conversion_state")
}

fn marker_path(input_index: usize, config: &Config) -> PathBuf {
    conversion_state_dir(config).join(format!("{}-{:06}.json", config.shard_prefix, input_index))
}

fn command_name(command: Command) -> &'static str {
    match command {
        Command::GigaSpeechParquet => "gigaspeech-parquet",
        Command::WenetSpeechLhotse => "wenetspeech-lhotse",
    }
}

fn input_signature(path: &Path) -> Result<(u64, u64), String> {
    let metadata =
        fs::metadata(path).map_err(|err| format!("Failed to stat {}: {err}", path.display()))?;
    let size = metadata.len();
    let mtime = metadata
        .modified()
        .map_err(|err| format!("Failed to get mtime for {}: {err}", path.display()))?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| format!("mtime before unix epoch for {}: {err}", path.display()))?
        .as_secs();
    Ok((size, mtime))
}

fn load_done_marker(
    input_index: usize,
    input_path: &Path,
    config: &Config,
) -> Result<Option<ShardStats>, String> {
    let path = marker_path(input_index, config);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)
        .map_err(|err| format!("Failed to read marker {}: {err}", path.display()))?;
    let marker: DoneMarker = serde_json::from_slice(&bytes)
        .map_err(|err| format!("Invalid marker {}: {err}", path.display()))?;
    let (size, mtime) = input_signature(input_path)?;
    let input_name = input_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if marker.command != command_name(config.command)
        || marker.input_index != input_index
        || marker.input_name != input_name
        || marker.input_size != size
        || marker.input_mtime_unix != mtime
        || marker.shard_prefix != config.shard_prefix
        || marker
            .text_normalization
            .as_deref()
            .unwrap_or(TextNormalization::None.as_str())
            != config.text_normalization.as_str()
    {
        return Ok(None);
    }
    for output_file in &marker.output_files {
        let path = config.output_root.join(output_file);
        let Ok(metadata) = fs::metadata(&path) else {
            return Ok(None);
        };
        if !metadata.is_file() || metadata.len() == 0 {
            return Ok(None);
        }
    }
    let mut stats = marker.stats;
    stats.resumed_shards = 1;
    stats.converted_shards = 0;
    Ok(Some(stats))
}

fn write_done_marker(
    input_index: usize,
    input_path: &Path,
    config: &Config,
    output_files: &[String],
    stats: &ShardStats,
) -> Result<(), String> {
    let state_dir = conversion_state_dir(config);
    fs::create_dir_all(&state_dir)
        .map_err(|err| format!("Failed to create marker dir {}: {err}", state_dir.display()))?;
    let (size, mtime) = input_signature(input_path)?;
    let input_name = input_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_string();
    let marker = DoneMarker {
        version: 1,
        command: command_name(config.command).to_string(),
        input_index,
        input_name,
        input_size: size,
        input_mtime_unix: mtime,
        shard_prefix: config.shard_prefix.clone(),
        text_normalization: Some(config.text_normalization.as_str().to_string()),
        output_files: output_files.to_vec(),
        stats: stats.clone(),
    };
    let marker_tmp = marker_path(input_index, config).with_extension("json.tmp");
    let marker_final = marker_path(input_index, config);
    fs::write(
        &marker_tmp,
        serde_json::to_vec_pretty(&marker)
            .map_err(|err| format!("Failed to serialize marker: {err}"))?,
    )
    .map_err(|err| format!("Failed to write marker {}: {err}", marker_tmp.display()))?;
    fs::rename(&marker_tmp, &marker_final)
        .map_err(|err| format!("Failed to publish marker {}: {err}", marker_final.display()))?;
    Ok(())
}

fn cleanup_output_for_input(input_index: usize, config: &Config) -> Result<(), String> {
    let prefix = format!("{}-{:06}-", config.shard_prefix, input_index);
    if config.output_root.exists() {
        for entry in fs::read_dir(&config.output_root).map_err(|err| {
            format!(
                "Failed to scan output root {}: {err}",
                config.output_root.display()
            )
        })? {
            let path = entry
                .map_err(|err| format!("Failed to read output entry: {err}"))?
                .path();
            let is_target = path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|name| {
                    name.starts_with(&prefix)
                        && (name.ends_with(".tar") || name.ends_with(".tar.part"))
                })
                .unwrap_or(false);
            if is_target {
                fs::remove_file(&path)
                    .map_err(|err| format!("Failed to remove {}: {err}", path.display()))?;
            }
        }
    }
    if let Some(staging_root) = &config.staging_root {
        let staging_dir = staging_root.join(&config.shard_prefix);
        if staging_dir.exists() {
            for entry in fs::read_dir(&staging_dir).map_err(|err| {
                format!(
                    "Failed to scan staging dir {}: {err}",
                    staging_dir.display()
                )
            })? {
                let path = entry
                    .map_err(|err| format!("Failed to read staging entry: {err}"))?
                    .path();
                let is_target = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .map(|name| name.starts_with(&prefix) && name.ends_with(".tar.part"))
                    .unwrap_or(false);
                if is_target {
                    fs::remove_file(&path)
                        .map_err(|err| format!("Failed to remove {}: {err}", path.display()))?;
                }
            }
        }
    }
    let marker = marker_path(input_index, config);
    if marker.exists() {
        fs::remove_file(&marker)
            .map_err(|err| format!("Failed to remove {}: {err}", marker.display()))?;
    }
    Ok(())
}

fn try_adopt_existing_output(
    input_index: usize,
    input_path: &Path,
    config: &Config,
) -> Result<Option<ShardStats>, String> {
    let output_files = existing_output_files_for_input(input_index, config)?;
    if output_files.is_empty() {
        return Ok(None);
    }
    let expected = expected_input_sample_count(input_path, config)?;
    if !validate_output_files_for_samples(
        input_index,
        &output_files,
        expected,
        config,
        "adopt-existing",
    )? {
        return Ok(None);
    }
    let stats = ShardStats {
        input_shards: 1,
        converted_shards: 0,
        resumed_shards: 1,
        samples: expected,
        ..ShardStats::default()
    };
    write_done_marker(input_index, input_path, config, &output_files, &stats)?;
    Ok(Some(stats))
}

fn existing_output_files_for_input(
    input_index: usize,
    config: &Config,
) -> Result<Vec<String>, String> {
    let prefix = format!("{}-{:06}-", config.shard_prefix, input_index);
    let mut output_files = Vec::new();
    if !config.output_root.exists() {
        return Ok(output_files);
    }
    for entry in fs::read_dir(&config.output_root).map_err(|err| {
        format!(
            "Failed to scan output root {}: {err}",
            config.output_root.display()
        )
    })? {
        let path = entry
            .map_err(|err| format!("Failed to read output entry: {err}"))?
            .path();
        if path.extension().and_then(|value| value.to_str()) != Some("tar") {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
            if name.starts_with(&prefix) {
                output_files.push(name.to_string());
            }
        }
    }
    output_files.sort();
    Ok(output_files)
}

fn validate_output_files_for_samples(
    input_index: usize,
    output_files: &[String],
    expected_samples: usize,
    config: &Config,
    context: &str,
) -> Result<bool, String> {
    let expected_parts = expected_samples.div_ceil(config.samples_per_shard);
    if expected_parts == 0 || output_files.len() != expected_parts {
        return Ok(false);
    }
    for output_file in output_files {
        let path = config.output_root.join(output_file);
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(err) => {
                eprintln!(
                    "[rwkvasr] Warning: invalid {context} output for input idx={input_index}; rebuilding shard: failed to stat {}: {err}",
                    path.display(),
                );
                cleanup_output_for_input(input_index, config)?;
                return Ok(false);
            }
        };
        if !metadata.is_file() || metadata.len() == 0 {
            eprintln!(
                "[rwkvasr] Warning: invalid {context} output for input idx={input_index}; rebuilding shard: empty or non-file {}",
                path.display(),
            );
            cleanup_output_for_input(input_index, config)?;
            return Ok(false);
        }
    }
    let expected_last = match expected_samples % config.samples_per_shard {
        0 => config.samples_per_shard,
        value => value,
    };
    let Some(last_file) = output_files.last() else {
        return Ok(false);
    };
    let observed_last = match count_json_samples_in_outputs(std::slice::from_ref(last_file), config)
    {
        Ok(count) => count,
        Err(err) => {
            eprintln!(
                "[rwkvasr] Warning: invalid {context} output for input idx={input_index}; rebuilding shard: {err}",
            );
            cleanup_output_for_input(input_index, config)?;
            return Ok(false);
        }
    };
    if observed_last != expected_last {
        eprintln!(
            "[rwkvasr] Warning: invalid {context} output for input idx={input_index}; rebuilding shard: final part has {observed_last} json entries, expected {expected_last}",
        );
        cleanup_output_for_input(input_index, config)?;
        return Ok(false);
    }
    Ok(true)
}

fn count_json_samples_in_outputs(
    output_files: &[String],
    config: &Config,
) -> Result<usize, String> {
    let mut count = 0_usize;
    for output_file in output_files {
        let path = config.output_root.join(output_file);
        let file =
            File::open(&path).map_err(|err| format!("Failed to open {}: {err}", path.display()))?;
        let mut archive = Archive::new(file);
        let entries = archive
            .entries()
            .map_err(|err| format!("Failed to read tar entries {}: {err}", path.display()))?;
        for entry in entries {
            let entry =
                entry.map_err(|err| format!("Invalid tar entry in {}: {err}", path.display()))?;
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let member = entry
                .path()
                .map_err(|err| format!("Invalid tar member path in {}: {err}", path.display()))?;
            if member.extension().and_then(|value| value.to_str()) == Some("json") {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn expected_input_sample_count(input_path: &Path, config: &Config) -> Result<usize, String> {
    match config.command {
        Command::GigaSpeechParquet => {
            let file = File::open(input_path).map_err(|err| {
                format!(
                    "Failed to open parquet shard {}: {err}",
                    input_path.display()
                )
            })?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|err| {
                format!(
                    "Failed to read parquet metadata {}: {err}",
                    input_path.display()
                )
            })?;
            Ok(builder.metadata().file_metadata().num_rows() as usize)
        }
        Command::WenetSpeechLhotse => {
            let file = File::open(input_path).map_err(|err| {
                format!(
                    "Failed to open WenetSpeech jsonl {}: {err}",
                    input_path.display()
                )
            })?;
            let decoder = GzDecoder::new(file);
            let reader = BufReader::new(decoder);
            let mut count = 0_usize;
            for line in reader.lines() {
                let line =
                    line.map_err(|err| format!("Failed reading {}: {err}", input_path.display()))?;
                if !line.trim().is_empty() {
                    count += 1;
                }
            }
            Ok(count)
        }
    }
}

fn convert_gigaspeech_shard(
    input_index: usize,
    parquet_path: &Path,
    config: &Config,
    global_samples: &AtomicUsize,
    progress: &ProgressBar,
) -> Result<ShardStats, String> {
    let file = File::open(parquet_path).map_err(|err| {
        format!(
            "Failed to open parquet shard {}: {err}",
            parquet_path.display()
        )
    })?;
    let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
        Ok(builder) => builder,
        Err(err) => {
            if config.skip_missing {
                eprintln!(
                    "[rwkvasr] Skipping unreadable parquet shard {}: {err}",
                    parquet_path.display()
                );
                return Ok(ShardStats {
                    input_shards: 1,
                    ..ShardStats::default()
                });
            }
            return Err(format!(
                "Failed to read parquet schema {}: {err}",
                parquet_path.display()
            ));
        }
    };
    let mut reader = builder
        .with_batch_size(config.parquet_batch_size)
        .build()
        .map_err(|err| {
            format!(
                "Failed to build parquet reader {}: {err}",
                parquet_path.display()
            )
        })?;
    let mut writer = ShardedTarWriter::new(
        config.output_root.clone(),
        config.staging_root.clone(),
        config.staging_max_bytes,
        config.shard_prefix.clone(),
        input_index,
        config.samples_per_shard,
    );
    let mut stats = ShardStats {
        input_shards: 1,
        converted_shards: 1,
        ..ShardStats::default()
    };
    let mut local_row_index = 0_usize;
    let source_split = parquet_path
        .file_name()
        .and_then(|value| value.to_str())
        .and_then(|name| name.split_once('-').map(|(prefix, _)| prefix.to_string()))
        .unwrap_or_else(|| String::from("unknown"));

    while let Some(batch) = reader.next() {
        let batch = batch.map_err(|err| {
            format!(
                "Failed reading parquet batch {}: {err}",
                parquet_path.display()
            )
        })?;
        for row in 0..batch.num_rows() {
            if config.max_samples > 0
                && global_samples.load(Ordering::Relaxed) >= config.max_samples
            {
                writer.finish()?;
                return Ok(stats);
            }
            local_row_index += 1;
            let Some(text) = first_string(&batch, TEXT_KEYS, row) else {
                stats.missing_text += 1;
                stats.skipped_samples += 1;
                continue;
            };
            let Some((audio_bytes, audio_path, sample_rate)) =
                parquet_audio(&batch, row, &config.input_root, parquet_path)?
            else {
                stats.missing_audio += 1;
                stats.skipped_samples += 1;
                continue;
            };
            let Some(audio_suffix) = infer_audio_suffix(&audio_bytes, audio_path.as_deref()) else {
                stats.unsupported_audio += 1;
                stats.skipped_samples += 1;
                continue;
            };
            let mut duration = first_number(&batch, &["duration"], row);
            if duration.is_none() {
                if let (Some(begin), Some(end)) = (
                    first_number(&batch, &["begin_time", "start_time", "start"], row),
                    first_number(&batch, &["end_time", "end"], row),
                ) {
                    duration = Some(end - begin);
                }
            }
            let wav_info = if sample_rate.is_none() || duration.is_none() {
                infer_wav_duration(&audio_bytes)
            } else {
                None
            };
            let (duration, sample_rate) = match duration {
                Some(value) if value > 0.0 => {
                    let rate = sample_rate.or_else(|| wav_info.map(|(_, rate)| rate));
                    (value, rate)
                }
                _ => match wav_info {
                    Some((duration, rate)) => (duration, sample_rate.or(Some(rate))),
                    None => {
                        stats.skipped_samples += 1;
                        continue;
                    }
                },
            };
            let raw_id = first_string(&batch, ID_KEYS, row)
                .unwrap_or_else(|| format!("{}_{}", parquet_stem(parquet_path), local_row_index));
            let sample_id = safe_key(
                &raw_id,
                &format!("{}_{}", parquet_stem(parquet_path), local_row_index),
            );
            let key = safe_key(
                &format!("gigaspeech_{sample_id}"),
                &format!(
                    "gigaspeech_{}_{}",
                    parquet_stem(parquet_path),
                    local_row_index
                ),
            );
            reserve_sample_slot(config, global_samples)?;
            let language = normalise_language(
                first_string(&batch, &["language", "lang"], row),
                &config.language,
            );
            let normalized_text =
                normalize_transcript_text(&text, &language, config.text_normalization);
            let mut metadata = json!({
                "id": key,
                "sid": key,
                "text": normalized_text,
                "language": language,
                "duration": duration,
                "sample_rate": sample_rate,
                "source": "gigaspeech",
                "source_split": source_split,
                "source_shard": parquet_path.file_name().and_then(|value| value.to_str()).unwrap_or(""),
                "source_id": raw_id,
                "text_normalization": config.text_normalization.as_str(),
            });
            if metadata["text"].as_str() != Some(text.as_str()) {
                metadata["source_text"] = json!(text);
            }
            if let Some(audio_path) = audio_path {
                metadata["source_audio_path"] = json!(audio_path);
            }
            writer.add_sample(Sample {
                key,
                audio_suffix,
                audio_bytes,
                metadata,
            })?;
            stats.samples += 1;
            if config.progress_every > 0 && stats.samples % config.progress_every == 0 {
                print_progress_line(progress, format!(
                    "[rwkvasr] GigaSpeech shard progress: idx={} shard={} samples={} skipped={} total_samples={}",
                    input_index,
                    parquet_path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or(""),
                    stats.samples,
                    stats.skipped_samples,
                    global_samples.load(Ordering::Relaxed),
                ));
                progress.set_message(format!(
                    "gigaspeech idx={} {} samples={} skipped={} total_samples={}",
                    input_index,
                    parquet_path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or(""),
                    stats.samples,
                    stats.skipped_samples,
                    global_samples.load(Ordering::Relaxed),
                ));
            }
        }
    }
    writer.finish()?;
    write_done_marker(
        input_index,
        parquet_path,
        config,
        &writer.output_files,
        &stats,
    )?;
    Ok(stats)
}

fn convert_wenetspeech_shard(
    input_index: usize,
    jsonl_path: &Path,
    config: &Config,
    global_samples: &AtomicUsize,
    progress: &ProgressBar,
) -> Result<ShardStats, String> {
    let tar_path = paired_wenet_tar(jsonl_path);
    if !tar_path.exists() {
        if config.skip_missing {
            eprintln!(
                "[rwkvasr] Skipping unpaired WenetSpeech shard {}; missing {}",
                jsonl_path.display(),
                tar_path.display()
            );
            return Ok(ShardStats {
                input_shards: 1,
                ..ShardStats::default()
            });
        }
        return Err(format!(
            "Missing paired tar file for {}: {}",
            jsonl_path.display(),
            tar_path.display()
        ));
    }
    let metadata = load_wenet_metadata(jsonl_path, config)?;
    let file = File::open(&tar_path).map_err(|err| {
        format!(
            "Failed to open WenetSpeech tar {}: {err}",
            tar_path.display()
        )
    })?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let mut writer = ShardedTarWriter::new(
        config.output_root.clone(),
        config.staging_root.clone(),
        config.staging_max_bytes,
        config.shard_prefix.clone(),
        input_index,
        config.samples_per_shard,
    );
    let mut stats = ShardStats {
        input_shards: 1,
        converted_shards: 1,
        ..ShardStats::default()
    };
    let mut matched = 0_usize;

    let entries = archive
        .entries()
        .map_err(|err| format!("Failed reading tar entries {}: {err}", tar_path.display()))?;
    for entry in entries {
        if config.max_samples > 0 && global_samples.load(Ordering::Relaxed) >= config.max_samples {
            writer.finish()?;
            return Ok(stats);
        }
        let mut entry = entry
            .map_err(|err| format!("Failed reading tar entry {}: {err}", tar_path.display()))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry
            .path()
            .map_err(|err| format!("Invalid tar member path {}: {err}", tar_path.display()))?
            .to_path_buf();
        let suffix = path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase());
        let Some(suffix) = suffix else {
            continue;
        };
        if !AUDIO_SUFFIXES.contains(&suffix.as_str()) {
            continue;
        }
        let Some(source_id) = path
            .file_stem()
            .and_then(|value| value.to_str())
            .map(String::from)
        else {
            continue;
        };
        let Some(mut sample_metadata) = metadata.get(&source_id).cloned() else {
            continue;
        };
        let mut audio_bytes = Vec::new();
        entry
            .read_to_end(&mut audio_bytes)
            .map_err(|err| format!("Failed reading audio member {}: {err}", path.display()))?;
        reserve_sample_slot(config, global_samples)?;
        sample_metadata["source_tar"] = json!(tar_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(""));
        sample_metadata["source_audio_member"] = json!(path.to_string_lossy().to_string());
        let key = sample_metadata["id"]
            .as_str()
            .ok_or_else(|| format!("Missing generated id for WenetSpeech cut {source_id}"))?
            .to_string();
        writer.add_sample(Sample {
            key,
            audio_suffix: suffix,
            audio_bytes,
            metadata: sample_metadata,
        })?;
        matched += 1;
        stats.samples += 1;
        if config.progress_every > 0 && stats.samples % config.progress_every == 0 {
            print_progress_line(progress, format!(
                "[rwkvasr] WenetSpeech shard progress: idx={} shard={} samples={} skipped={} total_samples={}",
                input_index,
                jsonl_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(""),
                stats.samples,
                stats.skipped_samples,
                global_samples.load(Ordering::Relaxed),
            ));
            progress.set_message(format!(
                "wenetspeech idx={} {} samples={} skipped={} total_samples={}",
                input_index,
                jsonl_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(""),
                stats.samples,
                stats.skipped_samples,
                global_samples.load(Ordering::Relaxed),
            ));
        }
    }
    if !(config.max_samples > 0 && global_samples.load(Ordering::Relaxed) >= config.max_samples) {
        let missing = metadata.len().saturating_sub(matched);
        stats.missing_audio += missing;
        stats.skipped_samples += missing;
    }
    writer.finish()?;
    write_done_marker(
        input_index,
        jsonl_path,
        config,
        &writer.output_files,
        &stats,
    )?;
    Ok(stats)
}

fn reserve_sample_slot(config: &Config, global_samples: &AtomicUsize) -> Result<(), String> {
    if config.max_samples == 0 {
        global_samples.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }
    loop {
        let current = global_samples.load(Ordering::Relaxed);
        if current >= config.max_samples {
            return Err(String::from("max samples reached"));
        }
        if global_samples
            .compare_exchange(current, current + 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            return Ok(());
        }
    }
}

fn load_wenet_metadata(
    jsonl_path: &Path,
    config: &Config,
) -> Result<HashMap<String, Value>, String> {
    let file = File::open(jsonl_path).map_err(|err| {
        format!(
            "Failed to open WenetSpeech jsonl {}: {err}",
            jsonl_path.display()
        )
    })?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);
    let mut output = HashMap::new();
    for line in reader.lines() {
        let line = line.map_err(|err| format!("Failed reading {}: {err}", jsonl_path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: Value = serde_json::from_str(&line)
            .map_err(|err| format!("Invalid JSON in {}: {err}", jsonl_path.display()))?;
        if let Some((source_id, metadata)) = wenet_metadata_from_value(&raw, jsonl_path, config) {
            output.insert(source_id, metadata);
        }
    }
    Ok(output)
}

fn wenet_metadata_from_value(
    raw: &Value,
    jsonl_path: &Path,
    config: &Config,
) -> Option<(String, Value)> {
    let cut_id = raw.get("id")?.as_str()?.to_string();
    let supervision = raw
        .get("supervisions")
        .and_then(|value| value.as_array())
        .and_then(|items| items.first())
        .cloned()
        .unwrap_or(Value::Null);
    let text = supervision
        .get("text")
        .or_else(|| raw.get("text"))?
        .as_str()?
        .to_string();
    let duration = raw
        .get("duration")
        .or_else(|| supervision.get("duration"))
        .or_else(|| {
            raw.get("recording")
                .and_then(|recording| recording.get("duration"))
        })?
        .as_f64()?;
    if duration <= 0.0 {
        return None;
    }
    let sample_rate = raw
        .get("recording")
        .and_then(|recording| recording.get("sampling_rate"))
        .and_then(|value| value.as_u64());
    let source_audio_path = raw
        .get("recording")
        .and_then(|recording| recording.get("sources"))
        .and_then(|value| value.as_array())
        .and_then(|items| items.first())
        .and_then(|source| source.get("source"))
        .and_then(|value| value.as_str());
    let num_frames = raw
        .get("features")
        .and_then(|features| features.get("num_frames"))
        .and_then(|value| value.as_u64())
        .unwrap_or_else(|| (duration * 100.0).round().max(1.0) as u64);
    let key = safe_key(
        &format!("wenetspeech_{}", safe_key(&cut_id, "wenetspeech_cut")),
        "wenetspeech_cut",
    );
    let language = normalise_language(
        supervision
            .get("language")
            .and_then(|value| value.as_str())
            .map(String::from),
        &config.language,
    );
    let normalized_text = normalize_transcript_text(&text, &language, config.text_normalization);
    let metadata = json!({
        "id": key,
        "sid": key,
        "text": normalized_text,
        "language": language,
        "duration": duration,
        "sample_rate": sample_rate,
        "source": "wenetspeech",
        "source_id": cut_id,
        "source_recording_id": supervision
            .get("recording_id")
            .or_else(|| raw.get("recording").and_then(|recording| recording.get("id")))
            .and_then(|value| value.as_str())
            .unwrap_or(""),
        "source_audio_path": source_audio_path,
        "source_shard": jsonl_path.file_name().and_then(|value| value.to_str()).unwrap_or(""),
        "num_frames": num_frames,
        "text_normalization": config.text_normalization.as_str(),
    });
    let mut metadata = metadata;
    if metadata["text"].as_str() != Some(text.as_str()) {
        metadata["source_text"] = json!(text);
    }
    Some((cut_id, metadata))
}

fn paired_wenet_tar(jsonl_path: &Path) -> PathBuf {
    let Some(name) = jsonl_path.file_name().and_then(|value| value.to_str()) else {
        return jsonl_path.with_extension("tar.gz");
    };
    let tar_name = name.strip_suffix(".jsonl.gz").unwrap_or(name).to_string() + ".tar.gz";
    jsonl_path.with_file_name(tar_name)
}

fn parquet_audio(
    batch: &RecordBatch,
    row: usize,
    input_root: &Path,
    parquet_path: &Path,
) -> Result<Option<(Vec<u8>, Option<String>, Option<u64>)>, String> {
    let mut audio_bytes = None;
    let mut audio_path = None;
    let mut sample_rate =
        first_number(batch, &["sampling_rate", "sample_rate"], row).map(|value| value as u64);
    if let Some(audio_array) = column(batch, "audio") {
        if let Some(struct_array) = audio_array.as_any().downcast_ref::<StructArray>() {
            audio_bytes = struct_binary_value(struct_array, "bytes", row);
            audio_path = struct_string_value(struct_array, "path", row);
            if sample_rate.is_none() {
                sample_rate = struct_number_value(struct_array, "sampling_rate", row)
                    .map(|value| value as u64);
            }
        } else if audio_array.as_any().is::<BinaryArray>()
            || audio_array.as_any().is::<LargeBinaryArray>()
        {
            audio_bytes = binary_value(audio_array, row);
        }
    }
    if audio_bytes.is_none() {
        for name in ["audio_bytes", "wav", "bytes"] {
            if let Some(array) = column(batch, name) {
                audio_bytes = binary_value(array, row);
                if audio_bytes.is_some() {
                    break;
                }
            }
        }
    }
    if audio_path.is_none() {
        audio_path = first_string(batch, &["audio_path", "path"], row);
    }
    if audio_bytes.is_none() {
        if let Some(path) = &audio_path {
            for candidate in [
                PathBuf::from(path),
                parquet_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(path),
                input_root.join(path),
            ] {
                if candidate.is_file() {
                    let bytes = fs::read(&candidate).map_err(|err| {
                        format!(
                            "Failed to read external audio {}: {err}",
                            candidate.display()
                        )
                    })?;
                    audio_bytes = Some(bytes);
                    break;
                }
            }
        }
    }
    Ok(audio_bytes.map(|bytes| (bytes, audio_path, sample_rate)))
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a ArrayRef> {
    let idx = batch.schema().index_of(name).ok()?;
    Some(batch.column(idx))
}

fn first_string(batch: &RecordBatch, names: &[&str], row: usize) -> Option<String> {
    for name in names {
        let Some(array) = column(batch, name) else {
            continue;
        };
        if let Some(value) = string_value(array, row) {
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn first_number(batch: &RecordBatch, names: &[&str], row: usize) -> Option<f64> {
    for name in names {
        let Some(array) = column(batch, name) else {
            continue;
        };
        if let Some(value) = number_value(array, row) {
            return Some(value);
        }
    }
    None
}

fn string_value(array: &ArrayRef, row: usize) -> Option<String> {
    if row >= array.len() || array.is_null(row) {
        return None;
    }
    if let Some(value) = array.as_any().downcast_ref::<StringArray>() {
        return Some(value.value(row).to_string());
    }
    if let Some(value) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Some(value.value(row).to_string());
    }
    None
}

fn binary_value(array: &ArrayRef, row: usize) -> Option<Vec<u8>> {
    if row >= array.len() || array.is_null(row) {
        return None;
    }
    if let Some(value) = array.as_any().downcast_ref::<BinaryArray>() {
        return Some(value.value(row).to_vec());
    }
    if let Some(value) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some(value.value(row).to_vec());
    }
    None
}

fn number_value(array: &ArrayRef, row: usize) -> Option<f64> {
    if row >= array.len() || array.is_null(row) {
        return None;
    }
    if let Some(value) = array.as_any().downcast_ref::<Float64Array>() {
        return Some(value.value(row));
    }
    if let Some(value) = array.as_any().downcast_ref::<Float32Array>() {
        return Some(value.value(row) as f64);
    }
    if let Some(value) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(value.value(row) as f64);
    }
    if let Some(value) = array.as_any().downcast_ref::<Int32Array>() {
        return Some(value.value(row) as f64);
    }
    if let Some(value) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some(value.value(row) as f64);
    }
    if let Some(value) = array.as_any().downcast_ref::<UInt32Array>() {
        return Some(value.value(row) as f64);
    }
    None
}

fn struct_child<'a>(struct_array: &'a StructArray, name: &str) -> Option<&'a ArrayRef> {
    let fields = struct_array.fields();
    let index = fields.iter().position(|field| field.name() == name)?;
    struct_array.columns().get(index)
}

fn struct_string_value(struct_array: &StructArray, name: &str, row: usize) -> Option<String> {
    let array = struct_child(struct_array, name)?;
    string_value(array, row)
}

fn struct_binary_value(struct_array: &StructArray, name: &str, row: usize) -> Option<Vec<u8>> {
    let array = struct_child(struct_array, name)?;
    binary_value(array, row)
}

fn struct_number_value(struct_array: &StructArray, name: &str, row: usize) -> Option<f64> {
    let array = struct_child(struct_array, name)?;
    number_value(array, row)
}

fn infer_audio_suffix(audio: &[u8], source_path: Option<&str>) -> Option<String> {
    if let Some(path) = source_path {
        if let Some(suffix) = Path::new(path)
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
        {
            if AUDIO_SUFFIXES.contains(&suffix.as_str()) {
                return Some(suffix);
            }
        }
    }
    if audio.len() >= 12 && &audio[0..4] == b"RIFF" && &audio[8..12] == b"WAVE" {
        return Some(String::from("wav"));
    }
    if audio.starts_with(b"fLaC") {
        return Some(String::from("flac"));
    }
    if audio.starts_with(b"ID3")
        || audio.starts_with(&[0xff, 0xfb])
        || audio.starts_with(&[0xff, 0xf3])
        || audio.starts_with(&[0xff, 0xf2])
    {
        return Some(String::from("mp3"));
    }
    None
}

fn infer_wav_duration(audio: &[u8]) -> Option<(f64, u64)> {
    if audio.len() < 44 || &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return None;
    }
    let mut offset = 12_usize;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data_size = None;
    while offset + 8 <= audio.len() {
        let chunk_id = &audio[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            audio[offset + 4],
            audio[offset + 5],
            audio[offset + 6],
            audio[offset + 7],
        ]) as usize;
        offset += 8;
        if offset + chunk_size > audio.len() {
            break;
        }
        if chunk_id == b"fmt " && chunk_size >= 16 {
            channels = Some(u16::from_le_bytes([audio[offset + 2], audio[offset + 3]]) as u64);
            sample_rate = Some(u32::from_le_bytes([
                audio[offset + 4],
                audio[offset + 5],
                audio[offset + 6],
                audio[offset + 7],
            ]) as u64);
            bits_per_sample =
                Some(u16::from_le_bytes([audio[offset + 14], audio[offset + 15]]) as u64);
        } else if chunk_id == b"data" {
            data_size = Some(chunk_size as u64);
        }
        offset += chunk_size + (chunk_size % 2);
    }
    let channels = channels?;
    let sample_rate = sample_rate?;
    let bits_per_sample = bits_per_sample?;
    let data_size = data_size?;
    if channels == 0 || sample_rate == 0 || bits_per_sample == 0 {
        return None;
    }
    let bytes_per_sample = bits_per_sample / 8;
    if bytes_per_sample == 0 {
        return None;
    }
    let frames = data_size / (channels * bytes_per_sample);
    Some((frames as f64 / sample_rate as f64, sample_rate))
}

fn safe_key(value: &str, fallback: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.trim().chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '=') {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    let output = output.trim_matches(['.', '_']).to_string();
    let output = if output.is_empty() {
        fallback.to_string()
    } else {
        output
    };
    if output.len() <= 180 {
        output
    } else {
        output[..150].to_string()
    }
}

fn normalise_language(value: Option<String>, default: &str) -> String {
    let raw = value
        .unwrap_or_else(|| default.to_string())
        .trim()
        .to_ascii_lowercase();
    match raw.as_str() {
        "chinese" | "mandarin" | "zh-cn" | "zh_cn" | "cmn" => String::from("zh"),
        "english" | "en-us" | "en_us" => String::from("en"),
        "" => default.to_string(),
        _ => raw,
    }
}

fn parquet_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.to_string())
        .unwrap_or_else(|| String::from("parquet"))
}

fn write_conversion_summary(
    config: &Config,
    stats: &ShardStats,
    elapsed_sec: f32,
) -> Result<(), String> {
    let filename = match config.command {
        Command::GigaSpeechParquet => "gigaspeech_parquet_conversion_summary.json",
        Command::WenetSpeechLhotse => "wenetspeech_lhotse_conversion_summary.json",
    };
    let output_path = config.output_root.join(filename);
    let data = json!({
        "command": format!("{:?}", config.command),
        "input_root": config.input_root,
        "input_glob": config.input_glob,
        "output_root": config.output_root,
        "input_shards": stats.input_shards,
        "converted_shards": stats.converted_shards,
        "resumed_shards": stats.resumed_shards,
        "samples": stats.samples,
        "skipped_samples": stats.skipped_samples,
        "missing_audio": stats.missing_audio,
        "missing_text": stats.missing_text,
        "unsupported_audio": stats.unsupported_audio,
        "text_normalization": config.text_normalization.as_str(),
        "elapsed_sec": elapsed_sec,
    });
    fs::write(
        &output_path,
        serde_json::to_vec_pretty(&data)
            .map_err(|err| format!("Failed to serialize summary: {err}"))?,
    )
    .map_err(|err| format!("Failed to write summary {}: {err}", output_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{normalize_transcript_text, TextNormalization};

    #[test]
    fn runtime_normalizes_gigaspeech_english() {
        let text = "YOU KNOW <COMMA> GET COURT DATES <PERIOD>";
        assert_eq!(
            normalize_transcript_text(text, "en", TextNormalization::Runtime),
            "you know, get court dates."
        );
    }

    #[test]
    fn runtime_drops_non_speech_tags() {
        let text = "HELLO <NOISE> WORLD <QUESTIONMARK>";
        assert_eq!(
            normalize_transcript_text(text, "en", TextNormalization::Runtime),
            "hello world?"
        );
    }

    #[test]
    fn none_preserves_source_text() {
        let text = "YOU KNOW <COMMA>";
        assert_eq!(
            normalize_transcript_text(text, "en", TextNormalization::None),
            text
        );
    }
}
