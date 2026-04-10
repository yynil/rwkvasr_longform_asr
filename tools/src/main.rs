use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

const TRAIN_NAME: &str = "train";
const EVAL_NAME: &str = "eval";
const TAR_BLOCK_SIZE: usize = 512;

#[derive(Clone, Copy, Debug)]
enum SplitBy {
    SampleId,
    ShardName,
}

impl SplitBy {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "sample_id" => Ok(Self::SampleId),
            "shard_name" => Ok(Self::ShardName),
            other => Err(format!("Unsupported --split-by {other:?}; expected sample_id or shard_name.")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SampleId => "sample_id",
            Self::ShardName => "shard_name",
        }
    }
}

#[derive(Clone, Debug)]
struct Config {
    webdataset_root: PathBuf,
    shard_pattern: String,
    output_path: PathBuf,
    summary_path: PathBuf,
    eval_ratio: f64,
    hash_seed: u64,
    utt_id_key: String,
    split_by: SplitBy,
    threads: usize,
}

#[derive(Default)]
struct PartialEntry {
    audio_member: Option<String>,
    audio_format: Option<String>,
    audio_offset: Option<u64>,
    audio_size: Option<u64>,
    json_member: Option<String>,
    json_offset: Option<u64>,
    json_size: Option<u64>,
    utt_id: Option<String>,
    num_frames: Option<u64>,
}

#[derive(Debug)]
struct ShardResult {
    shard_index: usize,
    shard_name: String,
    part_path: PathBuf,
    num_samples: u64,
    train_samples: u64,
    eval_samples: u64,
    min_frames: u64,
    max_frames: u64,
    frame_buckets: BTreeMap<u64, u64>,
}

enum WorkerMessage {
    Done(ShardResult),
    Err(String),
}

struct MetadataInfo {
    utt_id: String,
    num_frames: u64,
}

struct TarHeader {
    member_name: String,
    file_size: u64,
    is_regular_file: bool,
    payload_offset: u64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[rwkvasr] error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args()?;
    let shards = list_shards(&config.webdataset_root, &config.shard_pattern)?;
    if shards.is_empty() {
        return Err(format!(
            "No shard files matching {:?} under {}.",
            config.shard_pattern,
            config.webdataset_root.display()
        ));
    }

    if let Some(parent) = config.output_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create output directory {}: {err}",
                parent.display()
            )
        })?;
    }
    if let Some(parent) = config.summary_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create summary directory {}: {err}",
                parent.display()
            )
        })?;
    }

    let temp_dir = config
        .output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".{}.parts",
            config
                .output_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("webdataset_lengths.jsonl")
        ));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)
            .map_err(|err| format!("Failed to clean stale temp dir {}: {err}", temp_dir.display()))?;
    }
    fs::create_dir_all(&temp_dir)
        .map_err(|err| format!("Failed to create temp dir {}: {err}", temp_dir.display()))?;

    log(&format!(
        "Rust length inspection under {} with {} shard(s), threads={}.",
        config.webdataset_root.display(),
        shards.len(),
        config.threads
    ));

    let start = Instant::now();
    let next_index = Arc::new(AtomicUsize::new(0));
    let shared_shards = Arc::new(shards);
    let (tx, rx) = mpsc::channel::<WorkerMessage>();
    let mut handles = Vec::with_capacity(config.threads);

    for _ in 0..config.threads {
        let tx = tx.clone();
        let next_index = Arc::clone(&next_index);
        let shards = Arc::clone(&shared_shards);
        let temp_dir = temp_dir.clone();
        let config = config.clone();
        handles.push(thread::spawn(move || {
            loop {
                let shard_index = next_index.fetch_add(1, Ordering::Relaxed);
                if shard_index >= shards.len() {
                    break;
                }
                let shard_path = &shards[shard_index];
                let result = inspect_shard(shard_index, shard_path, &temp_dir, &config);
                let message = match result {
                    Ok(summary) => WorkerMessage::Done(summary),
                    Err(err) => WorkerMessage::Err(err),
                };
                if tx.send(message).is_err() {
                    break;
                }
            }
        }));
    }
    drop(tx);

    let mut collected = Vec::with_capacity(shared_shards.len());
    let mut num_samples = 0_u64;
    let mut train_samples = 0_u64;
    let mut eval_samples = 0_u64;
    let mut min_frames = u64::MAX;
    let mut max_frames = 0_u64;
    let mut frame_buckets = BTreeMap::<u64, u64>::new();

    for completed in 0..shared_shards.len() {
        match rx.recv().map_err(|err| format!("Worker channel failed: {err}"))? {
            WorkerMessage::Done(result) => {
                num_samples += result.num_samples;
                train_samples += result.train_samples;
                eval_samples += result.eval_samples;
                if result.num_samples > 0 {
                    min_frames = min_frames.min(result.min_frames);
                    max_frames = max_frames.max(result.max_frames);
                }
                for (bucket, count) in &result.frame_buckets {
                    *frame_buckets.entry(*bucket).or_insert(0) += *count;
                }
                let elapsed = start.elapsed().as_secs_f32();
                log(&format!(
                    "Rust length index progress: shards={}/{}, samples={}, elapsed={elapsed:.1}s, current={}",
                    completed + 1,
                    shared_shards.len(),
                    num_samples,
                    result.shard_name
                ));
                collected.push(result);
            }
            WorkerMessage::Err(err) => {
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
            .map_err(|_| String::from("A worker thread panicked during length inspection."))?;
    }

    collected.sort_by_key(|item| item.shard_index);
    merge_part_files(&config.output_path, &collected)?;
    write_summary(
        &config,
        shared_shards.len(),
        num_samples,
        train_samples,
        eval_samples,
        if min_frames == u64::MAX { 0 } else { min_frames },
        max_frames,
        &frame_buckets,
    )?;
    fs::remove_dir_all(&temp_dir)
        .map_err(|err| format!("Failed to remove temp dir {}: {err}", temp_dir.display()))?;

    println!(
        "inspect_webdataset_lengths samples={} train={} eval={}",
        num_samples, train_samples, eval_samples
    );
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut webdataset_root: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut summary_path: Option<PathBuf> = None;
    let mut eval_ratio = 0.01_f64;
    let mut hash_seed = 0_u64;
    let mut utt_id_key = String::from("sid");
    let mut split_by = SplitBy::ShardName;
    let mut shard_pattern = String::from("*.tar");
    let mut threads = thread::available_parallelism().map(|value| value.get()).unwrap_or(4);

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--webdataset-root" => webdataset_root = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--output-path" => output_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--summary-path" => summary_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--eval-ratio" => {
                eval_ratio = next_value(&mut args, &arg)?
                    .parse::<f64>()
                    .map_err(|err| format!("Invalid --eval-ratio: {err}"))?;
            }
            "--hash-seed" => {
                hash_seed = next_value(&mut args, &arg)?
                    .parse::<u64>()
                    .map_err(|err| format!("Invalid --hash-seed: {err}"))?;
            }
            "--utt-id-key" => utt_id_key = next_value(&mut args, &arg)?,
            "--split-by" => split_by = SplitBy::parse(&next_value(&mut args, &arg)?)?,
            "--shard-pattern" => shard_pattern = next_value(&mut args, &arg)?,
            "--threads" => {
                threads = next_value(&mut args, &arg)?
                    .parse::<usize>()
                    .map_err(|err| format!("Invalid --threads: {err}"))?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument {other:?}. Use --help for usage.")),
        }
    }

    if !(0.0..1.0).contains(&eval_ratio) {
        return Err(String::from("--eval-ratio must be within [0, 1)."));
    }
    if utt_id_key.is_empty() {
        return Err(String::from("--utt-id-key must be non-empty."));
    }
    if threads == 0 {
        return Err(String::from("--threads must be positive."));
    }
    if shard_pattern != "*.tar" {
        return Err(String::from(
            "Only --shard-pattern '*.tar' is supported in the Rust tool for now.",
        ));
    }

    let webdataset_root = webdataset_root.ok_or_else(|| String::from("--webdataset-root is required."))?;
    let output_path = output_path.unwrap_or_else(|| default_length_index_path(&webdataset_root));
    let summary_path = summary_path.unwrap_or_else(|| default_length_summary_path(&output_path));

    Ok(Config {
        webdataset_root,
        shard_pattern,
        output_path,
        summary_path,
        eval_ratio,
        hash_seed,
        utt_id_key,
        split_by,
        threads,
    })
}

fn print_help() {
    println!("Build a per-sample WebDataset length index from tar metadata.");
    println!();
    println!("Required:");
    println!("  --webdataset-root PATH");
    println!();
    println!("Optional:");
    println!("  --output-path PATH");
    println!("  --summary-path PATH");
    println!("  --eval-ratio FLOAT         default: 0.01");
    println!("  --hash-seed INT            default: 0");
    println!("  --utt-id-key KEY           default: sid");
    println!("  --split-by MODE            sample_id | shard_name; default: shard_name");
    println!("  --shard-pattern PATTERN    only '*.tar' is supported; default: *.tar");
    println!("  --threads INT              default: available_parallelism()");
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("Missing value for {flag}."))
}

fn default_length_index_path(root: &Path) -> PathBuf {
    if root.is_dir() {
        root.join("webdataset_lengths.jsonl")
    } else {
        root.with_extension("lengths.jsonl")
    }
}

fn default_length_summary_path(index_path: &Path) -> PathBuf {
    index_path.with_extension("summary.json")
}

fn list_shards(root: &Path, shard_pattern: &str) -> Result<Vec<PathBuf>, String> {
    if shard_pattern != "*.tar" {
        return Err(format!("Unsupported shard pattern {shard_pattern:?}."));
    }
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut shards = Vec::new();
    let entries = fs::read_dir(root)
        .map_err(|err| format!("Failed to read shard root {}: {err}", root.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("Failed to read shard entry: {err}"))?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) == Some("tar") {
            shards.push(path);
        }
    }
    shards.sort();
    Ok(shards)
}

fn inspect_shard(
    shard_index: usize,
    shard_path: &Path,
    temp_dir: &Path,
    config: &Config,
) -> Result<ShardResult, String> {
    let shard_name = shard_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("Invalid shard path {}", shard_path.display()))?
        .to_string();
    let part_path = temp_dir.join(format!("{shard_index:06}.jsonl"));
    let part_file = File::create(&part_path)
        .map_err(|err| format!("Failed to create temp part {}: {err}", part_path.display()))?;
    let mut part_writer = BufWriter::new(part_file);
    let mut reader = BufReader::new(
        File::open(shard_path).map_err(|err| format!("Failed to open {}: {err}", shard_path.display()))?,
    );
    let mut pending = HashMap::<String, PartialEntry>::new();
    let mut header = [0_u8; TAR_BLOCK_SIZE];
    let mut num_samples = 0_u64;
    let mut train_samples = 0_u64;
    let mut eval_samples = 0_u64;
    let mut min_frames = u64::MAX;
    let mut max_frames = 0_u64;
    let mut frame_buckets = BTreeMap::<u64, u64>::new();
    let shard_split = match config.split_by {
        SplitBy::ShardName => assign_split(&shard_name, config.hash_seed, config.eval_ratio),
        SplitBy::SampleId => TRAIN_NAME,
    };

    loop {
        let Some(tar_header) = read_tar_header(&mut reader, &mut header)
            .map_err(|err| format!("Failed while reading {}: {err}", shard_path.display()))?
        else {
            break;
        };

        if !tar_header.is_regular_file {
            skip_entry(&mut reader, tar_header.file_size)
                .map_err(|err| format!("Failed to skip {} member {}: {err}", shard_name, tar_header.member_name))?;
            continue;
        }

        let Some((sample_key, suffix)) = split_member_name(&tar_header.member_name) else {
            skip_entry(&mut reader, tar_header.file_size)
                .map_err(|err| format!("Failed to skip {} member {}: {err}", shard_name, tar_header.member_name))?;
            continue;
        };

        match suffix.as_str() {
            audio_suffix if is_supported_audio_suffix(audio_suffix) => {
                skip_entry(&mut reader, tar_header.file_size)
                    .map_err(|err| format!("Failed to skip audio payload {}:{}: {err}", shard_name, tar_header.member_name))?;
                let entry = pending.entry(sample_key.clone()).or_default();
                entry.audio_member = Some(tar_header.member_name);
                entry.audio_format = Some(audio_suffix.to_string());
                entry.audio_offset = Some(tar_header.payload_offset);
                entry.audio_size = Some(tar_header.file_size);
            }
            "json" => {
                let payload = read_entry_bytes(&mut reader, tar_header.file_size).map_err(|err| {
                    format!(
                        "Failed to read json payload {}:{}: {err}",
                        shard_name, tar_header.member_name
                    )
                })?;
                let json_text = String::from_utf8(payload).map_err(|err| {
                    format!(
                        "Metadata is not valid UTF-8 for {}:{}: {err}",
                        shard_name, tar_header.member_name
                    )
                })?;
                let metadata = parse_metadata(&json_text, &sample_key, &config.utt_id_key).map_err(|err| {
                    format!("Failed to parse metadata {}:{}: {err}", shard_name, tar_header.member_name)
                })?;
                let entry = pending.entry(sample_key.clone()).or_default();
                entry.json_member = Some(tar_header.member_name);
                entry.json_offset = Some(tar_header.payload_offset);
                entry.json_size = Some(tar_header.file_size);
                entry.utt_id = Some(metadata.utt_id);
                entry.num_frames = Some(metadata.num_frames);
            }
            _ => {
                skip_entry(&mut reader, tar_header.file_size)
                    .map_err(|err| format!("Failed to skip {} member {}: {err}", shard_name, tar_header.member_name))?;
            }
        }

        let should_emit = pending
            .get(&sample_key)
            .map(|entry| {
                entry.audio_member.is_some()
                    && entry.audio_format.is_some()
                    && entry.audio_offset.is_some()
                    && entry.audio_size.is_some()
                    && entry.json_member.is_some()
                    && entry.json_offset.is_some()
                    && entry.json_size.is_some()
                    && entry.utt_id.is_some()
                    && entry.num_frames.is_some()
            })
            .unwrap_or(false);
        if !should_emit {
            continue;
        }

        let complete = pending
            .remove(&sample_key)
            .ok_or_else(|| format!("Missing pending sample state for {}:{}.", shard_name, sample_key))?;
        let utt_id = complete
            .utt_id
            .ok_or_else(|| format!("Missing utt_id for {}:{}.", shard_name, sample_key))?;
        let num_frames = complete
            .num_frames
            .ok_or_else(|| format!("Missing num_frames for {}:{}.", shard_name, sample_key))?;
        let split = match config.split_by {
            SplitBy::SampleId => assign_split(&utt_id, config.hash_seed, config.eval_ratio),
            SplitBy::ShardName => shard_split,
        };
        let audio_member = complete
            .audio_member
            .ok_or_else(|| format!("Missing audio member for {}:{}.", shard_name, sample_key))?;
        let audio_format = complete
            .audio_format
            .ok_or_else(|| format!("Missing audio format for {}:{}.", shard_name, sample_key))?;
        let audio_offset = complete
            .audio_offset
            .ok_or_else(|| format!("Missing audio offset for {}:{}.", shard_name, sample_key))?;
        let audio_size = complete
            .audio_size
            .ok_or_else(|| format!("Missing audio size for {}:{}.", shard_name, sample_key))?;
        let json_member = complete
            .json_member
            .ok_or_else(|| format!("Missing json member for {}:{}.", shard_name, sample_key))?;
        let json_offset = complete
            .json_offset
            .ok_or_else(|| format!("Missing json offset for {}:{}.", shard_name, sample_key))?;
        let json_size = complete
            .json_size
            .ok_or_else(|| format!("Missing json size for {}:{}.", shard_name, sample_key))?;
        write_index_line(
            &mut part_writer,
            &shard_name,
            &sample_key,
            &utt_id,
            split,
            num_frames,
            &audio_member,
            &audio_format,
            &json_member,
            audio_offset,
            audio_size,
            json_offset,
            json_size,
        )
        .map_err(|err| format!("Failed writing shard part {}: {err}", part_path.display()))?;

        num_samples += 1;
        if split == TRAIN_NAME {
            train_samples += 1;
        } else {
            eval_samples += 1;
        }
        min_frames = min_frames.min(num_frames);
        max_frames = max_frames.max(num_frames);
        *frame_buckets.entry(num_frames / 80).or_insert(0) += 1;
    }

    part_writer
        .flush()
        .map_err(|err| format!("Failed to flush shard part {}: {err}", part_path.display()))?;
    Ok(ShardResult {
        shard_index,
        shard_name,
        part_path,
        num_samples,
        train_samples,
        eval_samples,
        min_frames: if min_frames == u64::MAX { 0 } else { min_frames },
        max_frames,
        frame_buckets,
    })
}

fn merge_part_files(output_path: &Path, parts: &[ShardResult]) -> Result<(), String> {
    let output_file = File::create(output_path)
        .map_err(|err| format!("Failed to create {}: {err}", output_path.display()))?;
    let mut writer = BufWriter::new(output_file);
    for result in parts {
        let mut input = BufReader::new(
            File::open(&result.part_path)
                .map_err(|err| format!("Failed to open part {}: {err}", result.part_path.display()))?,
        );
        io::copy(&mut input, &mut writer)
            .map_err(|err| format!("Failed to merge part {}: {err}", result.part_path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("Failed to flush {}: {err}", output_path.display()))?;
    Ok(())
}

fn write_summary(
    config: &Config,
    num_shards: usize,
    num_samples: u64,
    train_samples: u64,
    eval_samples: u64,
    min_frames: u64,
    max_frames: u64,
    frame_buckets: &BTreeMap<u64, u64>,
) -> Result<(), String> {
    let file = File::create(&config.summary_path)
        .map_err(|err| format!("Failed to create {}: {err}", config.summary_path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{{").map_err(io_error)?;
    writeln!(writer, "  \"version\": 2,").map_err(io_error)?;
    writeln!(
        writer,
        "  \"root\": {},",
        json_string(&config.webdataset_root.display().to_string())
    )
    .map_err(io_error)?;
    writeln!(
        writer,
        "  \"length_index_path\": {},",
        json_string(&config.output_path.display().to_string())
    )
    .map_err(io_error)?;
    writeln!(writer, "  \"num_shards\": {num_shards},").map_err(io_error)?;
    writeln!(writer, "  \"num_samples\": {num_samples},").map_err(io_error)?;
    writeln!(writer, "  \"min_frames\": {min_frames},").map_err(io_error)?;
    writeln!(writer, "  \"max_frames\": {max_frames},").map_err(io_error)?;
    writeln!(writer, "  \"audio_suffixes\": [\"wav\", \"mp3\", \"flac\"],").map_err(io_error)?;
    writeln!(writer, "  \"split\": {{").map_err(io_error)?;
    writeln!(writer, "    \"type\": \"stable_hash\",").map_err(io_error)?;
    writeln!(writer, "    \"split_by\": {},", json_string(config.split_by.as_str())).map_err(io_error)?;
    writeln!(writer, "    \"train_name\": \"train\",").map_err(io_error)?;
    writeln!(writer, "    \"eval_name\": \"eval\",").map_err(io_error)?;
    writeln!(writer, "    \"eval_ratio\": {},", format_float(config.eval_ratio)).map_err(io_error)?;
    writeln!(writer, "    \"hash_seed\": {},", config.hash_seed).map_err(io_error)?;
    writeln!(writer, "    \"utt_id_key\": {}", json_string(&config.utt_id_key)).map_err(io_error)?;
    writeln!(writer, "  }},").map_err(io_error)?;
    writeln!(writer, "  \"splits\": {{").map_err(io_error)?;
    writeln!(writer, "    \"train\": {{ \"num_samples\": {train_samples} }},").map_err(io_error)?;
    writeln!(writer, "    \"eval\": {{ \"num_samples\": {eval_samples} }}").map_err(io_error)?;
    writeln!(writer, "  }},").map_err(io_error)?;
    writeln!(writer, "  \"frame_buckets\": {{").map_err(io_error)?;
    let mut iter = frame_buckets.iter().peekable();
    while let Some((bucket, count)) = iter.next() {
        let suffix = if iter.peek().is_some() { "," } else { "" };
        writeln!(writer, "    {}: {count}{suffix}", json_string(&bucket.to_string())).map_err(io_error)?;
    }
    writeln!(writer, "  }}").map_err(io_error)?;
    writeln!(writer, "}}").map_err(io_error)?;
    writer
        .flush()
        .map_err(|err| format!("Failed to flush summary {}: {err}", config.summary_path.display()))?;
    Ok(())
}

fn io_error(err: io::Error) -> String {
    err.to_string()
}

fn read_tar_header(reader: &mut BufReader<File>, block: &mut [u8; TAR_BLOCK_SIZE]) -> io::Result<Option<TarHeader>> {
    let mut read = 0;
    while read < TAR_BLOCK_SIZE {
        let count = reader.read(&mut block[read..])?;
        if count == 0 {
            if read == 0 {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected EOF while reading tar header.",
            ));
        }
        read += count;
    }
    if block.iter().all(|value| *value == 0) {
        return Ok(None);
    }
    let name = parse_tar_string(&block[..100]);
    let prefix = parse_tar_string(&block[345..500]);
    let member_name = if prefix.is_empty() {
        name
    } else if name.is_empty() {
        prefix
    } else {
        format!("{prefix}/{name}")
    };
    let file_size = parse_tar_octal(&block[124..136])?;
    let typeflag = block[156];
    let is_regular_file = typeflag == 0 || typeflag == b'0';
    let payload_offset = reader.stream_position()?;
    Ok(Some(TarHeader {
        member_name,
        file_size,
        is_regular_file,
        payload_offset,
    }))
}

fn parse_tar_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|value| *value == 0).unwrap_or(bytes.len());
    let trimmed = bytes[..end]
        .iter()
        .copied()
        .take_while(|value| *value != 0)
        .collect::<Vec<u8>>();
    String::from_utf8_lossy(&trimmed).trim().to_string()
}

fn parse_tar_octal(bytes: &[u8]) -> io::Result<u64> {
    let raw = String::from_utf8_lossy(bytes);
    let trimmed = raw.trim_matches(char::from(0)).trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(trimmed, 8).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

fn split_member_name(member_name: &str) -> Option<(String, String)> {
    let basename = member_name.rsplit('/').next()?;
    let (key, suffix) = basename.rsplit_once('.')?;
    let suffix = suffix.to_ascii_lowercase();
    if suffix != "json" && !is_supported_audio_suffix(&suffix) {
        return None;
    }
    Some((key.to_string(), suffix))
}

fn is_supported_audio_suffix(suffix: &str) -> bool {
    matches!(suffix, "wav" | "mp3" | "flac")
}

fn skip_entry(reader: &mut BufReader<File>, file_size: u64) -> io::Result<()> {
    let padded = padded_tar_size(file_size);
    reader.seek(SeekFrom::Current(padded as i64))?;
    Ok(())
}

fn read_entry_bytes(reader: &mut BufReader<File>, file_size: u64) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0_u8; file_size as usize];
    reader.read_exact(&mut buffer)?;
    let padding = padded_tar_size(file_size) - file_size;
    if padding > 0 {
        reader.seek(SeekFrom::Current(padding as i64))?;
    }
    Ok(buffer)
}

fn padded_tar_size(file_size: u64) -> u64 {
    let remainder = file_size % TAR_BLOCK_SIZE as u64;
    if remainder == 0 {
        file_size
    } else {
        file_size + (TAR_BLOCK_SIZE as u64 - remainder)
    }
}

fn parse_metadata(json: &str, sample_key: &str, utt_id_key: &str) -> Result<MetadataInfo, String> {
    let utt_id = extract_json_string(json, utt_id_key).unwrap_or_else(|| sample_key.to_string());
    let num_frames = infer_num_frames(json)?;
    Ok(MetadataInfo { utt_id, num_frames })
}

fn infer_num_frames(json: &str) -> Result<u64, String> {
    if let Some(num_frames) = extract_json_number(json, "num_frames") {
        let value = num_frames.round() as i64;
        if value > 0 {
            return Ok(value as u64);
        }
    }

    let duration_sec = if let (Some(begin_time), Some(end_time)) = (
        extract_json_number(json, "begin_time"),
        extract_json_number(json, "end_time"),
    ) {
        Some(end_time - begin_time)
    } else if let Some(duration) = extract_json_number(json, "duration") {
        Some(duration)
    } else if let (Some(num_samples), Some(sample_rate)) = (
        extract_json_number(json, "num_samples"),
        extract_json_number(json, "sample_rate"),
    ) {
        if sample_rate <= 0.0 {
            None
        } else {
            Some(num_samples / sample_rate)
        }
    } else {
        None
    };

    let duration_sec = duration_sec.ok_or_else(|| String::from("Unable to infer num_frames from metadata."))?;
    if duration_sec <= 0.0 {
        return Err(String::from("Duration is non-positive."));
    }
    Ok((duration_sec * 100.0).round().max(1.0) as u64)
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let key_pos = find_json_key(json, key)?;
    let bytes = json.as_bytes();
    let colon = find_after_colon(bytes, key_pos + key.len() + 2)?;
    let (value, _) = parse_json_string_value(bytes, colon)?;
    Some(value)
}

fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let key_pos = find_json_key(json, key)?;
    let bytes = json.as_bytes();
    let colon = find_after_colon(bytes, key_pos + key.len() + 2)?;
    let (value, _) = parse_json_number_value(bytes, colon)?;
    Some(value)
}

fn find_json_key(json: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{key}\"");
    json.find(&pattern)
}

fn find_after_colon(bytes: &[u8], mut position: usize) -> Option<usize> {
    while position < bytes.len() && bytes[position].is_ascii_whitespace() {
        position += 1;
    }
    if position >= bytes.len() || bytes[position] != b':' {
        return None;
    }
    position += 1;
    while position < bytes.len() && bytes[position].is_ascii_whitespace() {
        position += 1;
    }
    Some(position)
}

fn parse_json_string_value(bytes: &[u8], mut position: usize) -> Option<(String, usize)> {
    if bytes.get(position)? != &b'"' {
        return None;
    }
    position += 1;
    let mut output = String::new();
    while position < bytes.len() {
        let byte = bytes[position];
        position += 1;
        match byte {
            b'"' => return Some((output, position)),
            b'\\' => {
                let escaped = *bytes.get(position)?;
                position += 1;
                match escaped {
                    b'"' => output.push('"'),
                    b'\\' => output.push('\\'),
                    b'/' => output.push('/'),
                    b'b' => output.push('\u{0008}'),
                    b'f' => output.push('\u{000C}'),
                    b'n' => output.push('\n'),
                    b'r' => output.push('\r'),
                    b't' => output.push('\t'),
                    b'u' => {
                        let hex = std::str::from_utf8(bytes.get(position..position + 4)?).ok()?;
                        let code = u16::from_str_radix(hex, 16).ok()?;
                        let ch = char::from_u32(code as u32)?;
                        output.push(ch);
                        position += 4;
                    }
                    _ => return None,
                }
            }
            value => output.push(value as char),
        }
    }
    None
}

fn parse_json_number_value(bytes: &[u8], mut position: usize) -> Option<(f64, usize)> {
    let start = position;
    while position < bytes.len() {
        let byte = bytes[position];
        if matches!(byte, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
            position += 1;
        } else {
            break;
        }
    }
    if position == start {
        return None;
    }
    let raw = std::str::from_utf8(&bytes[start..position]).ok()?;
    let value = raw.parse::<f64>().ok()?;
    Some((value, position))
}

fn write_index_line(
    writer: &mut BufWriter<File>,
    shard_name: &str,
    key: &str,
    utt_id: &str,
    split: &str,
    num_frames: u64,
    audio_member: &str,
    audio_format: &str,
    json_member: &str,
    audio_offset: u64,
    audio_size: u64,
    json_offset: u64,
    json_size: u64,
) -> io::Result<()> {
    writeln!(
        writer,
        "{{\"shard_name\":{},\"key\":{},\"utt_id\":{},\"split\":{},\"num_frames\":{},\"audio_member\":{},\"audio_format\":{},\"json_member\":{},\"audio_offset\":{},\"audio_size\":{},\"json_offset\":{},\"json_size\":{}}}",
        json_string(shard_name),
        json_string(key),
        json_string(utt_id),
        json_string(split),
        num_frames,
        json_string(audio_member),
        json_string(audio_format),
        json_string(json_member),
        audio_offset,
        audio_size,
        json_offset,
        json_size,
    )
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\u{0008}' => escaped.push_str("\\b"),
            '\u{000C}' => escaped.push_str("\\f"),
            value if value.is_control() => escaped.push_str(&format!("\\u{:04x}", value as u32)),
            value => escaped.push(value),
        }
    }
    escaped.push('"');
    escaped
}

fn format_float(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

fn assign_split(sample_id: &str, hash_seed: u64, eval_ratio: f64) -> &'static str {
    if eval_ratio <= 0.0 {
        return TRAIN_NAME;
    }
    let digest = sha1_digest(format!("{hash_seed}:{sample_id}").as_bytes());
    let bucket = u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]);
    let bucket_ratio = (bucket as f64) / 18446744073709551616.0_f64;
    if bucket_ratio < eval_ratio {
        EVAL_NAME
    } else {
        TRAIN_NAME
    }
}

fn sha1_digest(input: &[u8]) -> [u8; 20] {
    let mut data = input.to_vec();
    let bit_len = (data.len() as u64) * 8;
    data.push(0x80);
    while (data.len() % 64) != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());

    let mut h0: u32 = 0x6745_2301;
    let mut h1: u32 = 0xEFCD_AB89;
    let mut h2: u32 = 0x98BA_DCFE;
    let mut h3: u32 = 0x1032_5476;
    let mut h4: u32 = 0xC3D2_E1F0;

    let mut w = [0_u32; 80];
    for chunk in data.chunks(64) {
        for (index, block) in chunk.chunks(4).take(16).enumerate() {
            w[index] = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
        }
        for index in 16..80 {
            w[index] = (w[index - 3] ^ w[index - 8] ^ w[index - 14] ^ w[index - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for (index, word) in w.iter().enumerate() {
            let (f, k) = match index {
                0..=19 => (((b & c) | ((!b) & d)), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => (((b & c) | (b & d) | (c & d)), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut digest = [0_u8; 20];
    digest[0..4].copy_from_slice(&h0.to_be_bytes());
    digest[4..8].copy_from_slice(&h1.to_be_bytes());
    digest[8..12].copy_from_slice(&h2.to_be_bytes());
    digest[12..16].copy_from_slice(&h3.to_be_bytes());
    digest[16..20].copy_from_slice(&h4.to_be_bytes());
    digest
}

fn log(message: &str) {
    println!("[rwkvasr] {message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn sha1_matches_known_vector() {
        let digest = sha1_digest(b"abc");
        assert_eq!(hex_string(&digest), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn infer_num_frames_prefers_metadata_duration() {
        let json = r#"{"sid":"utt-1","begin_time":0.25,"end_time":1.70}"#;
        let metadata = parse_metadata(json, "fallback", "sid").expect("metadata");
        assert_eq!(metadata.utt_id, "utt-1");
        assert_eq!(metadata.num_frames, 145);
    }

    #[test]
    fn split_member_name_accepts_mp3_audio() {
        let parsed = split_member_name("folder/example.mp3").expect("member");
        assert_eq!(parsed.0, "example");
        assert_eq!(parsed.1, "mp3");
    }

    #[test]
    fn inspect_single_shard_writes_length_index() {
        let temp_root = unique_temp_dir("rwkvasr_tools_tar_test");
        fs::create_dir_all(&temp_root).expect("mkdir");
        let shard_path = temp_root.join("shard_00000001.tar");
        write_tar(
            &shard_path,
            &[
                (
                    "abc.wav",
                    b"RIFF....WAVE".to_vec(),
                ),
                (
                    "abc.json",
                    br#"{"sid":"utt-abc","begin_time":0.0,"end_time":2.34}"#.to_vec(),
                ),
                (
                    "def.wav",
                    b"RIFF....WAVE".to_vec(),
                ),
                (
                    "def.json",
                    br#"{"sid":"utt-def","duration":1.25}"#.to_vec(),
                ),
            ],
        )
        .expect("write tar");

        let output_path = temp_root.join("webdataset_lengths.jsonl");
        let summary_path = temp_root.join("webdataset_lengths.summary.json");
        let config = Config {
            webdataset_root: temp_root.clone(),
            shard_pattern: String::from("*.tar"),
            output_path: output_path.clone(),
            summary_path: summary_path.clone(),
            eval_ratio: 0.0,
            hash_seed: 0,
            utt_id_key: String::from("sid"),
            split_by: SplitBy::ShardName,
            threads: 1,
        };

        let temp_dir = temp_root.join(".parts");
        fs::create_dir_all(&temp_dir).expect("temp dir");
        let shard = inspect_shard(0, &shard_path, &temp_dir, &config).expect("inspect");
        merge_part_files(&output_path, &[shard]).expect("merge");
        write_summary(&config, 1, 2, 2, 0, 125, 234, &BTreeMap::from([(1, 1), (2, 1)])).expect("summary");

        let index_text = fs::read_to_string(output_path).expect("read index");
        assert!(index_text.contains("\"utt_id\":\"utt-abc\""));
        assert!(index_text.contains("\"num_frames\":234"));
        assert!(index_text.contains("\"audio_member\":\"abc.wav\""));
        assert!(index_text.contains("\"audio_format\":\"wav\""));
        assert!(index_text.contains("\"utt_id\":\"utt-def\""));
        assert!(index_text.contains("\"num_frames\":125"));

        let summary_text = fs::read_to_string(summary_path).expect("read summary");
        assert!(summary_text.contains("\"num_samples\": 2"));

        fs::remove_dir_all(temp_root).expect("cleanup");
    }

    fn hex_string(bytes: &[u8]) -> String {
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            output.push_str(&format!("{byte:02x}"));
        }
        output
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        env::temp_dir().join(format!("{prefix}_{stamp}_{}", std::process::id()))
    }

    fn write_tar(path: &Path, entries: &[(&str, Vec<u8>)]) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for (name, payload) in entries {
            let header = tar_header(name, payload.len() as u64);
            writer.write_all(&header)?;
            writer.write_all(payload)?;
            let padding = padded_tar_size(payload.len() as u64) - payload.len() as u64;
            if padding > 0 {
                writer.write_all(&vec![0_u8; padding as usize])?;
            }
        }
        writer.write_all(&[0_u8; TAR_BLOCK_SIZE])?;
        writer.write_all(&[0_u8; TAR_BLOCK_SIZE])?;
        writer.flush()?;
        Ok(())
    }

    fn tar_header(name: &str, file_size: u64) -> [u8; TAR_BLOCK_SIZE] {
        let mut header = [0_u8; TAR_BLOCK_SIZE];
        let name_bytes = name.as_bytes();
        header[..name_bytes.len()].copy_from_slice(name_bytes);
        write_octal(&mut header[100..108], 0o644);
        write_octal(&mut header[108..116], 0);
        write_octal(&mut header[116..124], 0);
        write_octal(&mut header[124..136], file_size);
        write_octal(&mut header[136..148], 0);
        header[148..156].fill(b' ');
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let checksum: u32 = header.iter().map(|value| *value as u32).sum();
        write_checksum(&mut header[148..156], checksum);
        header
    }

    fn write_octal(field: &mut [u8], value: u64) {
        let width = field.len();
        let text = format!("{value:0width$o}\0", width = width - 1);
        field.copy_from_slice(text.as_bytes());
    }

    fn write_checksum(field: &mut [u8], value: u32) {
        let text = format!("{value:06o}\0 ",);
        field.copy_from_slice(text.as_bytes());
    }
}
