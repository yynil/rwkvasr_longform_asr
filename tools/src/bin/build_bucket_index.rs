use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
struct Config {
    shard_root: PathBuf,
    length_index_path: PathBuf,
    output_dir: PathBuf,
    manifest_path: PathBuf,
    bucket_width: u64,
    entries_per_part: u64,
}

#[derive(Debug, Deserialize)]
struct LengthEntry {
    shard_name: String,
    split: String,
    num_frames: u64,
}

#[derive(Debug, Serialize)]
struct BucketManifest {
    version: u32,
    root: String,
    source_length_index_path: String,
    bucket_width: u64,
    entries_per_part: u64,
    splits: BTreeMap<String, SplitManifest>,
}

#[derive(Debug, Serialize, Default)]
struct SplitManifest {
    num_samples: u64,
    buckets: Vec<BucketInfo>,
}

#[derive(Debug, Serialize)]
struct BucketInfo {
    bucket_id: u64,
    num_samples: u64,
    parts: Vec<PartInfo>,
}

#[derive(Debug, Serialize)]
struct PartInfo {
    path: String,
    num_samples: u64,
    first_shard: Option<String>,
    last_shard: Option<String>,
}

struct BucketWriter {
    split: String,
    bucket_id: u64,
    output_dir: PathBuf,
    entries_per_part: u64,
    part_index: u64,
    total_samples: u64,
    current_count: u64,
    current_first_shard: Option<String>,
    current_last_shard: Option<String>,
    current_rel_path: Option<String>,
    current_writer: Option<BufWriter<File>>,
    parts: Vec<PartInfo>,
}

impl BucketWriter {
    fn new(split: String, bucket_id: u64, output_dir: PathBuf, entries_per_part: u64) -> Self {
        Self {
            split,
            bucket_id,
            output_dir,
            entries_per_part,
            part_index: 0,
            total_samples: 0,
            current_count: 0,
            current_first_shard: None,
            current_last_shard: None,
            current_rel_path: None,
            current_writer: None,
            parts: Vec::new(),
        }
    }

    fn append_line(&mut self, line: &str, shard_name: &str) -> Result<(), String> {
        if self.current_writer.is_none() || self.current_count >= self.entries_per_part {
            self.finish_current_part()?;
            self.open_new_part()?;
        }
        let writer = self.current_writer.as_mut().ok_or_else(|| String::from("missing current writer"))?;
        writer
            .write_all(line.as_bytes())
            .map_err(|err| format!("Failed writing bucket part: {err}"))?;
        self.current_count += 1;
        self.total_samples += 1;
        if self.current_first_shard.is_none() {
            self.current_first_shard = Some(shard_name.to_string());
        }
        self.current_last_shard = Some(shard_name.to_string());
        Ok(())
    }

    fn finalize(mut self) -> Result<BucketInfo, String> {
        self.finish_current_part()?;
        Ok(BucketInfo {
            bucket_id: self.bucket_id,
            num_samples: self.total_samples,
            parts: self.parts,
        })
    }

    fn open_new_part(&mut self) -> Result<(), String> {
        let relative_dir = PathBuf::from(&self.split).join(format!("bucket_{:04}", self.bucket_id));
        let relative_path = relative_dir.join(format!("part_{:06}.jsonl", self.part_index));
        let full_path = self.output_dir.join(&relative_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Failed to create bucket part dir {}: {err}", parent.display()))?;
        }
        let file = File::create(&full_path)
            .map_err(|err| format!("Failed to create bucket part {}: {err}", full_path.display()))?;
        self.current_writer = Some(BufWriter::new(file));
        self.current_rel_path = Some(relative_path.to_string_lossy().to_string());
        self.current_count = 0;
        self.current_first_shard = None;
        self.current_last_shard = None;
        self.part_index += 1;
        Ok(())
    }

    fn finish_current_part(&mut self) -> Result<(), String> {
        let Some(mut writer) = self.current_writer.take() else {
            return Ok(());
        };
        writer.flush().map_err(|err| format!("Failed flushing bucket part: {err}"))?;
        if let Some(path) = self.current_rel_path.take() {
            self.parts.push(PartInfo {
                path,
                num_samples: self.current_count,
                first_shard: self.current_first_shard.take(),
                last_shard: self.current_last_shard.take(),
            });
        }
        self.current_count = 0;
        Ok(())
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[rwkvasr] error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args()?;
    if config.output_dir.exists() {
        fs::remove_dir_all(&config.output_dir)
            .map_err(|err| format!("Failed to remove stale bucket dir {}: {err}", config.output_dir.display()))?;
    }
    fs::create_dir_all(&config.output_dir)
        .map_err(|err| format!("Failed to create bucket dir {}: {err}", config.output_dir.display()))?;
    if let Some(parent) = config.manifest_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Failed to create manifest dir {}: {err}", parent.display()))?;
    }

    let mut writers = BTreeMap::<(String, u64), BucketWriter>::new();
    let input = BufReader::new(
        File::open(&config.length_index_path)
            .map_err(|err| format!("Failed to open {}: {err}", config.length_index_path.display()))?,
    );
    let start = Instant::now();
    let mut processed = 0_u64;

    for line in input.lines() {
        let line = line.map_err(|err| format!("Failed reading length index: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: LengthEntry =
            serde_json::from_str(&line).map_err(|err| format!("Invalid length index JSON: {err}"))?;
        let bucket_id = entry.num_frames / config.bucket_width;
        let key = (entry.split.clone(), bucket_id);
        let writer = writers.entry(key).or_insert_with(|| {
            BucketWriter::new(
                entry.split.clone(),
                bucket_id,
                config.output_dir.clone(),
                config.entries_per_part,
            )
        });
        writer.append_line(&(line + "\n"), &entry.shard_name)?;
        processed += 1;
        if processed % 1_000_000 == 0 {
            let elapsed = start.elapsed().as_secs_f32();
            println!(
                "[rwkvasr] Bucket index progress: samples={}, elapsed={elapsed:.1}s",
                processed
            );
        }
    }

    let mut splits = BTreeMap::<String, SplitManifest>::new();
    for ((split_name, _bucket_id), writer) in writers {
        let bucket = writer.finalize()?;
        let split_entry = splits.entry(split_name).or_default();
        split_entry.num_samples += bucket.num_samples;
        split_entry.buckets.push(bucket);
    }
    for split in splits.values_mut() {
        split.buckets.sort_by_key(|bucket| bucket.bucket_id);
    }

    let manifest = BucketManifest {
        version: 1,
        root: config.shard_root.display().to_string(),
        source_length_index_path: config.length_index_path.display().to_string(),
        bucket_width: config.bucket_width,
        entries_per_part: config.entries_per_part,
        splits,
    };
    let writer = BufWriter::new(
        File::create(&config.manifest_path)
            .map_err(|err| format!("Failed to create manifest {}: {err}", config.manifest_path.display()))?,
    );
    serde_json::to_writer_pretty(writer, &manifest)
        .map_err(|err| format!("Failed writing manifest {}: {err}", config.manifest_path.display()))?;
    println!(
        "build_bucket_index samples={} manifest={}",
        processed,
        config.manifest_path.display()
    );
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut shard_root: Option<PathBuf> = None;
    let mut length_index_path: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut manifest_path: Option<PathBuf> = None;
    let mut bucket_width = 80_u64;
    let mut entries_per_part = 100_000_u64;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--shard-root" => shard_root = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--length-index-path" => length_index_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--manifest-path" => manifest_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--bucket-width" => {
                bucket_width = next_value(&mut args, &arg)?
                    .parse::<u64>()
                    .map_err(|err| format!("Invalid --bucket-width: {err}"))?;
            }
            "--entries-per-part" => {
                entries_per_part = next_value(&mut args, &arg)?
                    .parse::<u64>()
                    .map_err(|err| format!("Invalid --entries-per-part: {err}"))?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument {other:?}. Use --help for usage.")),
        }
    }

    let shard_root = shard_root.ok_or_else(|| String::from("--shard-root is required."))?;
    let length_index_path =
        length_index_path.ok_or_else(|| String::from("--length-index-path is required."))?;
    let output_dir = output_dir.ok_or_else(|| String::from("--output-dir is required."))?;
    let manifest_path = manifest_path.unwrap_or_else(|| output_dir.join("manifest.json"));
    if bucket_width == 0 {
        return Err(String::from("--bucket-width must be positive."));
    }
    if entries_per_part == 0 {
        return Err(String::from("--entries-per-part must be positive."));
    }

    Ok(Config {
        shard_root,
        length_index_path,
        output_dir,
        manifest_path,
        bucket_width,
        entries_per_part,
    })
}

fn print_help() {
    println!("Build a bucketed external-memory manifest from a large WebDataset length index.");
    println!();
    println!("Required:");
    println!("  --shard-root PATH");
    println!("  --length-index-path PATH");
    println!("  --output-dir PATH");
    println!();
    println!("Optional:");
    println!("  --manifest-path PATH         default: <output-dir>/manifest.json");
    println!("  --bucket-width INT           default: 80");
    println!("  --entries-per-part INT       default: 100000");
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("Missing value for {flag}."))
}
