use serde::{Deserialize, Serialize};
use serde_json::Value;
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
    text_cost_source: TextCostSource,
    text_cost_weight: f64,
    json_size_text_offset: u64,
    json_size_bytes_per_token: f64,
    source_field: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LengthEntry {
    shard_name: String,
    split: String,
    num_frames: u64,
    #[serde(default)]
    num_text_tokens: Option<u64>,
    #[serde(default)]
    num_text_chars: Option<u64>,
    #[serde(default)]
    text_bytes: Option<u64>,
    #[serde(default)]
    json_size: Option<u64>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
struct BucketManifest {
    version: u32,
    root: String,
    source_length_index_path: String,
    bucket_width: u64,
    bucket_metric: String,
    text_cost_source: String,
    text_cost_weight: f64,
    json_size_text_offset: u64,
    json_size_bytes_per_token: f64,
    entries_per_part: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_field: Option<String>,
    splits: BTreeMap<String, SplitManifest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextCostSource {
    None,
    Auto,
    NumTextTokens,
    NumTextChars,
    TextBytes,
    JsonSize,
}

impl TextCostSource {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(Self::None),
            "auto" => Ok(Self::Auto),
            "num_text_tokens" => Ok(Self::NumTextTokens),
            "num_text_chars" => Ok(Self::NumTextChars),
            "text_bytes" => Ok(Self::TextBytes),
            "json_size" => Ok(Self::JsonSize),
            other => Err(format!(
                "Unsupported --text-cost-source {other:?}; expected none, auto, num_text_tokens, num_text_chars, text_bytes, or json_size."
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Auto => "auto",
            Self::NumTextTokens => "num_text_tokens",
            Self::NumTextChars => "num_text_chars",
            Self::TextBytes => "text_bytes",
            Self::JsonSize => "json_size",
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    source_label: Option<String>,
}

struct BucketWriter {
    split: String,
    bucket_id: u64,
    source_label: Option<String>,
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
    fn new(
        split: String,
        bucket_id: u64,
        source_label: Option<String>,
        output_dir: PathBuf,
        entries_per_part: u64,
    ) -> Self {
        Self {
            split,
            bucket_id,
            source_label,
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
        let writer = self
            .current_writer
            .as_mut()
            .ok_or_else(|| String::from("missing current writer"))?;
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
        let mut relative_dir =
            PathBuf::from(&self.split).join(format!("bucket_{:04}", self.bucket_id));
        if let Some(source_label) = &self.source_label {
            relative_dir = relative_dir.join(format!(
                "source_{}",
                encode_path_component(source_label.as_bytes())
            ));
        }
        let relative_path = relative_dir.join(format!("part_{:06}.jsonl", self.part_index));
        let full_path = self.output_dir.join(&relative_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "Failed to create bucket part dir {}: {err}",
                    parent.display()
                )
            })?;
        }
        let file = File::create(&full_path).map_err(|err| {
            format!(
                "Failed to create bucket part {}: {err}",
                full_path.display()
            )
        })?;
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
        writer
            .flush()
            .map_err(|err| format!("Failed flushing bucket part: {err}"))?;
        if let Some(path) = self.current_rel_path.take() {
            self.parts.push(PartInfo {
                path,
                num_samples: self.current_count,
                first_shard: self.current_first_shard.take(),
                last_shard: self.current_last_shard.take(),
                source_label: self.source_label.clone(),
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
        fs::remove_dir_all(&config.output_dir).map_err(|err| {
            format!(
                "Failed to remove stale bucket dir {}: {err}",
                config.output_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&config.output_dir).map_err(|err| {
        format!(
            "Failed to create bucket dir {}: {err}",
            config.output_dir.display()
        )
    })?;
    if let Some(parent) = config.manifest_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Failed to create manifest dir {}: {err}", parent.display()))?;
    }

    let mut writers = BTreeMap::<(String, u64, String), BucketWriter>::new();
    let input = BufReader::new(File::open(&config.length_index_path).map_err(|err| {
        format!(
            "Failed to open {}: {err}",
            config.length_index_path.display()
        )
    })?);
    let start = Instant::now();
    let mut processed = 0_u64;

    for line in input.lines() {
        let line = line.map_err(|err| format!("Failed reading length index: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: LengthEntry = serde_json::from_str(&line)
            .map_err(|err| format!("Invalid length index JSON: {err}"))?;
        let bucket_cost = combined_bucket_cost(&entry, &config);
        let bucket_id = bucket_cost / config.bucket_width;
        let source_label = entry_source_label(&entry, config.source_field.as_deref());
        let source_key = source_label.clone().unwrap_or_default();
        let key = (entry.split.clone(), bucket_id, source_key);
        let writer = writers.entry(key).or_insert_with(|| {
            BucketWriter::new(
                entry.split.clone(),
                bucket_id,
                source_label,
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

    let mut merged_buckets = BTreeMap::<(String, u64), BucketInfo>::new();
    for ((split_name, bucket_id, _source_key), writer) in writers {
        let bucket = writer.finalize()?;
        let merged = merged_buckets
            .entry((split_name, bucket_id))
            .or_insert_with(|| BucketInfo {
                bucket_id,
                num_samples: 0,
                parts: Vec::new(),
            });
        merged.num_samples += bucket.num_samples;
        merged.parts.extend(bucket.parts);
    }

    let mut splits = BTreeMap::<String, SplitManifest>::new();
    for ((split_name, _bucket_id), bucket) in merged_buckets {
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
        bucket_metric: if config.text_cost_weight > 0.0
            && config.text_cost_source != TextCostSource::None
        {
            String::from("audio_frames_plus_text_cost")
        } else {
            String::from("audio_frames")
        },
        text_cost_source: config.text_cost_source.as_str().to_string(),
        text_cost_weight: config.text_cost_weight,
        json_size_text_offset: config.json_size_text_offset,
        json_size_bytes_per_token: config.json_size_bytes_per_token,
        entries_per_part: config.entries_per_part,
        source_field: config.source_field.clone(),
        splits,
    };
    let writer = BufWriter::new(File::create(&config.manifest_path).map_err(|err| {
        format!(
            "Failed to create manifest {}: {err}",
            config.manifest_path.display()
        )
    })?);
    serde_json::to_writer_pretty(writer, &manifest).map_err(|err| {
        format!(
            "Failed writing manifest {}: {err}",
            config.manifest_path.display()
        )
    })?;
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
    let mut text_cost_source = TextCostSource::None;
    let mut text_cost_weight = 0.0_f64;
    let mut json_size_text_offset = 256_u64;
    let mut json_size_bytes_per_token = 4.0_f64;
    let mut source_field: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--shard-root" => shard_root = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--length-index-path" => {
                length_index_path = Some(PathBuf::from(next_value(&mut args, &arg)?))
            }
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
            "--text-cost-source" => {
                text_cost_source = TextCostSource::parse(&next_value(&mut args, &arg)?)?;
            }
            "--text-cost-weight" => {
                text_cost_weight = next_value(&mut args, &arg)?
                    .parse::<f64>()
                    .map_err(|err| format!("Invalid --text-cost-weight: {err}"))?;
            }
            "--json-size-text-offset" => {
                json_size_text_offset = next_value(&mut args, &arg)?
                    .parse::<u64>()
                    .map_err(|err| format!("Invalid --json-size-text-offset: {err}"))?;
            }
            "--json-size-bytes-per-token" => {
                json_size_bytes_per_token = next_value(&mut args, &arg)?
                    .parse::<f64>()
                    .map_err(|err| format!("Invalid --json-size-bytes-per-token: {err}"))?;
            }
            "--source-field" => source_field = Some(next_value(&mut args, &arg)?),
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
    if text_cost_weight < 0.0 {
        return Err(String::from("--text-cost-weight must be non-negative."));
    }
    if json_size_bytes_per_token <= 0.0 {
        return Err(String::from(
            "--json-size-bytes-per-token must be positive.",
        ));
    }

    Ok(Config {
        shard_root,
        length_index_path,
        output_dir,
        manifest_path,
        bucket_width,
        entries_per_part,
        text_cost_source,
        text_cost_weight,
        json_size_text_offset,
        json_size_bytes_per_token,
        source_field,
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
    println!("  --text-cost-source VALUE     none|auto|num_text_tokens|num_text_chars|text_bytes|json_size; default: none");
    println!("  --text-cost-weight FLOAT     frames per estimated text token; default: 0");
    println!(
        "  --json-size-text-offset INT  bytes to subtract when using json_size proxy; default: 256"
    );
    println!(
        "  --json-size-bytes-per-token FLOAT  json bytes per estimated text token; default: 4.0"
    );
    println!("  --source-field FIELD         group parts by this JSON field; default: disabled");
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("Missing value for {flag}."))
}

fn entry_source_label(entry: &LengthEntry, source_field: Option<&str>) -> Option<String> {
    let source_field = source_field?;
    let label = entry.extra.get(source_field).and_then(|value| match value {
        Value::String(value) => Some(value.trim().to_string()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    });
    Some(
        label
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| String::from("unknown")),
    )
}

fn encode_path_component(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn combined_bucket_cost(entry: &LengthEntry, config: &Config) -> u64 {
    if config.text_cost_weight <= 0.0 || config.text_cost_source == TextCostSource::None {
        return entry.num_frames;
    }
    let text_units = text_cost_units(entry, config).unwrap_or(0.0);
    let text_cost = (config.text_cost_weight * text_units).round().max(0.0) as u64;
    entry.num_frames.saturating_add(text_cost)
}

fn text_cost_units(entry: &LengthEntry, config: &Config) -> Option<f64> {
    match config.text_cost_source {
        TextCostSource::None => Some(0.0),
        TextCostSource::Auto => entry
            .num_text_tokens
            .map(|value| value as f64)
            .or_else(|| entry.num_text_chars.map(|value| value as f64))
            .or_else(|| entry.text_bytes.map(|value| value as f64 / 4.0))
            .or_else(|| json_size_proxy_units(entry, config)),
        TextCostSource::NumTextTokens => entry.num_text_tokens.map(|value| value as f64),
        TextCostSource::NumTextChars => entry.num_text_chars.map(|value| value as f64),
        TextCostSource::TextBytes => entry.text_bytes.map(|value| value as f64 / 4.0),
        TextCostSource::JsonSize => json_size_proxy_units(entry, config),
    }
}

fn json_size_proxy_units(entry: &LengthEntry, config: &Config) -> Option<f64> {
    let json_size = entry.json_size?;
    let text_bytes = json_size.saturating_sub(config.json_size_text_offset);
    Some(text_bytes as f64 / config.json_size_bytes_per_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length_entry(payload: &str) -> LengthEntry {
        serde_json::from_str(payload).expect("valid length entry")
    }

    #[test]
    fn source_label_reads_dynamic_string_field() {
        let entry = length_entry(
            r#"{"shard_name":"x.tar","split":"train","num_frames":100,"dataset":"cv22_en"}"#,
        );

        assert_eq!(
            entry_source_label(&entry, Some("dataset")).as_deref(),
            Some("cv22_en")
        );
        assert_eq!(entry_source_label(&entry, None), None);
    }

    #[test]
    fn source_label_maps_missing_or_empty_values_to_unknown() {
        let missing = length_entry(r#"{"shard_name":"x.tar","split":"train","num_frames":100}"#);
        let empty = length_entry(
            r#"{"shard_name":"x.tar","split":"train","num_frames":100,"dataset":"  "}"#,
        );

        assert_eq!(
            entry_source_label(&missing, Some("dataset")).as_deref(),
            Some("unknown")
        );
        assert_eq!(
            entry_source_label(&empty, Some("dataset")).as_deref(),
            Some("unknown")
        );
    }

    #[test]
    fn source_path_component_is_unambiguous_hex() {
        assert_eq!(encode_path_component(b"cv22/en"), "637632322f656e");
        assert_ne!(encode_path_component(b"/"), encode_path_component(b"2f"));
    }
}
