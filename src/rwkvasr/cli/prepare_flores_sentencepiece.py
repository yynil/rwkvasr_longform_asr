from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import sentencepiece as spm

from rwkvasr.config import save_yaml


DEFAULT_FLORES_URL = "https://tinyurl.com/flores200dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build multilingual SentencePiece unigram tokenizers from the official FLORES-200 text archive."
    )
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--archive-url", default=DEFAULT_FLORES_URL)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[4000, 8000])
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--model-prefix-base", default="flores200_unigram")
    parser.add_argument("--normalization-rule-name", default="nmt_nfkc")
    parser.add_argument("--shuffle-input-sentence", action="store_true", default=True)
    parser.add_argument("--no-shuffle-input-sentence", dest="shuffle_input_sentence", action="store_false")
    parser.add_argument("--input-sentence-size", type=int, default=0)
    parser.add_argument("--seed-sentencepiece-size", type=int, default=1_000_000)
    return parser


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _build_corpus_from_flores_archive(archive_path: Path, corpus_path: Path) -> dict[str, int]:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    num_lines = 0
    num_files = 0
    num_languages = 0
    seen_languages: set[str] = set()

    with tarfile.open(archive_path, "r:gz") as archive, corpus_path.open("w", encoding="utf-8") as output:
        for member in archive:
            if not member.isfile():
                continue
            member_path = Path(member.name)
            if len(member_path.parts) < 3:
                continue
            split_dir = member_path.parts[-2]
            file_name = member_path.name
            if split_dir not in {"dev", "devtest"}:
                continue
            if not (file_name.endswith(".dev") or file_name.endswith(".devtest")):
                continue

            language_code = file_name.split(".", 1)[0]
            seen_languages.add(language_code)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            num_files += 1
            for raw_line in extracted:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                output.write(line)
                output.write("\n")
                num_lines += 1

    num_languages = len(seen_languages)
    return {
        "num_lines": num_lines,
        "num_files": num_files,
        "num_languages": num_languages,
    }


def _train_sentencepiece_models(
    *,
    corpus_path: Path,
    output_dir: Path,
    model_prefix_base: str,
    vocab_sizes: list[int],
    character_coverage: float,
    normalization_rule_name: str,
    shuffle_input_sentence: bool,
    input_sentence_size: int,
    seed_sentencepiece_size: int,
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for vocab_size in vocab_sizes:
        suffix = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
        model_prefix = output_dir / f"{model_prefix_base}_{suffix}"
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            model_type="unigram",
            vocab_size=int(vocab_size),
            character_coverage=float(character_coverage),
            normalization_rule_name=normalization_rule_name,
            shuffle_input_sentence=bool(shuffle_input_sentence),
            input_sentence_size=int(input_sentence_size),
            seed_sentencepiece_size=int(seed_sentencepiece_size),
            split_digits=True,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )
        artifacts.append(
            {
                "vocab_size": int(vocab_size),
                "model_path": str(model_prefix.with_suffix(".model")),
                "vocab_path": str(model_prefix.with_suffix(".vocab")),
            }
        )
    return artifacts


def main() -> None:
    args = build_parser().parse_args()
    archive_path = Path(args.archive_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        if not args.download_if_missing:
            raise FileNotFoundError(
                f"FLORES archive not found: {archive_path}. Pass --download-if-missing to fetch it."
            )
        print(f"[rwkvasr] Downloading FLORES archive -> {archive_path}", flush=True)
        _download_file(args.archive_url, archive_path)

    corpus_path = output_dir / "flores200_dev_devtest_multilingual.txt"
    corpus_summary = _build_corpus_from_flores_archive(archive_path, corpus_path)
    print(
        "[rwkvasr] Built FLORES corpus "
        f"languages={corpus_summary['num_languages']} files={corpus_summary['num_files']} "
        f"lines={corpus_summary['num_lines']} -> {corpus_path}",
        flush=True,
    )

    artifacts = _train_sentencepiece_models(
        corpus_path=corpus_path,
        output_dir=output_dir,
        model_prefix_base=args.model_prefix_base,
        vocab_sizes=[int(vocab_size) for vocab_size in args.vocab_sizes],
        character_coverage=args.character_coverage,
        normalization_rule_name=args.normalization_rule_name,
        shuffle_input_sentence=args.shuffle_input_sentence,
        input_sentence_size=args.input_sentence_size,
        seed_sentencepiece_size=args.seed_sentencepiece_size,
    )

    summary = {
        "source_archive": str(archive_path),
        "source_url": args.archive_url,
        "corpus_path": str(corpus_path),
        "corpus_summary": corpus_summary,
        "model_type": "unigram",
        "character_coverage": float(args.character_coverage),
        "normalization_rule_name": args.normalization_rule_name,
        "artifacts": artifacts,
    }
    save_yaml(output_dir / "sentencepiece_flores200_summary.yaml", summary)
    (output_dir / "sentencepiece_flores200_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for artifact in artifacts:
        vocab_size = artifact["vocab_size"]
        tokenizer_config = {
            "tokenizer_type": "sentencepiece",
            "tokenizer_model_path": artifact["model_path"],
            "tokenizer_language": None,
            "tokenizer_task": None,
            "vocab_size": int(vocab_size),
        }
        suffix = f"{int(vocab_size) // 1000}k" if int(vocab_size) % 1000 == 0 else str(vocab_size)
        save_yaml(output_dir / f"{args.model_prefix_base}_{suffix}.tokenizer_config.yaml", tokenizer_config)

    print(
        "prepare_flores_sentencepiece "
        f"output_dir={output_dir} vocab_sizes={','.join(str(item['vocab_size']) for item in artifacts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
