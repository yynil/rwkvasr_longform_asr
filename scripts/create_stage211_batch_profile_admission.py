from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rwkvasr.eval.stage211_batch_profile import (
    STAGE211_BATCH_PROFILE_PHASES,
    build_stage211_batch_profile_admission,
    validate_stage211_batch_profile_admission,
)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Refusing to overwrite a different admission receipt: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an explicit immutable Stage211 batch-profile admission receipt."
    )
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--phase", choices=STAGE211_BATCH_PROFILE_PHASES, required=True)
    parser.add_argument("--admitted-by", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--admit-recommended-profile",
        action="store_true",
        help="Required explicit acknowledgement that the measured recommendation is admitted.",
    )
    args = parser.parse_args()
    if not args.admit_recommended_profile:
        parser.error("--admit-recommended-profile is required")
    output = args.output.expanduser().resolve()
    if output.is_file():
        validated = validate_stage211_batch_profile_admission(
            output,
            phase=str(args.phase),
        )
        expected_existing = {
            "preflight_report_path": str(args.preflight_report.expanduser().resolve()),
            "admitted_by": str(args.admitted_by).strip(),
            "reason": str(args.reason).strip(),
        }
        if any(validated.get(key) != value for key, value in expected_existing.items()):
            raise ValueError(
                f"Existing admission receipt does not match this request: {output}"
            )
        print(
            "batch_profile_admission="
            f"{output} phase={validated['phase']} "
            f"profile={validated['selected_profile']['name']} "
            f"sha256={validated['receipt_sha256']} status=reused",
            flush=True,
        )
        return 0
    receipt = build_stage211_batch_profile_admission(
        args.preflight_report,
        phase=str(args.phase),
        admitted_by=str(args.admitted_by),
        reason=str(args.reason),
    )
    _atomic_write_json(output, receipt)
    validated = validate_stage211_batch_profile_admission(
        output,
        phase=str(args.phase),
    )
    print(
        "batch_profile_admission="
        f"{output} phase={validated['phase']} "
        f"profile={validated['selected_profile']['name']} "
        f"sha256={validated['receipt_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
