from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.run_stage211_strict_chained_alignment import (
        PHASE_SEQUENCE,
        _build_promotion_receipt,
        _validate_promotion_receipt,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage211_strict_chained_alignment import (
        PHASE_SEQUENCE,
        _build_promotion_receipt,
        _validate_promotion_receipt,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a passing Stage211 gate report to the exact checkpoint selected "
            "for the next strict phase."
        )
    )
    parser.add_argument("--source-phase", choices=PHASE_SEQUENCE[:-1], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-gate-passed",
        action="store_true",
        help="Required explicit confirmation; training completion alone is not a passing gate.",
    )
    args = parser.parse_args()

    if not args.confirm_gate_passed:
        parser.error("--confirm-gate-passed is required")
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        target_phase = PHASE_SEQUENCE[PHASE_SEQUENCE.index(str(args.source_phase)) + 1]
        receipt = _validate_promotion_receipt(
            receipt_path=output_path,
            target_phase=target_phase,
            checkpoint_path=args.checkpoint,
        )
        if Path(str(receipt["gate_report_path"])).resolve() != args.gate_report.resolve():
            raise ValueError(f"Refusing to overwrite a different receipt: {output_path}")
    else:
        receipt = _build_promotion_receipt(
            source_phase=str(args.source_phase),
            checkpoint_path=args.checkpoint,
            gate_report_path=args.gate_report,
        )
        output_path.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"promotion_receipt={output_path} "
        f"transition={receipt['source_phase']}->{receipt['target_phase']} "
        f"checkpoint_sha256={receipt['checkpoint_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
