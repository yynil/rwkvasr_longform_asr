from __future__ import annotations

import os
from pathlib import Path

import pytest

from rwkvasr.eval import stage211_artifact_io


def test_immutable_publication_is_idempotent_and_rejects_different_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "gate.json"
    stage211_artifact_io.write_immutable_json(
        output,
        {"gate_passed": True},
        label="Stage211 gate",
    )
    original = output.read_bytes()

    stage211_artifact_io.write_immutable_json(
        output,
        {"gate_passed": True},
        label="Stage211 gate",
    )
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="Refusing to overwrite a different Stage211 gate"):
        stage211_artifact_io.write_immutable_json(
            output,
            {"gate_passed": False},
            label="Stage211 gate",
        )


def test_failed_atomic_publish_leaves_no_partial_final_and_restart_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "stage211_complete.json"
    payload = b'{"complete":true}\n'
    original_link = os.link

    def fail_publish(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        assert Path(source).read_bytes() == payload
        assert Path(target) == output
        assert not output.exists()
        raise OSError("injected publish failure")

    monkeypatch.setattr(stage211_artifact_io.os, "link", fail_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        stage211_artifact_io.write_immutable_bytes(
            output,
            payload,
            label="Stage211 final report",
        )

    assert not output.exists()
    assert list(tmp_path.glob(".stage211_complete.json.*.tmp")) == []

    monkeypatch.setattr(stage211_artifact_io.os, "link", original_link)
    stage211_artifact_io.write_immutable_bytes(
        output,
        payload,
        label="Stage211 final report",
    )
    assert output.read_bytes() == payload


def test_nonfinite_strict_json_is_rejected_before_publication(tmp_path: Path) -> None:
    output = tmp_path / "alignment.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        stage211_artifact_io.write_immutable_json(
            output,
            {"loss": float("nan")},
            label="Stage211 alignment eval",
            allow_nan=False,
        )
    assert not output.exists()
