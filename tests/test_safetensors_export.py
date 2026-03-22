from pathlib import Path

import torch

from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.training.checkpoint import export_checkpoint_to_safetensors, load_checkpoint, save_checkpoint


def _build_model() -> RWKVCTCModel:
    return RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=32,
            n_embd=32,
            dim_att=32,
            dim_ff=64,
            num_layers=1,
            vocab_size=8,
            head_size=8,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
        )
    )


def test_export_checkpoint_to_safetensors_roundtrip(tmp_path: Path) -> None:
    model = _build_model()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(0.01 * (index + 1))

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model=model, step=11, extra={"tag": "best"})

    safetensors_path = tmp_path / "model.safetensors"
    exported = export_checkpoint_to_safetensors(checkpoint_path, safetensors_path)

    restored_model = _build_model()
    restored = load_checkpoint(safetensors_path, model=restored_model)

    assert exported["step"] == 11
    assert restored["step"] == 0
    assert safetensors_path.exists()

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored_model.state_dict()[name]), name
