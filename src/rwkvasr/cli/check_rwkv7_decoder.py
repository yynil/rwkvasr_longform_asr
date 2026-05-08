from __future__ import annotations

import argparse

import torch

from rwkvasr.data import build_text_tokenizer
from rwkvasr.modules import RWKV7DecoderLM, infer_rwkv7_decoder_config_from_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate official RWKV-7 tokenizer and decoder checkpoint wiring.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-model-path", required=True)
    parser.add_argument("--prompt", default="User: say hello in English and Chinese\n\nAssistant:")
    parser.add_argument("--top-k", default=5, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = build_text_tokenizer("rwkv", model_path=args.tokenizer_model_path)
    token_ids = tokenizer.encode(args.prompt)
    decoded = tokenizer.decode(token_ids)
    config = infer_rwkv7_decoder_config_from_checkpoint(args.checkpoint_path)
    model = RWKV7DecoderLM(config)
    model.load_official_checkpoint(args.checkpoint_path)
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward_tokens(torch.tensor([token_ids], dtype=torch.long))
        next_logits = logits[0, -1].float()
        probs = torch.softmax(next_logits, dim=-1)
        values, indices = torch.topk(probs, k=max(1, int(args.top_k)))

    print(f"rwkv7_decoder_check tokenizer_vocab={tokenizer.vocab_size} decoder_vocab={config.vocab_size}")
    print(f"prompt_roundtrip_ok={decoded == args.prompt}")
    print(
        "decoder_config "
        f"layers={config.num_layers} n_embd={config.n_embd} "
        f"head_size={config.head_size} ffn_hidden={config.resolved_ffn_hidden_size}"
    )
    for rank, (token_id, prob) in enumerate(zip(indices.tolist(), values.tolist(), strict=True), start=1):
        token_text = tokenizer.decode([int(token_id)]) if int(token_id) != 0 else "<missing-token-0>"
        print(f"top{rank} token_id={int(token_id)} prob={float(prob):.4f} text={token_text!r}")


if __name__ == "__main__":
    main()
