"""Export a fine-tuned sentence-transformers model to ONNX for fastembed (`make export`).

The output folder is what EMBEDDING_MODEL can point at (`retrieval/embedder.py`):

- `model.onnx`: the transformer only (token ids in, one vector per token out), with
  dynamic batch and sequence length. Pooling and normalization stay outside the graph:
  fastembed does them, as for its own models.
- the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, ...) and `config.json`.
- `fastembed.json`: dim, pooling and normalization, read from the model's own
  `modules.json` and `1_Pooling/config.json`, so serving matches training.

torch is in the `train` group, so it's imported inside `export_onnx`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rig_assist.retrieval.embedder import FASTEMBED_SPEC

MODEL_FILE = "model.onnx"
_POOLING = {"cls": "CLS", "mean": "MEAN", "lasttoken": "LAST_TOKEN"}  # what fastembed can do


def fastembed_spec(model_dir: Path) -> dict[str, Any]:
    """fastembed's description of a sentence-transformers folder (no torch needed)."""
    modules = json.loads((model_dir / "modules.json").read_text(encoding="utf-8"))
    pooling_module = next(m for m in modules if m["type"].endswith("Pooling"))
    pooling = json.loads(
        (model_dir / pooling_module["path"] / "config.json").read_text(encoding="utf-8")
    )
    mode = pooling["pooling_mode"]
    if mode not in _POOLING:
        raise ValueError(f"pooling {mode!r} can't be served by fastembed ({', '.join(_POOLING)})")
    return {
        "dim": int(pooling["embedding_dimension"]),
        "pooling": _POOLING[mode],
        "normalization": any(m["type"].endswith("Normalize") for m in modules),
        "model_file": MODEL_FILE,
    }


def export_onnx(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Write `out_dir` (model.onnx, tokenizer, config, fastembed.json); returns the spec."""
    import torch
    from sentence_transformers import SentenceTransformer

    spec = fastembed_spec(model_dir)
    model = SentenceTransformer(str(model_dir), device="cpu")
    transformer = model[0].auto_model.eval()

    class TokenVectors(torch.nn.Module):
        """The transformer with plain tensor inputs and one output, as fastembed expects."""

        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any) -> Any:
            return self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    example = model.tokenizer(
        ["an example sentence", "a second, longer example sentence for the export"],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: example[k] for k in ("input_ids", "attention_mask", "token_type_ids")}
    batch, sequence = torch.export.Dim("batch"), torch.export.Dim("sequence", max=512)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            TokenVectors().eval(),  # no dropout in the exported graph
            (),
            str(out_dir / MODEL_FILE),
            kwargs=inputs,
            input_names=list(inputs),
            output_names=["last_hidden_state"],
            dynamic_shapes={name: {0: batch, 1: sequence} for name in inputs},
            external_data=False,  # one self-contained file (130 MB, far below the 2 GB limit)
            dynamo=True,
        )
    model.tokenizer.save_pretrained(str(out_dir))
    transformer.config.save_pretrained(str(out_dir))
    (out_dir / FASTEMBED_SPEC).write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a fine-tuned embedder to ONNX.")
    parser.add_argument("model_dir", type=Path, help="a sentence-transformers folder")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <model_dir>-onnx")
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.model_dir.with_name(args.model_dir.name + "-onnx")
    spec = export_onnx(args.model_dir, out_dir)
    print(f"{out_dir}: {json.dumps(spec)}")
    print(f"Serve it with EMBEDDING_MODEL={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
