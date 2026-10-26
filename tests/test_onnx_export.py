"""ONNX export of a fine-tuned sentence-transformers model, for serving through fastembed.

`fastembed_spec` reads the model's pooling setup (plain JSON, no torch). The parity test
exports a real model and needs the `train` group: it is marked `slow`, skipped when torch
isn't installed (CI), and run with `make test-slow`.
"""

import json
from pathlib import Path

import pytest

from rig_assist.training.export_onnx import fastembed_spec


def st_model(tmp_path: Path, pooling: str = "cls", normalize: bool = True) -> Path:
    folder = tmp_path / "st"
    (folder / "1_Pooling").mkdir(parents=True)
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "...Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "...Pooling"},
    ]
    if normalize:
        modules.append({"idx": 2, "name": "2", "path": "2_Normalize", "type": "...Normalize"})
    (folder / "modules.json").write_text(json.dumps(modules))
    (folder / "1_Pooling" / "config.json").write_text(
        json.dumps({"embedding_dimension": 384, "pooling_mode": pooling})
    )
    return folder


def test_fastembed_spec_reads_dimension_pooling_and_normalization(tmp_path: Path) -> None:
    assert fastembed_spec(st_model(tmp_path)) == {
        "dim": 384,
        "pooling": "CLS",
        "normalization": True,
        "model_file": "model.onnx",
    }


def test_fastembed_spec_mean_pooling_without_normalize(tmp_path: Path) -> None:
    spec = fastembed_spec(st_model(tmp_path, pooling="mean", normalize=False))
    assert spec["pooling"] == "MEAN"
    assert spec["normalization"] is False


def test_fastembed_spec_rejects_pooling_fastembed_cannot_do(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max"):
        fastembed_spec(st_model(tmp_path, pooling="max"))


@pytest.mark.slow
def test_exported_onnx_matches_the_pytorch_model(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("onnxscript")
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from rig_assist.retrieval.embedder import FastEmbedEmbedder
    from rig_assist.training.export_onnx import export_onnx

    source = tmp_path / "source"
    SentenceTransformer("BAAI/bge-small-en-v1.5").save_pretrained(str(source))
    export_onnx(source, tmp_path / "onnx")

    texts = [
        "The mud pump liner is worn and leaking",
        "Check the accumulator precharge pressure every 14 days before running the BOP stack.",
        "h2s",
    ]
    expected = SentenceTransformer(str(source)).encode(texts, normalize_embeddings=True)
    served = FastEmbedEmbedder(str(tmp_path / "onnx"))
    passages = np.array(served.embed_passages(texts))
    query = np.array(served.embed_query(texts[0]))
    assert np.abs(passages - expected).max() < 1e-4
    assert np.abs(query - expected[0]).max() < 1e-4
