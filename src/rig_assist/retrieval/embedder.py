"""Text → vector. The real model runs on CPU through fastembed (ONNX, no PyTorch).

`BAAI/bge-small-en-v1.5` in fastembed 0.9: 384 dimensions, a 67 MB fp16 ONNX file
downloaded from Hugging Face on first use, max 512 input tokens (longer text is cut).

EMBEDDING_MODEL can also be a folder made by `training/export_onnx.py` (the fine-tuned
model): `model.onnx`, the tokenizer files, and `fastembed.json` saying how to turn token
vectors into one vector (pooling, normalization). It's registered with fastembed as a
custom model and loaded from the folder: same runtime, no PyTorch in the API.
"""

import hashlib
import json
from pathlib import Path
from typing import Protocol

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

FASTEMBED_SPEC = "fastembed.json"  # in an exported folder: dim, pooling, normalization, file

# BAAI's suggested instruction for short queries that search longer passages (bge v1.5).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def load_text_embedding(model_name: str, cache_dir: str | None = None) -> TextEmbedding:
    """A fastembed model by name, or an exported model from a folder."""
    folder = Path(model_name)
    if not folder.is_dir():
        return TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    spec_path = folder / FASTEMBED_SPEC
    if not spec_path.exists():
        raise ValueError(
            f"{folder} has no {FASTEMBED_SPEC}: export it with training/export_onnx.py"
        )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    folder = folder.resolve()
    path_hash = hashlib.sha256(str(folder).encode()).hexdigest()[:8]
    name = f"local/{folder.name}-{path_hash}"  # unique per folder
    registered = {m["model"].lower() for m in TextEmbedding.list_supported_models()}
    if name.lower() not in registered:
        TextEmbedding.add_custom_model(
            model=name,
            pooling=PoolingType(spec["pooling"]),
            normalization=bool(spec["normalization"]),
            # fastembed wants a source; `specific_model_path` below means it's never used.
            sources=ModelSource(hf=name),
            dim=int(spec["dim"]),
            model_file=spec["model_file"],
        )
    return TextEmbedding(model_name=name, cache_dir=cache_dir, specific_model_path=str(folder))


class FastEmbedEmbedder:
    """fastembed model, loaded on first use (the app starts without downloading it).

    `query_prefix` is prepended to queries only. BAAI suggests "Represent this sentence
    for searching relevant passages: " for retrieval; v1 used none, and fastembed adds
    none, so the default is "" and Phase 3 measures whether the prefix helps.
    """

    def __init__(
        self, model_name: str, query_prefix: str = "", cache_dir: str | None = None
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        self._dim: int | None = None

    def _load(self) -> TextEmbedding:
        if self._model is None:
            self._model = load_text_embedding(self.model_name, self.cache_dir)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_passages(["dimension probe"])[0])
        return self._dim

    def unload(self) -> None:
        """Free the model's memory; the next call reloads it."""
        self._model = None

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._load().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed(self.query_prefix + text))).tolist()
