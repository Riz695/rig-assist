"""Text → vector. The real model runs on CPU through fastembed (ONNX, no PyTorch).

`BAAI/bge-small-en-v1.5` in fastembed 0.9: 384 dimensions, a 67 MB quantized ONNX file
downloaded from Hugging Face on first use, max 512 input tokens (longer text is cut).
"""

from typing import Protocol

from fastembed import TextEmbedding

# BAAI's suggested instruction for short queries that search longer passages (bge v1.5).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


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
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_passages(["dimension probe"])[0])
        return self._dim

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._load().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed(self.query_prefix + text))).tolist()
