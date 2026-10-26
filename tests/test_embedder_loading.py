"""EMBEDDING_MODEL as a fastembed model name or as a folder holding an exported model.

fastembed's TextEmbedding is replaced by a fake that records how it was registered and
loaded: no model file is read and nothing is downloaded.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rig_assist.config import Settings
from rig_assist.retrieval import embedder as embedder_module
from rig_assist.retrieval.dense import make_dense_index
from rig_assist.retrieval.embedder import (
    FASTEMBED_SPEC,
    FastEmbedEmbedder,
    load_text_embedding,
)


class FakeTextEmbedding:
    """Mimics TextEmbedding's class-level registry of custom models."""

    custom: list[dict[str, Any]] = []
    loads: list[dict[str, Any]] = []

    def __init__(self, model_name: str, cache_dir: str | None = None, **kwargs: Any) -> None:
        type(self).loads.append({"model_name": model_name, "cache_dir": cache_dir, **kwargs})

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        return [{"model": "BAAI/bge-small-en-v1.5"}, *({"model": c["model"]} for c in cls.custom)]

    @classmethod
    def add_custom_model(cls, model: str, **kwargs: Any) -> None:
        if any(m["model"].lower() == model.lower() for m in cls.list_supported_models()):
            raise ValueError(f"Model {model} is already registered")
        cls.custom.append({"model": model, **kwargs})

    def query_embed(self, text: str) -> Iterator[np.ndarray]:
        yield np.zeros(3)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type[FakeTextEmbedding]:
    FakeTextEmbedding.custom = []
    FakeTextEmbedding.loads = []
    monkeypatch.setattr(embedder_module, "TextEmbedding", FakeTextEmbedding)
    return FakeTextEmbedding


def exported(tmp_path: Path, **spec: Any) -> Path:
    folder = tmp_path / "ft-model"
    folder.mkdir()
    values = {"dim": 384, "pooling": "CLS", "normalization": True, "model_file": "model.onnx"}
    (folder / FASTEMBED_SPEC).write_text(json.dumps(values | spec))
    return folder


def test_a_model_name_loads_the_fastembed_model(fake: type[FakeTextEmbedding]) -> None:
    load_text_embedding("BAAI/bge-small-en-v1.5", cache_dir="data/models")
    assert fake.custom == []
    assert fake.loads == [{"model_name": "BAAI/bge-small-en-v1.5", "cache_dir": "data/models"}]


def test_a_folder_registers_a_custom_model_from_its_spec_and_loads_the_folder(
    fake: type[FakeTextEmbedding], tmp_path: Path
) -> None:
    folder = exported(tmp_path)
    load_text_embedding(str(folder), cache_dir="data/models")

    [custom] = fake.custom
    assert custom["dim"] == 384
    assert custom["pooling"] == embedder_module.PoolingType.CLS
    assert custom["normalization"] is True
    assert custom["model_file"] == "model.onnx"
    [load] = fake.loads
    assert load["model_name"] == custom["model"]
    assert load["specific_model_path"] == str(folder.resolve())


def test_loading_the_same_folder_twice_registers_it_once(
    fake: type[FakeTextEmbedding], tmp_path: Path
) -> None:
    folder = exported(tmp_path)
    load_text_embedding(str(folder))
    load_text_embedding(str(folder))
    assert len(fake.custom) == 1
    assert len(fake.loads) == 2


def test_the_spec_sets_pooling_and_normalization(
    fake: type[FakeTextEmbedding], tmp_path: Path
) -> None:
    load_text_embedding(str(exported(tmp_path, pooling="MEAN", normalization=False)))
    assert fake.custom[0]["pooling"] == embedder_module.PoolingType.MEAN
    assert fake.custom[0]["normalization"] is False


def test_a_folder_without_a_spec_is_an_error(fake: type[FakeTextEmbedding], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=FASTEMBED_SPEC):
        load_text_embedding(str(tmp_path))


def test_the_embedder_loads_through_load_text_embedding(
    fake: type[FakeTextEmbedding], tmp_path: Path
) -> None:
    folder = exported(tmp_path)
    embedder = FastEmbedEmbedder(str(folder), cache_dir="data/models")
    assert embedder.embed_query("pump") == [0.0, 0.0, 0.0]
    assert fake.loads[0]["specific_model_path"] == str(folder.resolve())


def test_embedding_model_setting_switches_the_served_dense_index_to_the_folder(
    fake: type[FakeTextEmbedding], tmp_path: Path
) -> None:
    folder = exported(tmp_path)
    settings = Settings(_env_file=None, embedding_model=str(folder))  # type: ignore[call-arg]
    embedder = make_dense_index(settings).embedder
    assert isinstance(embedder, FastEmbedEmbedder)
    embedder.embed_query("pump")
    assert fake.loads[0]["specific_model_path"] == str(folder.resolve())
