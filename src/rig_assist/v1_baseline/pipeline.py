"""Build a v1 configuration's indexes and its retriever.

Each configuration has its own chunk store (`<DATA_INDEX_DIR>/baselines/<config>/`) and
Qdrant collection (`<QDRANT_COLLECTION>__<config>`), so it never touches v2's index.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

from pypdf import PdfReader

from rig_assist.ingestion.cache import file_sha256
from rig_assist.ingestion.models import ParsedPage
from rig_assist.ingestion.pipeline import DenseWriter, make_doc_id, make_title
from rig_assist.ingestion.store import DocumentInfo, IndexStore
from rig_assist.retrieval.bm25 import BM25Index
from rig_assist.retrieval.hybrid import HybridRetriever
from rig_assist.retrieval.models import Retriever
from rig_assist.v1_baseline.chunking import chunk_pages
from rig_assist.v1_baseline.parsing import parse_pdf

V1Config = Literal["v1_baseline", "v1_retrieval_on_v2_pages"]
V1_CONFIGS: tuple[V1Config, ...] = get_args(V1Config)

# (pdf path, doc_id) → parsed pages. For `v1_retrieval_on_v2_pages`: v2's parser.
PageSource = Callable[[Path, str], list[ParsedPage]]


def v1_store_dir(index_dir: Path, config: V1Config) -> Path:
    return index_dir / "baselines" / config


def v1_collection(base_collection: str, config: V1Config) -> str:
    return f"{base_collection}__{config}"


def build_v1_index(
    raw_dir: Path,
    config: V1Config,
    store: IndexStore,
    dense: DenseWriter,
    parse_v2: PageSource | None = None,
) -> list[DocumentInfo]:
    """Parse, chunk (v1's way) and index every *.pdf in `raw_dir`. Re-running replaces."""
    if config not in V1_CONFIGS:
        raise ValueError(f"unknown v1 config {config!r}: use one of {', '.join(V1_CONFIGS)}")
    if config == "v1_retrieval_on_v2_pages" and parse_v2 is None:
        raise ValueError("v1_retrieval_on_v2_pages needs parse_v2 (v2's page parser)")

    infos = []
    for path in sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"):
        doc_id, title = make_doc_id(path.name), make_title(path.name)
        if config == "v1_baseline":
            pages = parse_pdf(path, doc_id)
        else:
            assert parse_v2 is not None  # checked above; narrows the type for mypy
            pages = parse_v2(path, doc_id)
        chunks = chunk_pages(pages, doc_title=title)
        total = len(PdfReader(path).pages)
        with_chunks = {c.page for c in chunks}
        info = DocumentInfo(
            doc_id=doc_id,
            title=title,
            filename=path.name,
            file_sha256=file_sha256(path),
            pages=total,
            chunks=len(chunks),
            missing_pages=[n for n in range(1, total + 1) if n not in with_chunks],
        )
        # Dense first: the store's document list marks "fully indexed" (as in v2).
        dense.upsert_document(doc_id, chunks)
        store.put(info, chunks)
        infos.append(info)
    return infos


def make_v1_retriever(store: IndexStore, dense: Retriever) -> HybridRetriever:
    """v1's retrieval: dense + Okapi BM25, 20 results each, RRF k=60.

    v1 returned the top 5 (`top_n=5`); here k is the caller's, so Phase 3 can also
    compute MRR@10.
    """
    bm25 = BM25Index.build(store.chunks(), idf="okapi")
    return HybridRetriever(dense, bm25, k_each=20, rrf_k=60)
