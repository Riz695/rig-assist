"""`python -m rig_assist.ingestion` (what `make ingest` runs).

Ingests every PDF in DATA_RAW_DIR into both indexes (the chunk store in DATA_INDEX_DIR
for BM25, and Qdrant for dense search), using OCR_MODE and the OCR settings from the
environment / .env, and prints a summary table. --skip-dense = BM25 only (no Qdrant).
--variant builds an index variant instead (see `rig_assist.ingestion.variants`), from the
OCR cache only, into its own store folder and Qdrant collection.
"""

import argparse
import logging
import sys

from rig_assist.config import get_settings
from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.parsers.olmocr import OcrFailedError
from rig_assist.ingestion.pipeline import ingest_directory, make_ocr_client
from rig_assist.ingestion.store import IndexStore
from rig_assist.ingestion.variants import (
    INDEX_VARIANTS,
    VARIANT_OCR_MODE,
    CacheOnlyOcrClient,
    variant_collection,
    variant_store_dir,
)
from rig_assist.retrieval.dense import DenseIndex, make_dense_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the PDF manuals into the indexes.")
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="BM25 index only: don't embed chunks into Qdrant (no Qdrant needed)",
    )
    parser.add_argument(
        "--variant",
        choices=INDEX_VARIANTS,
        help="build this index variant (parser comparison) instead of the main index",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    index_dir, ocr_mode, ocr_client = (
        settings.data_index_dir,
        settings.ocr_mode,
        make_ocr_client(settings),
    )
    if args.variant:
        index_dir = variant_store_dir(settings.data_index_dir, args.variant)
        ocr_mode, ocr_client = VARIANT_OCR_MODE[args.variant], CacheOnlyOcrClient()
        collection = variant_collection(settings.qdrant_collection, args.variant)
        settings = settings.model_copy(update={"qdrant_collection": collection})
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    # huggingface_hub checks the Hub for model updates on every run (a dozen requests even
    # when the model is cached); its HTTP client logs each one at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    dense: DenseIndex | None = None
    if not args.skip_dense:
        dense = make_dense_index(settings)
        try:
            dense.client.get_collections()  # fail now, not after a minute of parsing
        except Exception as error:  # connection refused, DNS, auth...
            print(
                f"Can't reach Qdrant at {settings.qdrant_url}: {error}\n"
                "Start it with `docker compose up -d qdrant`, or run with --skip-dense.",
                file=sys.stderr,
            )
            return 1

    store = IndexStore(index_dir)
    try:
        infos = ingest_directory(
            settings.data_raw_dir,
            store,
            ocr_mode=ocr_mode,
            ocr_client=ocr_client,
            cache=PageCache(settings.data_parsed_dir),
            ocr_concurrency=settings.ocr_concurrency,
            dense=dense,
        )
    except (OcrFailedError, ValueError, FileNotFoundError) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    print(f"\n{'doc_id':48} {'pages':>5} {'chunks':>6} {'no text':>7}")
    for info in infos:
        print(f"{info.doc_id:48} {info.pages:5} {info.chunks:6} {len(info.missing_pages):7}")
    print(
        f"{len(infos)} documents, {sum(i.chunks for i in infos)} chunks → {index_dir}"
        + ("" if dense is None else f" + Qdrant '{settings.qdrant_collection}'")
    )
    if any(info.missing_pages for info in infos) and ocr_mode == "off":
        print("Some pages have no text layer and OCR_MODE=off: they are not searchable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
