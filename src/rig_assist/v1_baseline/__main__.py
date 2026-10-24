"""`python -m rig_assist.v1_baseline` (what `make ingest-v1` runs).

Builds the v1 baseline indexes from every PDF in DATA_RAW_DIR, each configuration into its
own store folder and Qdrant collection. `v1_retrieval_on_v2_pages` parses with v2's parser
and the OCR settings from .env, like `make ingest`: use the same OCR_MODE for both so
the two compare like for like (pages already in the OCR cache cost nothing).
"""

import argparse
import logging
import sys
from pathlib import Path

from rig_assist.config import get_settings
from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.models import ParsedPage
from rig_assist.ingestion.parsers.olmocr import OcrFailedError
from rig_assist.ingestion.parsing import parse_document
from rig_assist.ingestion.pipeline import make_ocr_client
from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.dense import make_dense_index
from rig_assist.v1_baseline.pipeline import V1_CONFIGS, build_v1_index, v1_collection, v1_store_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the v1 baseline indexes.")
    parser.add_argument(
        "--config",
        choices=V1_CONFIGS,
        action="append",
        help="configuration to build (repeatable); default: all",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ocr_client = make_ocr_client(settings)
    cache = PageCache(settings.data_parsed_dir)

    def parse_v2(path: Path, doc_id: str) -> list[ParsedPage]:
        return parse_document(
            path,
            doc_id,
            settings.ocr_mode,
            ocr_client,
            cache,
            ocr_concurrency=settings.ocr_concurrency,
        ).pages

    for config in args.config or V1_CONFIGS:
        collection = v1_collection(settings.qdrant_collection, config)
        dense = make_dense_index(settings.model_copy(update={"qdrant_collection": collection}))
        try:
            dense.client.get_collections()  # fail now, not after minutes of parsing
        except Exception as error:  # connection refused, DNS, auth...
            print(
                f"Can't reach Qdrant at {settings.qdrant_url}: {error}\n"
                "Start it with `docker compose up -d qdrant`.",
                file=sys.stderr,
            )
            return 1
        store_dir = v1_store_dir(settings.data_index_dir, config)
        print(f"\n== {config} → {store_dir} + Qdrant '{collection}'", flush=True)
        try:
            infos = build_v1_index(
                settings.data_raw_dir, config, IndexStore(store_dir), dense, parse_v2=parse_v2
            )
        except (OcrFailedError, ValueError, FileNotFoundError) as error:
            print(f"Building {config} failed: {error}", file=sys.stderr)
            return 1
        print(f"{'doc_id':48} {'pages':>5} {'chunks':>6} {'no text':>7}")
        for info in infos:
            print(f"{info.doc_id:48} {info.pages:5} {info.chunks:6} {len(info.missing_pages):7}")
        print(f"{len(infos)} documents, {sum(i.chunks for i in infos)} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
