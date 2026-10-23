"""`python -m rig_assist.ingestion` (what `make ingest` runs).

Ingests every PDF in DATA_RAW_DIR into DATA_INDEX_DIR, using OCR_MODE and the OCR
settings from the environment / .env, and prints a summary table.
"""

import argparse
import logging
import sys

from rig_assist.config import get_settings
from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.parsers.olmocr import OcrFailedError
from rig_assist.ingestion.pipeline import ingest_directory, make_ocr_client
from rig_assist.ingestion.store import IndexStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the PDF manuals into the index.")
    parser.parse_args(argv)  # no options yet; gives --help for free
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

    store = IndexStore(settings.data_index_dir)
    try:
        infos = ingest_directory(
            settings.data_raw_dir,
            store,
            ocr_mode=settings.ocr_mode,
            ocr_client=make_ocr_client(settings),
            cache=PageCache(settings.data_parsed_dir),
            ocr_concurrency=settings.ocr_concurrency,
        )
    except (OcrFailedError, ValueError, FileNotFoundError) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    print(f"\n{'doc_id':48} {'pages':>5} {'chunks':>6} {'no text':>7}")
    for info in infos:
        print(f"{info.doc_id:48} {info.pages:5} {info.chunks:6} {len(info.missing_pages):7}")
    print(
        f"{len(infos)} documents, {sum(i.chunks for i in infos)} chunks → {settings.data_index_dir}"
    )
    if any(info.missing_pages for info in infos) and settings.ocr_mode == "off":
        print("Some pages have no text layer and OCR_MODE=off: they are not searchable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
