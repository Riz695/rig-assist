"""Parsed-page cache: OCR each page once, even across re-ingestions and crashes.

Key = (SHA-256 of the PDF's bytes, page, parser). Hashing the content, not the file
name, means a renamed file still hits the cache and an edited file doesn't.
Layout: <root>/<parser>/<file hash>/<page>.json, one small readable JSON file per page.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from rig_assist.ingestion.files import write_text_atomic
from rig_assist.ingestion.parsers.olmocr import OcrResult


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):  # 1 MiB at a time
            digest.update(block)
    return digest.hexdigest()


class PageCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, file_hash: str, page: int, parser: str) -> Path:
        return self.root / parser / file_hash / f"{page:04d}.json"

    def get(self, file_hash: str, page: int, parser: str) -> OcrResult | None:
        path = self._path(file_hash, page, parser)
        if not path.exists():
            return None
        return OcrResult(**json.loads(path.read_text(encoding="utf-8")))

    def put(self, file_hash: str, page: int, parser: str, result: OcrResult) -> None:
        # Atomic: a crash mid-write never leaves a half-written entry that would later
        # be read as a valid (truncated) page.
        write_text_atomic(
            self._path(file_hash, page, parser), json.dumps(asdict(result), ensure_ascii=False)
        )
