"""The index on disk: what was ingested, and every chunk, one pair of files per document.

Layout under the index directory (`DATA_INDEX_DIR`, gitignored):
    documents/<doc_id>.json   DocumentInfo: title, file hash, page/chunk counts, missing pages
    chunks/<doc_id>.jsonl     one Chunk per line

Per-document files let one manual be added or re-ingested without touching the others,
and re-ingesting overwrites the pair, so chunks are never duplicated. Search indexes
(BM25 now, Qdrant in Phase 2) are built from `chunks()`; BM25 rebuilds in milliseconds.
Plain JSON, not pickle: readable, diffable, and safe to load.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rig_assist.ingestion.files import write_text_atomic
from rig_assist.ingestion.models import Chunk

# doc_id becomes a file name, so allow only lowercase slugs: no "/", "..", or spaces.
_DOC_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    title: str
    filename: str
    file_sha256: str
    pages: int  # pages in the PDF
    chunks: int
    missing_pages: list[int]  # pages left without text (scans when OCR is off)


class IndexStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _info_path(self, doc_id: str) -> Path:
        return self.root / "documents" / f"{doc_id}.json"

    def _chunks_path(self, doc_id: str) -> Path:
        return self.root / "chunks" / f"{doc_id}.jsonl"

    def put(self, info: DocumentInfo, chunks: list[Chunk]) -> None:
        """Add a document, or replace it if its doc_id already exists."""
        if not _DOC_ID.match(info.doc_id):
            raise ValueError(
                f"doc_id must be a lowercase slug like 'drilling-manual': {info.doc_id!r}"
            )
        # Chunks first, info last: a document is only listed once its chunks are complete.
        write_text_atomic(
            self._chunks_path(info.doc_id),
            "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks),
        )
        write_text_atomic(self._info_path(info.doc_id), json.dumps(asdict(info), indent=2))

    def documents(self) -> list[DocumentInfo]:
        """Every stored document, sorted by doc_id."""
        folder = self.root / "documents"
        if not folder.exists():
            return []
        return [
            DocumentInfo(**json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(folder.glob("*.json"))
        ]

    def chunks(self) -> list[Chunk]:
        """Every chunk of every stored document, in doc_id then file order."""
        chunks: list[Chunk] = []
        for info in self.documents():
            with self._chunks_path(info.doc_id).open(encoding="utf-8") as f:
                chunks.extend(Chunk(**json.loads(line)) for line in f if line.strip())
        return chunks
