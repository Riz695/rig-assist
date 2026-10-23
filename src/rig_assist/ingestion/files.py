"""Small file helpers shared by the page cache and the index store."""

import os
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` so readers see either the old file or the new one, never half.

    Writes a temp file next to the target, then renames it over the target; a rename
    within one filesystem is atomic. A crash mid-write leaves only a stray .tmp file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
