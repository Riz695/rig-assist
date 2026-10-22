from pathlib import Path

from rig_assist.ingestion.cache import PageCache, file_sha256
from rig_assist.ingestion.parsers.olmocr import OcrResult

RESULT = OcrResult(text="# Mooring\nCheck anchor chains.", input_tokens=1900, output_tokens=600)


def test_file_sha256_depends_on_content_not_name(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    c.write_bytes(b"other bytes")
    assert file_sha256(a) == file_sha256(b)
    assert file_sha256(a) != file_sha256(c)
    assert len(file_sha256(a)) == 64


def test_get_returns_none_when_nothing_is_cached(tmp_path: Path) -> None:
    assert PageCache(tmp_path).get("abc", page=1, parser="olmocr") is None


def test_put_then_get_returns_the_same_result(tmp_path: Path) -> None:
    cache = PageCache(tmp_path)
    cache.put("abc", page=3, parser="olmocr", result=RESULT)
    assert cache.get("abc", page=3, parser="olmocr") == RESULT


def test_entries_are_keyed_by_file_hash_page_and_parser(tmp_path: Path) -> None:
    cache = PageCache(tmp_path)
    cache.put("abc", page=3, parser="olmocr", result=RESULT)
    assert cache.get("abc", page=4, parser="olmocr") is None
    assert cache.get("xyz", page=3, parser="olmocr") is None
    assert cache.get("abc", page=3, parser="other") is None


def test_cache_survives_a_restart(tmp_path: Path) -> None:
    PageCache(tmp_path).put("abc", page=1, parser="olmocr", result=RESULT)
    assert PageCache(tmp_path).get("abc", page=1, parser="olmocr") == RESULT


def test_cache_files_are_readable_json_and_leave_no_temp_files(tmp_path: Path) -> None:
    PageCache(tmp_path).put("abc", page=1, parser="olmocr", result=RESULT)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1
    assert files[0].suffix == ".json"
    assert "Check anchor chains." in files[0].read_text(encoding="utf-8")
