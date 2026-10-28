"""Blob fetch for the Azure init containers, with an in-memory fake Blob source (no network)."""

import hashlib
from pathlib import Path

import pytest
from azure.core.exceptions import ClientAuthenticationError

from rig_assist.deploy.blob_fetch import (
    BlobInfo,
    FetchError,
    FetchReport,
    fetch_prefix,
    main,
)


class FakeSource:
    """Blobs as name -> bytes; records downloads. `md5=False` drops the MD5, like some uploads."""

    def __init__(self, blobs: dict[str, bytes], md5: bool = True, error: Exception | None = None):
        self.blobs = blobs
        self.md5 = md5
        self.error = error
        self.downloaded: list[str] = []

    def list(self, prefix: str) -> list[BlobInfo]:
        if self.error:
            raise self.error
        return [
            BlobInfo(name, len(data), hashlib.md5(data).digest() if self.md5 else None)
            for name, data in sorted(self.blobs.items())
            if name.startswith(prefix)
        ]

    def download(self, name: str, dest: Path) -> None:
        self.downloaded.append(name)
        dest.write_bytes(self.blobs[name])


BLOBS = {
    "index/chunks/a.json": b'{"a": 1}',
    "index/bm25.pkl": b"bm25-bytes",
    "raw/x.pdf": b"%PDF-1.7",
}


def test_blobs_under_the_prefix_land_under_dest_keeping_subfolders(tmp_path: Path) -> None:
    report = fetch_prefix(FakeSource(BLOBS), "index/", tmp_path / "index")

    assert report == FetchReport(downloaded=2, skipped=0)
    assert (tmp_path / "index/chunks/a.json").read_bytes() == b'{"a": 1}'
    assert (tmp_path / "index/bm25.pkl").read_bytes() == b"bm25-bytes"
    assert not (tmp_path / "index/x.pdf").exists()


def test_an_unchanged_file_is_skipped(tmp_path: Path) -> None:
    fetch_prefix(FakeSource(BLOBS), "index/", tmp_path)
    source = FakeSource(BLOBS)

    report = fetch_prefix(source, "index/", tmp_path)

    assert report == FetchReport(downloaded=0, skipped=2)
    assert source.downloaded == []


def test_a_changed_blob_is_downloaded_again(tmp_path: Path) -> None:
    fetch_prefix(FakeSource(BLOBS), "index/", tmp_path)
    (tmp_path / "bm25.pkl").write_bytes(b"bm25-BYTES")  # same size, different content

    report = fetch_prefix(FakeSource(BLOBS), "index/", tmp_path)

    assert report == FetchReport(downloaded=1, skipped=1)
    assert (tmp_path / "bm25.pkl").read_bytes() == b"bm25-bytes"


def test_a_blob_without_md5_is_always_downloaded(tmp_path: Path) -> None:
    fetch_prefix(FakeSource(BLOBS), "index/", tmp_path)

    report = fetch_prefix(FakeSource(BLOBS, md5=False), "index/", tmp_path)

    assert report == FetchReport(downloaded=2, skipped=0)


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    fetch_prefix(FakeSource(BLOBS), "index/", tmp_path)

    names = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert names == ["bm25.pkl", "chunks/a.json"]


def test_an_empty_prefix_raises_fetch_error_naming_it(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="qdrant/"):
        fetch_prefix(FakeSource(BLOBS), "qdrant/", tmp_path)


def run_cli(source: FakeSource, *maps: str) -> int:
    argv = ["--account-url", "https://acct.blob.core.windows.net", "--container", "data"]
    for m in maps:
        argv += ["--map", m]
    return main(argv, source_factory=lambda account_url, container: source)


def test_the_cli_reports_each_map(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run_cli(FakeSource(BLOBS), f"index/={tmp_path}/index", f"raw/={tmp_path}/raw")

    out = capsys.readouterr().out
    assert code == 0
    assert f"index/ → {tmp_path}/index: 2 downloaded, 0 skipped" in out
    assert f"raw/ → {tmp_path}/raw: 1 downloaded, 0 skipped" in out


def test_an_empty_prefix_exits_1_with_the_prefix_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(FakeSource(BLOBS), f"qdrant/={tmp_path}")

    err = capsys.readouterr().err
    assert code == 1
    assert "qdrant/" in err and "Traceback" not in err


def test_an_auth_error_exits_1_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    denied = FakeSource(BLOBS, error=ClientAuthenticationError("no managed identity"))

    code = run_cli(denied, f"index/={tmp_path}")

    err = capsys.readouterr().err
    assert code == 1
    assert "no managed identity" in err and "Traceback" not in err


def test_a_map_without_equals_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_:
        run_cli(FakeSource(BLOBS), "index/")
    assert exit_.value.code == 2
