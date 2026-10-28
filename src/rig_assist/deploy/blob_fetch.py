"""Copy folders from Azure Blob Storage into the container before the app starts.

On Azure Container Apps, the `api` and `qdrant` apps have init containers that run

    python -m rig_assist.deploy.blob_fetch --account-url URL --container data \\
        --map index/=/app/data/index --map raw/=/app/data/raw

signing in with the app's managed identity (no keys). A file already there with the same
size and MD5 is skipped; anything else is downloaded to a temp file and renamed into place,
so an interrupted copy never leaves a half file under the real name. A missing folder or a
sign-in failure exits 1 with one line, so the revision fails visibly instead of the API
starting with an empty index.
"""

import argparse
import hashlib
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from azure.core.exceptions import AzureError


@dataclass(frozen=True)
class BlobInfo:
    name: str
    size: int
    md5: bytes | None  # the blob's Content-MD5; None when the upload didn't set one


@dataclass(frozen=True)
class FetchReport:
    downloaded: int
    skipped: int


class FetchError(Exception):
    """Nothing to fetch where something was expected."""


class BlobSource(Protocol):
    def list(self, prefix: str) -> list[BlobInfo]: ...

    def download(self, name: str, dest: Path) -> None: ...


class AzureBlobSource:
    """One Blob container, signed in with `DefaultAzureCredential`: the managed identity
    named by `AZURE_CLIENT_ID` on Azure, your `az login` on a laptop."""

    def __init__(self, account_url: str, container: str) -> None:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient

        credential = DefaultAzureCredential(
            managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID")
        )
        self.client = ContainerClient(account_url, container, credential=credential)

    def list(self, prefix: str) -> list[BlobInfo]:
        blobs = []
        for blob in self.client.list_blobs(name_starts_with=prefix):
            md5 = blob.content_settings.content_md5
            blobs.append(BlobInfo(blob.name, blob.size, bytes(md5) if md5 else None))
        return blobs

    def download(self, name: str, dest: Path) -> None:
        with dest.open("wb") as file:
            self.client.download_blob(name).readinto(file)


def _md5(path: Path) -> bytes:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.digest()


def _unchanged(blob: BlobInfo, path: Path) -> bool:
    if blob.md5 is None or not path.is_file() or path.stat().st_size != blob.size:
        return False
    return _md5(path) == blob.md5


def fetch_prefix(source: BlobSource, prefix: str, dest: Path) -> FetchReport:
    """Copy every blob under `prefix` to `dest`, keeping the path below the prefix."""
    blobs = source.list(prefix)
    if not blobs:
        raise FetchError(f"no blobs under '{prefix}': publish the data first")
    downloaded = skipped = 0
    for blob in blobs:
        path = dest / blob.name.removeprefix(prefix)
        if _unchanged(blob, path):
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(fd)
        try:
            source.download(blob.name, Path(tmp))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        downloaded += 1
    return FetchReport(downloaded, skipped)


def _mapping(text: str) -> tuple[str, Path]:
    prefix, sep, dest = text.partition("=")
    if not sep or not prefix or not dest:
        raise argparse.ArgumentTypeError(f"expected PREFIX=DEST, got {text!r}")
    return prefix, Path(dest)


SourceFactory = Callable[[str, str], BlobSource]


def main(argv: list[str] | None = None, source_factory: SourceFactory = AzureBlobSource) -> int:
    parser = argparse.ArgumentParser(description="Copy Blob folders into the container.")
    parser.add_argument("--account-url", required=True, help="https://<account>.blob.core...")
    parser.add_argument("--container", required=True)
    parser.add_argument(
        "--map", action="append", type=_mapping, required=True, metavar="PREFIX=DEST"
    )
    args = parser.parse_args(argv)
    try:
        source = source_factory(args.account_url, args.container)
        for prefix, dest in args.map:
            report = fetch_prefix(source, prefix, dest)
            print(f"{prefix} → {dest}: {report.downloaded} downloaded, {report.skipped} skipped")
    except (FetchError, AzureError) as error:
        print(f"blob fetch failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
