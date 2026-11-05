"""Write the LLM and Langfuse keys from `.env` to Key Vault without showing them.

    uv run python scripts/azure/set_secrets.py --vault NAME
    uv run python scripts/azure/set_secrets.py --dry-run

Each value reaches `az` through an owner-only temp file, deleted straight after, so it is never
on a command line, and `az`'s output is off because `az keyvault secret set` returns the value.
Only names and SHA-256 fingerprints are printed: compare them to check two keys match.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import SecretStr, ValidationError

from rig_assist.config import Settings

Run = Callable[..., subprocess.CompletedProcess[str]]


class MissingSecrets(Exception):
    """A value the API references is not set in `.env`."""


def _plain(value: SecretStr | str | None) -> str:
    return value.get_secret_value() if isinstance(value, SecretStr) else value or ""


def collect(settings: Settings) -> dict[str, str]:
    """The Key Vault name and value of every secret; the API's revision needs all of them."""
    secrets = {
        "llm-api-key": _plain(settings.llm_api_key),
        "langfuse-public-key": _plain(settings.langfuse_public_key),
        "langfuse-secret-key": _plain(settings.langfuse_secret_key),
    }
    missing = [name for name, value in secrets.items() if not value]
    if missing:
        raise MissingSecrets(f"not set in .env: {', '.join(missing)}")
    return secrets


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def write_secret(vault: str, name: str, value: str, run: Run = subprocess.run) -> None:
    fd, tmp = tempfile.mkstemp(prefix="secret-")  # mkstemp makes it owner-only
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(value)
        command = ["az", "keyvault", "secret", "set", "--vault-name", vault, "--name", name]
        command += ["--file", tmp, "--output", "none"]
        run(command, check=True, capture_output=True, text=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main(
    argv: list[str] | None = None,
    load: Callable[[], Settings] = Settings,
    run: Run = subprocess.run,
) -> int:
    parser = argparse.ArgumentParser(description="Write the keys from .env to Key Vault.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--vault", help="Key Vault name")
    target.add_argument("--dry-run", action="store_true", help="print fingerprints, write nothing")
    args = parser.parse_args(argv)
    try:
        secrets = collect(load())
    except MissingSecrets as error:
        print(f"nothing written, {error}", file=sys.stderr)
        return 1
    except ValidationError as error:  # str(error) would include the values from .env
        for problem in error.errors():
            print(f"nothing written, {problem['msg']}", file=sys.stderr)
        return 1
    for name, value in secrets.items():
        if not args.dry_run:
            try:
                write_secret(args.vault, name, value, run=run)
            except subprocess.CalledProcessError as error:
                print(f"{name} failed: {error.stderr}", file=sys.stderr)
                return 1
        print(f"{name} {fingerprint(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
