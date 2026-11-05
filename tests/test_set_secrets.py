"""Writing keys to Key Vault: every value through a temp file, none ever printed (fake `az`)."""

import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr
from scripts.azure.set_secrets import MissingSecrets, collect, fingerprint, main, write_secret

from rig_assist.config import Settings

LLM_KEY = "llm-value-do-not-print"
PUBLIC_KEY = "pk-lf-value-do-not-print"
SECRET_KEY = "sk-lf-value-do-not-print"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_api_key": SecretStr(LLM_KEY),
        "langfuse_public_key": PUBLIC_KEY,
        "langfuse_secret_key": SecretStr(SECRET_KEY),
    }
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[call-arg, arg-type]


class FakeAz:
    """Stands in for `subprocess.run`: records each command and the file it was pointed at."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.commands: list[list[str]] = []
        self.files: dict[str, str] = {}
        self.modes: list[int] = []
        self.paths: list[Path] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        path = Path(command[command.index("--file") + 1])
        self.paths.append(path)
        self.files[command[command.index("--name") + 1]] = path.read_text()
        self.modes.append(path.stat().st_mode & 0o777)
        if self.returncode:
            raise subprocess.CalledProcessError(self.returncode, command, "", self.stderr)
        return subprocess.CompletedProcess(command, 0, "", self.stderr)


def test_collect_maps_settings_to_the_key_vault_names() -> None:
    assert collect(settings()) == {
        "llm-api-key": LLM_KEY,
        "langfuse-public-key": PUBLIC_KEY,
        "langfuse-secret-key": SECRET_KEY,
    }


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_a_missing_llm_key_is_refused_by_name(key: SecretStr | None) -> None:
    # The API references all three, so a missing one would fail its revision later and less clearly.
    with pytest.raises(MissingSecrets, match="llm-api-key"):
        collect(settings(llm_api_key=key))


def test_missing_langfuse_keys_are_refused_by_name() -> None:
    with pytest.raises(MissingSecrets, match="langfuse-public-key, langfuse-secret-key"):
        collect(settings(langfuse_public_key=None, langfuse_secret_key=None))


def test_the_fingerprint_is_8_hex_of_the_sha256() -> None:
    assert fingerprint("abc") == "ba7816bf"


def test_the_value_goes_through_an_owner_only_file_never_the_command_line() -> None:
    az = FakeAz()

    write_secret("vault", "llm-api-key", LLM_KEY, run=az)

    assert az.files == {"llm-api-key": LLM_KEY}
    assert az.modes == [0o600]
    assert LLM_KEY not in " ".join(az.commands[0])


def test_az_output_is_switched_off() -> None:
    # `az keyvault secret set` returns the secret, value included, unless told otherwise.
    az = FakeAz()

    write_secret("vault", "llm-api-key", LLM_KEY, run=az)

    command = az.commands[0]
    assert command[command.index("--output") + 1] == "none"


def test_the_temp_file_is_deleted_after() -> None:
    az = FakeAz()

    write_secret("vault", "llm-api-key", LLM_KEY, run=az)

    assert not az.paths[0].exists()


def test_the_temp_file_is_deleted_when_az_fails() -> None:
    az = FakeAz(returncode=1, stderr="Forbidden")

    with pytest.raises(subprocess.CalledProcessError):
        write_secret("vault", "llm-api-key", LLM_KEY, run=az)

    assert not az.paths[0].exists()


def test_dry_run_prints_names_and_fingerprints_and_calls_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    az = FakeAz()

    code = main(["--dry-run"], load=settings, run=az)

    out = capsys.readouterr().out
    assert code == 0
    assert az.commands == []
    assert f"llm-api-key {fingerprint(LLM_KEY)}" in out
    assert not {LLM_KEY, PUBLIC_KEY, SECRET_KEY} & set(out.split())


def test_writing_needs_a_vault(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        main([], load=settings, run=FakeAz())

    assert exit_.value.code == 2


def test_nothing_is_written_when_one_value_is_missing(capsys: pytest.CaptureFixture[str]) -> None:
    az = FakeAz()

    code = main(["--vault", "v"], load=lambda: settings(llm_api_key=None), run=az)

    assert code == 1
    assert az.commands == []
    assert "llm-api-key" in capsys.readouterr().err


def test_a_settings_error_shows_its_message_but_no_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # pydantic's own error text includes the input, here the half-set Langfuse pair.
    code = main(["--dry-run"], load=lambda: settings(langfuse_secret_key=None), run=FakeAz())

    err = capsys.readouterr().err
    assert code == 1
    assert "Set both Langfuse keys or neither" in err
    assert PUBLIC_KEY not in err and "Traceback" not in err


def test_all_three_are_written_with_no_value_in_the_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    az = FakeAz()

    code = main(["--vault", "v"], load=settings, run=az)

    captured = capsys.readouterr()
    assert code == 0
    assert sorted(az.files) == ["langfuse-public-key", "langfuse-secret-key", "llm-api-key"]
    for value in (LLM_KEY, PUBLIC_KEY, SECRET_KEY):
        assert value not in captured.out + captured.err


def test_an_az_failure_exits_1_naming_the_secret_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--vault", "v"], load=settings, run=FakeAz(returncode=1, stderr="Forbidden"))

    err = capsys.readouterr().err
    assert code == 1
    assert "llm-api-key" in err and "Forbidden" in err and "Traceback" not in err
