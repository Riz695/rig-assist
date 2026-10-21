import rig_assist


def test_version_matches_pyproject() -> None:
    assert rig_assist.__version__ == "0.1.0"
