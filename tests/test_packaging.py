import tomllib
from pathlib import Path

import tiny_pytorch as tp


def test_import_version_matches_project_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    assert tp.__version__ == project["version"]
