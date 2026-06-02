from pathlib import Path

import tomli

_DEFAULT_PARAMS = Path(__file__).parent / "params.toml"


def load(path: Path = _DEFAULT_PARAMS) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)
