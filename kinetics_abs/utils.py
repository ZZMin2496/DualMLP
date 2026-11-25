from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_SEQUENCE_CHARS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
SEQUENCE_WHITESPACE = re.compile(r"\s+")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_yaml_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_sequence(sequence: Any) -> tuple[str, str]:
    if sequence is None:
        return "", "missing_sequence"
    cleaned = SEQUENCE_WHITESPACE.sub("", str(sequence)).upper()
    if not cleaned:
        return "", "empty_sequence"
    invalid = sorted(set(cleaned) - ALLOWED_SEQUENCE_CHARS)
    if invalid:
        return cleaned, "invalid_residues:" + ",".join(invalid)
    return cleaned, "ok"


def sequence_id(sequence: str) -> str:
    return "seq_" + sha256_text(sequence)[:24]


def chemical_id(smiles: str, scope: str = "") -> str:
    payload = {"scope": scope, "smiles": smiles}
    return "chem_" + sha256_json(payload)[:24]


def reaction_id(reaction_smiles: str) -> str:
    return "rxn_" + sha256_text(reaction_smiles)[:24]


def sample_id(task: str, source_file_sha256: str, row_index: int) -> str:
    return f"{task}_{sha256_text(f'{source_file_sha256}:{row_index}')[:20]}"


def split_reaction_smiles(reaction_smiles: Any) -> dict[str, Any]:
    text = "" if reaction_smiles is None else str(reaction_smiles).strip()
    if not text:
        return {"status": "missing", "reactants": "", "products": "", "reaction_smiles": ""}
    if text.count(">>") != 1:
        return {"status": "invalid_separator", "reactants": "", "products": "", "reaction_smiles": text}
    left, right = text.split(">>", 1)
    left, right = left.strip(), right.strip()
    status = "ok"
    if not left or not right:
        status = "incomplete_side"
    return {"status": status, "reactants": left, "products": right, "reaction_smiles": text}


def finite_or_none(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator
