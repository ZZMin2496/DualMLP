from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .chemical_features import CHEM_TOTAL_DIM, chemical_cache_id, chemical_config_hash
from .data_adapters import load_unified_csv
from .protein_features import ESMC_DIM, protein_config_hash
from .reaction_features import RXNFP_DIM, reaction_config_hash
from .structure_features import (
    EDGE_ATTR_DIM,
    EDGE_QUALITY_DIM,
    NODE_QUALITY_DIM,
    structure_config_hash,
    structure_key,
)
from .utils import ensure_dir, project_path, sha256_file, write_json_atomic

FEATURE_INDEX_COLUMNS = [
    "sample_id",
    "task",
    "sequence_id",
    "protein_cache_key",
    "protein_status",
    "protein_length",
    "chemical_id",
    "chemical_cache_key",
    "chemical_status",
    "chemical_scope",
    "reaction_id",
    "reaction_cache_key",
    "reaction_status",
    "reaction_token_length",
    "structure_key",
    "structure_status",
    "structure_ref",
    "coordinate_coverage",
    "n_undirected_edges",
]


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, str]]:
    records = frame.to_dict(orient="records")
    if limit is not None:
        records = records[: int(limit)]
    return records


def _modality_enabled(config: dict[str, Any], name: str) -> bool:
    return bool(config.get("modalities", {}).get(name, True))


def build_feature_caches(
    unified_csv: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    device: str = "cpu",
    limit: int | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    unified_csv = Path(unified_csv)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    frame = load_unified_csv(unified_csv)
    if task:
        frame = frame[frame["task"] == task].copy()
    if limit is not None:
        frame = frame.iloc[: int(limit)].copy()
    records = _records(frame)

    protein_config = config.get("protein", {})
    chemical_config = config.get("chemical", {})
    reaction_config = config.get("reaction", {})
    structure_config = config.get("structure", {})
    cache_root = project_path(config.get("cache_root", "features_cache"))
    # All modality caches share one root; cache keys are prefixed by modality (seq_/chem_/rxn_/struct_).
    protein_root = cache_root
    chemical_root = cache_root
    reaction_root = cache_root
    structure_root = cache_root

    protein_statuses: dict[str, str] = {}
    chemical_statuses: dict[str, str] = {}
    reaction_statuses: dict[str, str] = {}
    structure_statuses: dict[str, str] = {}
    all_failures: dict[str, list[dict[str, Any]]] = {"protein": [], "chemical": [], "reaction": [], "structure": []}

    if _modality_enabled(config, "protein"):
        protein_records = [r for r in records if r.get("protein_available") == "1" and r.get("sequence_id")]
        protein_statuses, all_failures["protein"] = _build_protein(protein_records, protein_root, protein_config, device, limit)
    if _modality_enabled(config, "chemical"):
        chemical_records = [r for r in records if r.get("chem_available") == "1" and r.get("chem_context_smiles")]
        chemical_statuses, all_failures["chemical"] = _build_chemical(chemical_records, chemical_root, chemical_config, device, limit)
    if _modality_enabled(config, "reaction"):
        reaction_records = [r for r in records if r.get("reaction_available") == "1" and r.get("reaction_smiles")]
        reaction_statuses, all_failures["reaction"] = _build_reaction(reaction_records, reaction_root, reaction_config, device, limit)
    if _modality_enabled(config, "structure"):
        structure_records = [r for r in records if r.get("structure_ref") and r.get("sequence_id")]
        roots = [project_path(p) for p in config.get("structure_roots", [])]
        if not roots:
            roots = [project_path(config.get("catpred_root", ""))] if config.get("catpred_root") else []
        structure_statuses, all_failures["structure"] = _build_structure(structure_records, structure_root, structure_config, roots, limit)

    index_rows = []
    for row in tqdm(records, desc="feature-index"):
        seq = row.get("sequence_actual", "")
        seq_id = row.get("sequence_id", "")
        smiles = row.get("chem_context_smiles", "")
        scope = row.get("chemical_scope", "")
        rxn = row.get("reaction_smiles", "")
        ref = row.get("structure_ref", "")
        chemical_id_value, _, _ = chemical_cache_id(smiles, scope) if smiles else ("", "", "")
        reaction_id_value = _reaction_id(rxn) if rxn else ""
        struct_key = structure_key(seq_id, ref) if (ref and seq_id) else ""
        index_rows.append({
            "sample_id": row.get("sample_id", ""),
            "task": row.get("task", ""),
            "sequence_id": seq_id,
            "protein_cache_key": seq_id,
            "protein_status": _status_for(protein_statuses, seq_id, row.get("protein_available") == "1", _modality_enabled(config, "protein"), "sequence_id"),
            "protein_length": str(len(seq)) if seq else "",
            "chemical_id": chemical_id_value,
            "chemical_cache_key": chemical_id_value,
            "chemical_status": _status_for(chemical_statuses, chemical_id_value, row.get("chem_available") == "1" and bool(smiles), _modality_enabled(config, "chemical"), "chemical_id"),
            "chemical_scope": scope,
            "reaction_id": reaction_id_value,
            "reaction_cache_key": reaction_id_value,
            "reaction_status": _status_for(reaction_statuses, reaction_id_value, row.get("reaction_available") == "1" and bool(rxn), _modality_enabled(config, "reaction"), "reaction_id"),
            "reaction_token_length": "",
            "structure_key": struct_key,
            "structure_status": _status_for(structure_statuses, struct_key, bool(ref and seq_id), _modality_enabled(config, "structure"), "structure_key"),
            "structure_ref": ref,
            "coordinate_coverage": "",
            "n_undirected_edges": "",
        })
    index = pd.DataFrame(index_rows, columns=FEATURE_INDEX_COLUMNS)
    index_path = output_dir / "feature_index.csv"
    _write_csv_atomic(index, index_path)

    failure_rows = []
    for modality, failures in all_failures.items():
        for failure in failures:
            row = {"modality": modality}
            row.update(failure)
            failure_rows.append(row)
    failures_path = output_dir / "feature_failures.csv"
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
    else:
        pd.DataFrame(columns=["modality"]).to_csv(failures_path, index=False)

    summary = {
        "unified_csv": str(unified_csv),
        "unified_sha256": sha256_file(unified_csv),
        "output_dir": str(output_dir),
        "task": task or "all",
        "n_rows": int(len(frame)),
        "config": config,
        "config_hashes": {
            "protein": protein_config_hash(protein_config),
            "chemical": chemical_config_hash(chemical_config),
            "reaction": reaction_config_hash(reaction_config),
            "structure": structure_config_hash(structure_config),
        },
        "dimensions": {
            "protein_residue": ESMC_DIM,
            "protein_global": ESMC_DIM,
            "chemical": CHEM_TOTAL_DIM,
            "reaction": RXNFP_DIM,
            "node_quality": NODE_QUALITY_DIM,
            "edge_attr": EDGE_ATTR_DIM,
            "edge_quality": EDGE_QUALITY_DIM,
        },
        "status_counts": {
            "protein": index["protein_status"].value_counts().to_dict(),
            "chemical": index["chemical_status"].value_counts().to_dict(),
            "reaction": index["reaction_status"].value_counts().to_dict(),
            "structure": index["structure_status"].value_counts().to_dict(),
        },
        "failure_counts": {k: len(v) for k, v in all_failures.items()},
        "cache_root": str(cache_root),
        "device": device,
        "limit": limit,
    }
    write_json_atomic(output_dir / "build_summary.json", summary)
    return summary


def _status_for(statuses: dict[str, str], key: str, requested: bool, enabled: bool, key_name: str) -> str:
    if not enabled:
        return "disabled"
    if not requested or not key:
        return "missing_input"
    return statuses.get(key, "not_built")


def _reaction_id(reaction_smiles: str) -> str:
    from .utils import reaction_id
    return reaction_id(reaction_smiles)


def _build_protein(records, cache_root, config, device, limit):
    from .protein_features import build_protein_cache
    return build_protein_cache(records, cache_root, config, device=device, limit=limit)


def _build_chemical(records, cache_root, config, device, limit):
    from .chemical_features import build_chemical_cache
    return build_chemical_cache(records, cache_root, config, device=device, limit=limit)


def _build_reaction(records, cache_root, config, device, limit):
    from .reaction_features import encode_reactions
    return encode_reactions(records, cache_root, config, device=device, limit=limit)


def _build_structure(records, cache_root, config, roots, limit):
    from .structure_features import build_structure_cache
    return build_structure_cache(records, cache_root, config, roots, limit=limit)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)
