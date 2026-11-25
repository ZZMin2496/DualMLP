from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import (
    chemical_id,
    finite_or_none,
    normalize_sequence,
    project_path,
    reaction_id,
    sample_id,
    sequence_id,
    sha256_file,
    split_reaction_smiles,
)

CATPRED_FILES = {
    "kcat": "kcat_max_wt_singleSeqs_wpdbs.csv",
    "Km": "km_mean_wt_singleSeqs_wpdbs.csv",
}
CATPRED_LABEL_COLUMNS = {
    "kcat": ("log10kcat_max", "log10kcat_max", "s^-1"),
    "Km": ("log10km_mean", "log10km_mean", "mM"),
}
UNIFIED_COLUMNS = [
    "sample_id",
    "task",
    "sequence_actual",
    "sequence_id",
    "sequence_status",
    "uniprot",
    "ec",
    "taxonomy_id",
    "chem_context_smiles",
    "chem_context_rule",
    "chemical_scope",
    "target_substrate_smiles",
    "reaction_smiles",
    "reaction_status",
    "temperature_raw",
    "ph_raw",
    "structure_ref",
    "structure_status",
    "label_log10",
    "label_available",
    "label_kind",
    "label_unit",
    "source_record_id",
    "reference_id",
    "assay_id",
    "parent_sequence_id",
    "mutation_description",
    "construct_description",
    "official_split",
    "sequence_40cluster",
    "sequence_60cluster",
    "sequence_80cluster",
    "sequence_99cluster",
    "protein_available",
    "chem_available",
    "reaction_available",
    "source_metadata",
]


def _json_metadata(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def _nonempty(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _catpred_split_frame(task: str, catpred_root: Path) -> pd.DataFrame:
    splits_dir = catpred_root / "datasets" / "splits"
    prefix = "kcat" if task == "kcat" else "km"
    frames = []
    for split_name, filename in (
        ("train", f"{prefix}-random_train.csv"),
        ("val", f"{prefix}-random_val.csv"),
        ("test", f"{prefix}-random_test.csv"),
    ):
        path = splits_dir / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=["Unnamed: 0"], dtype={"Unnamed: 0": "int64"})
        frame["official_split"] = split_name
        frames.append(frame)
    official = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["Unnamed: 0", "official_split"]
    )
    all_rows_path = splits_dir / f"{prefix}-random_trainvaltest.csv"
    cluster_columns = [
        "sequence_40cluster",
        "sequence_60cluster",
        "sequence_80cluster",
        "sequence_99cluster",
    ]
    clusters = pd.DataFrame(columns=["row_index", *cluster_columns])
    if all_rows_path.is_file():
        available = set(pd.read_csv(all_rows_path, nrows=0).columns)
        usecols = ["Unnamed: 0"] + [c for c in cluster_columns if c in available]
        clusters = pd.read_csv(all_rows_path, usecols=usecols)
        clusters = clusters.rename(columns={"Unnamed: 0": "row_index"})
    official = official.rename(columns={"Unnamed: 0": "row_index"})
    merged = official.merge(clusters, on="row_index", how="outer")
    return merged


def adapt_catpred(
    task: str,
    catpred_root: str | Path,
    output_path: str | Path,
    *,
    include_splits: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    if task not in CATPRED_FILES:
        raise ValueError(f"Unsupported CatPred task: {task}")
    catpred_root = Path(catpred_root)
    source_path = catpred_root / "datasets" / "processed" / CATPRED_FILES[task]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_sha = sha256_file(source_path)
    frame = pd.read_csv(source_path, dtype=str, keep_default_na=False, low_memory=False)
    if limit is not None:
        frame = frame.iloc[: int(limit)].copy()
    label_column, label_kind, label_unit = CATPRED_LABEL_COLUMNS[task]
    split_frame = _catpred_split_frame(task, catpred_root) if include_splits else pd.DataFrame()

    records = []
    for row_index, row in frame.iterrows():
        sequence, sequence_status = normalize_sequence(row.get("sequence", ""))
        seq_id = sequence_id(sequence) if sequence_status == "ok" else ""
        reaction = split_reaction_smiles(row.get("reaction_smiles", ""))
        if task == "kcat":
            chem_context = _nonempty(row.get("reactant_smiles", ""))
            chem_rule = "reactant_smiles"
            if not chem_context:
                if reaction["status"] == "ok" and reaction["reactants"]:
                    chem_context = reaction["reactants"]
                    chem_rule = "reaction_smiles_left_side"
                else:
                    chem_rule = "missing_reactant_smiles"
            chemical_scope = "reactant_set"
            target_substrate = ""
            reaction_smiles = reaction["reaction_smiles"]
            reaction_status = reaction["status"]
        else:
            chem_context = _nonempty(row.get("substrate_smiles", ""))
            chem_rule = "substrate_smiles" if chem_context else "missing_substrate_smiles"
            chemical_scope = "target_substrate"
            target_substrate = chem_context
            reaction_smiles = ""
            reaction_status = "not_applicable"
        label_value = finite_or_none(row.get(label_column, ""))
        label_available = int(label_value is not None)
        structure_ref = _nonempty(row.get("pdbpath", ""))
        metadata = {
            "original_row_index": int(row_index),
            "sequence_source": _nonempty(row.get("sequence_source", "")),
            "value_raw": _nonempty(row.get("value", "")),
            "log10_value_raw": _nonempty(row.get("log10_value", "")),
            "product_smiles": _nonempty(row.get("product_smiles", row.get("substrate_smiles", ""))),
            "reaction_mw_diff_perc_raw": _nonempty(row.get("reaction_mw_diff_perc", "")),
            "group_raw": _nonempty(row.get("group", "")),
        }
        record = {
            "sample_id": sample_id(task, file_sha, int(row_index)),
            "task": task,
            "sequence_actual": sequence,
            "sequence_id": seq_id,
            "sequence_status": sequence_status,
            "uniprot": _nonempty(row.get("uniprot", "")),
            "ec": _nonempty(row.get("ec", "")),
            "taxonomy_id": _nonempty(row.get("taxonomy_id", "")),
            "chem_context_smiles": chem_context,
            "chem_context_rule": chem_rule,
            "chemical_scope": chemical_scope,
            "target_substrate_smiles": target_substrate,
            "reaction_smiles": reaction_smiles,
            "reaction_status": reaction_status,
            "temperature_raw": _nonempty(row.get("temperature", "")),
            "ph_raw": _nonempty(row.get("ph", "")),
            "structure_ref": structure_ref,
            "structure_status": "reference_only" if structure_ref else "missing_reference",
            "label_log10": "" if label_value is None else f"{label_value:.12g}",
            "label_available": label_available,
            "label_kind": label_kind,
            "label_unit": label_unit,
            "source_record_id": f"catpred_{task}_row_{int(row_index):08d}",
            "reference_id": "",
            "assay_id": "",
            "parent_sequence_id": "",
            "mutation_description": "",
            "construct_description": "",
            "official_split": "",
            "sequence_40cluster": "",
            "sequence_60cluster": "",
            "sequence_80cluster": "",
            "sequence_99cluster": "",
            "protein_available": int(sequence_status == "ok"),
            "chem_available": int(bool(chem_context)),
            "reaction_available": int(bool(reaction_smiles) and reaction_status == "ok"),
            "source_metadata": _json_metadata(metadata),
        }
        records.append(record)

    adapted = pd.DataFrame(records, columns=UNIFIED_COLUMNS)
    if include_splits and not split_frame.empty:
        split_frame["row_index"] = split_frame["row_index"].astype(int)
        adapted["row_index"] = range(len(adapted))
        adapted = adapted.merge(split_frame, on="row_index", how="left", suffixes=("", "_split"))
        adapted = adapted.drop(columns=["row_index"])
        split_columns = ["official_split", "sequence_40cluster", "sequence_60cluster", "sequence_80cluster", "sequence_99cluster"]
        for column in split_columns:
            split_column = f"{column}_split"
            if split_column in adapted.columns:
                base = adapted[column].fillna("").astype(str)
                split_values = adapted[split_column].fillna("").astype(str)
                adapted[column] = base.where(base.ne(""), split_values)
                adapted = adapted.drop(columns=[split_column])
        for column in split_columns:
            if column in adapted.columns:
                adapted[column] = adapted[column].fillna("").astype(str)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    adapted.to_csv(temp_path, index=False)
    temp_path.replace(output_path)

    summary = {
        "task": task,
        "source_path": str(source_path),
        "source_sha256": file_sha,
        "output_path": str(output_path),
        "n_rows": int(len(adapted)),
        "n_unique_sequences": int(adapted.loc[adapted["sequence_id"] != "", "sequence_id"].nunique()),
        "n_unique_chemical_contexts": int(adapted.loc[adapted["chem_context_smiles"] != "", "chem_context_smiles"].nunique()),
        "n_unique_reactions": int(adapted.loc[adapted["reaction_smiles"] != "", "reaction_smiles"].nunique()),
        "label_available": int(adapted["label_available"].sum()),
        "protein_available": int(adapted["protein_available"].sum()),
        "chem_available": int(adapted["chem_available"].sum()),
        "reaction_available": int(adapted["reaction_available"].sum()),
        "structure_references": int((adapted["structure_ref"] != "").sum()),
        "official_split_counts": adapted["official_split"].value_counts().to_dict() if "official_split" in adapted else {},
    }
    return summary


def load_unified_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = [column for column in UNIFIED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Unified CSV missing columns: {missing}")
    return frame


def write_unified_csv(frame: pd.DataFrame, path: str | Path) -> None:
    missing = [column for column in UNIFIED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Unified frame missing columns: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame[UNIFIED_COLUMNS].to_csv(temp_path, index=False)
    temp_path.replace(path)
