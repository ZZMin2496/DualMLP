from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_adapters import load_unified_csv
from .utils import finite_or_none, project_path, sha256_file, split_reaction_smiles


def _smiles_components_valid(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        from rdkit import Chem
    except Exception:
        return True
    for component in smiles.split("."):
        component = component.strip()
        if not component:
            continue
        mol = Chem.MolFromSmiles(component)
        if mol is None:
            return False
    return True


def _reaction_valid(reaction_smiles: str) -> bool:
    parsed = split_reaction_smiles(reaction_smiles)
    if parsed["status"] != "ok":
        return False
    return _smiles_components_valid(parsed["reactants"]) and _smiles_components_valid(parsed["products"])


def audit_unified(
    unified_csv: str | Path,
    output_dir: str | Path,
    *,
    structure_root: str | Path | None = None,
    max_protein_length: int = 2048,
    max_chem_context_chars: int = 2048,
    max_reaction_chars: int = 4096,
) -> dict[str, Any]:
    unified_csv = Path(unified_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_unified_csv(unified_csv)
    n = len(frame)

    sequence_lengths = frame["sequence_actual"].fillna("").str.len()
    chem_lengths = frame["chem_context_smiles"].fillna("").str.len()
    reaction_lengths = frame["reaction_smiles"].fillna("").str.len()
    temperature = frame["temperature_raw"].map(finite_or_none)
    ph = frame["ph_raw"].map(finite_or_none)

    reaction_status = Counter(frame["reaction_status"].fillna("").tolist())
    sequence_status = Counter(frame["sequence_status"].fillna("").tolist())
    chem_rule = Counter(frame["chem_context_rule"].fillna("").tolist())

    smiles_valid = frame["chem_context_smiles"].map(_smiles_components_valid)
    reaction_valid = frame["reaction_smiles"].map(_reaction_valid)

    structure_refs = frame["structure_ref"].fillna("").astype(str)
    structure_nonempty = structure_refs.str.strip().ne("")
    structure_existing = pd.Series(False, index=frame.index)
    if structure_root is not None:
        root = Path(structure_root)
        for idx, ref in structure_refs.items():
            if not ref:
                continue
            candidates = [Path(ref), root / ref]
            if "alphafold" in ref:
                candidates.append(root / "datasets" / "raw" / "alphafold" / Path(ref).name)
                candidates.append(root / "data" / "raw" / "alphafold" / Path(ref).name)
            structure_existing.loc[idx] = any(path.is_file() for path in candidates)

    combo = (
        frame["sequence_id"].fillna("") + "|" +
        frame["chem_context_smiles"].fillna("") + "|" +
        frame["reaction_smiles"].fillna("") + "|" +
        frame["label_log10"].fillna("")
    )
    duplicate_mask = combo.duplicated(keep=False)

    label_missing = int((frame["label_available"].astype(str) != "1").sum())
    protein_missing = int((frame["protein_available"].astype(str) != "1").sum())
    chem_missing = int((frame["chem_available"].astype(str) != "1").sum())

    report = {
        "input_csv": str(unified_csv),
        "input_sha256": sha256_file(unified_csv),
        "n_rows": int(n),
        "columns": list(frame.columns),
        "unique_sequence_ids": int(frame.loc[frame["sequence_id"] != "", "sequence_id"].nunique()),
        "unique_chemical_contexts": int(frame.loc[frame["chem_context_smiles"] != "", "chem_context_smiles"].nunique()),
        "unique_reactions": int(frame.loc[frame["reaction_smiles"] != "", "reaction_smiles"].nunique()),
        "label_missing_rows": label_missing,
        "protein_missing_rows": protein_missing,
        "chem_missing_rows": chem_missing,
        "sequence_status_counts": dict(sequence_status),
        "reaction_status_counts": dict(reaction_status),
        "chem_context_rule_counts": dict(chem_rule),
        "chem_context_smiles_parse_failures": int((frame["chem_context_smiles"] != "").sum() - int(smiles_valid.sum())),
        "reaction_smiles_parse_failures": int((frame["reaction_smiles"] != "").sum() - int(reaction_valid.sum())),
        "duplicate_input_combination_rows": int(duplicate_mask.sum()),
        "structure_reference_rows": int(structure_nonempty.sum()),
        "structure_local_existing_rows": int(structure_existing.sum()) if structure_root is not None else None,
        "structure_local_coverage": (float(structure_existing.sum() / max(1, structure_nonempty.sum())) if structure_root is not None and structure_nonempty.sum() else None),
        "temperature_numeric_coverage": float(temperature.notna().mean()),
        "ph_numeric_coverage": float(ph.notna().mean()),
        "sequence_length": {
            "min": int(sequence_lengths.min()) if n else 0,
            "median": float(sequence_lengths.median()) if n else 0.0,
            "max": int(sequence_lengths.max()) if n else 0,
            "over_limit_rows": int((sequence_lengths > max_protein_length).sum()),
            "over_limit_fraction": float((sequence_lengths > max_protein_length).mean()) if n else 0.0,
        },
        "chem_context_length_chars": {
            "median": float(chem_lengths.median()) if n else 0.0,
            "max": int(chem_lengths.max()) if n else 0,
            "over_char_limit_rows": int((chem_lengths > max_chem_context_chars).sum()),
        },
        "reaction_length_chars": {
            "median": float(reaction_lengths.median()) if n else 0.0,
            "max": int(reaction_lengths.max()) if n else 0,
            "over_char_limit_rows": int((reaction_lengths > max_reaction_chars).sum()),
        },
        "task_counts": frame["task"].value_counts().to_dict(),
        "official_split_counts": frame["official_split"].value_counts().to_dict() if "official_split" in frame else {},
    }

    failures = pd.DataFrame({
        "sample_id": frame["sample_id"],
        "sequence_status": frame["sequence_status"],
        "chem_context_rule": frame["chem_context_rule"],
        "reaction_status": frame["reaction_status"],
        "label_available": frame["label_available"],
        "protein_available": frame["protein_available"],
        "chem_available": frame["chem_available"],
        "structure_ref": frame["structure_ref"],
        "sequence_length": sequence_lengths,
        "chem_context_length_chars": chem_lengths,
        "reaction_length_chars": reaction_lengths,
    })
    failures.to_csv(output_dir / "audit_failures.csv", index=False)
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 数据审计报告",
        "",
        f"- 输入：`{unified_csv}`",
        f"- 行数：{n}",
        f"- 唯一序列：{report['unique_sequence_ids']}",
        f"- 唯一化学上下文：{report['unique_chemical_contexts']}",
        f"- 唯一反应：{report['unique_reactions']}",
        f"- 标签缺失：{label_missing}",
        f"- 蛋白不可用：{protein_missing}",
        f"- 化学上下文缺失：{chem_missing}",
        f"- 结构引用行：{report['structure_reference_rows']}",
        f"- 结构本地已存在：{report['structure_local_existing_rows']}",
        f"- 温度数值覆盖：{report['temperature_numeric_coverage']:.4f}",
        f"- pH 数值覆盖：{report['ph_numeric_coverage']:.4f}",
        "",
        "## 状态计数",
        "",
        f"- sequence_status：{report['sequence_status_counts']}",
        f"- reaction_status：{report['reaction_status_counts']}",
        f"- chem_context_rule：{report['chem_context_rule_counts']}",
        "",
        "## 长度检查",
        "",
        f"- sequence_length：{report['sequence_length']}",
        f"- chem_context_length_chars：{report['chem_context_length_chars']}",
        f"- reaction_length_chars：{report['reaction_length_chars']}",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
