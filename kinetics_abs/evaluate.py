from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir, write_json_atomic


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "pcc": float("nan"), "scc": float("nan"), "r2": float("nan")}
    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    if n >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pcc = float("nan")
    try:
        from scipy.stats import spearmanr
        scc = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        if n >= 2:
            rank_true = pd.Series(y_true).rank().to_numpy()
            rank_pred = pd.Series(y_pred).rank().to_numpy()
            scc = float(np.corrcoef(rank_true, rank_pred)[0, 1]) if np.std(rank_true) > 0 and np.std(rank_pred) > 0 else float("nan")
        else:
            scc = float("nan")
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": n, "rmse": rmse, "mae": mae, "pcc": pcc, "scc": scc, "r2": r2}


def _safe_metric_frame(df: pd.DataFrame, label_col: str, pred_col: str) -> dict[str, Any]:
    return compute_metrics(df[label_col].to_numpy(), df[pred_col].to_numpy())


def evaluate_predictions_csv(predictions_csv: str | Path, output_dir: str | Path, *, label_col: str = "label_log10", pred_col: str = "prediction_log10", identity_col: str = "max_identity_train", group_cols: list[str] | None = None) -> dict[str, Any]:
    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    frame = pd.read_csv(predictions_csv, dtype=str, keep_default_na=False, low_memory=False)
    for col in [label_col, pred_col]:
        if col not in frame.columns:
            raise ValueError(f"Missing column {col} in {predictions_csv}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid = frame[frame[label_col].notna() & frame[pred_col].notna()].copy()
    overall = _safe_metric_frame(valid, label_col, pred_col)

    stratified: dict[str, Any] = {"available": False, "reason": "max_identity_train column not present"}
    if identity_col in valid.columns:
        identity = pd.to_numeric(valid[identity_col], errors="coerce")
        bins = [-np.inf, 40.0, 80.0, 99.0, np.inf]
        labels = ["<40", "40-<80", "80-<99", ">=99"]
        valid["_identity_bin"] = pd.cut(identity, bins=bins, labels=labels, right=False)
        stratified = {"available": True, "bins": {}}
        for label in labels:
            part = valid[valid["_identity_bin"].astype(str) == label]
            stratified["bins"][label] = _safe_metric_frame(part, label_col, pred_col) if len(part) else {"n": 0}
    else:
        # Fall back to reporting cluster-column group sizes without calling them identity strata.
        group_cols = group_cols or ["sequence_40cluster"]
        present = [c for c in group_cols if c in valid.columns]
        if present:
            stratified = {"available": False, "reason": "identity unavailable", "cluster_columns_present": present}

    group_metrics = {}
    for col in ["official_split", "sequence_40cluster", "sequence_60cluster", "sequence_80cluster", "sequence_99cluster"]:
        if col in valid.columns:
            group_metrics[col] = {}
            for key, part in valid.groupby(col, dropna=False):
                if len(part) >= 3:
                    group_metrics[col][str(key)] = _safe_metric_frame(part, label_col, pred_col)

    report = {
        "input_csv": str(predictions_csv),
        "n_rows": int(len(frame)),
        "n_valid": int(len(valid)),
        "overall": overall,
        "identity_stratified": stratified,
        "group_metrics": group_metrics,
    }
    write_json_atomic(output_dir / "evaluation_metrics.json", report)
    lines = [
        "# 预测评价报告",
        "",
        f"- 输入：`{predictions_csv}`",
        f"- 有效样本：{len(valid)} / {len(frame)}",
        f"- RMSE：{overall.get('rmse')}",
        f"- MAE：{overall.get('mae')}",
        f"- PCC：{overall.get('pcc')}",
        f"- SCC：{overall.get('scc')}",
        f"- R2：{overall.get('r2')}",
        "",
        f"- identity分层：{stratified.get('available')}",
    ]
    (output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def aggregate_oof(prediction_files: list[str | Path], output_dir: str | Path, *, id_col: str = "sample_id", label_col: str = "label_log10", pred_col: str = "prediction_log10") -> dict[str, Any]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    frames = []
    for path in prediction_files:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        frame["_source"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No prediction files provided")
    all_df = pd.concat(frames, ignore_index=True)
    for col in [label_col, pred_col]:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
    # Keep the first OOF occurrence per sample; callers should pass one file per held-out fold.
    oof = all_df.drop_duplicates(subset=[id_col], keep="first").copy()
    metrics = compute_metrics(oof[label_col].to_numpy(), oof[pred_col].to_numpy())
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    report = {"n_files": len(prediction_files), "n_oof": int(len(oof)), "metrics": metrics}
    write_json_atomic(output_dir / "oof_metrics.json", report)
    return report
