from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import KineticsDataset, LengthBucketBatchSampler, collate_kinetics_batch
from .evaluate import compute_metrics
from .model import baseline_input, build_model, count_parameters
from .utils import ensure_dir, project_path, read_yaml, set_seed, write_json_atomic


def _numpy_state_to_safe(state: Any) -> dict[str, Any]:
    name, keys, pos, has_gauss, cached_gauss = state
    return {
        "name": str(name),
        "keys": [int(x) for x in np.asarray(keys, dtype=np.uint32).tolist()],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gauss": float(cached_gauss),
    }


def _numpy_state_from_safe(state: dict[str, Any]) -> Any:
    return (
        str(state["name"]),
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gauss"]),
    )


def _move_state_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_state_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_state_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_state_to_device(v, device) for v in value)
    return value


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _model_needs(model_config: dict[str, Any]) -> dict[str, Any]:
    architecture = str(model_config.get("architecture", "fusion"))
    if architecture == "baseline":
        variant = str(model_config.get("variant", "b0"))
        return {
            "need_residue": False,
            "need_reaction": variant == "b1",
            "need_structure": False,
        }
    graph_mode = str(model_config.get("graph_mode", "none"))
    return {
        "need_residue": True,
        "need_reaction": bool(model_config.get("use_reaction", True)),
        "need_structure": graph_mode != "none",
    }


def _dataset(config: dict[str, Any], split_name: str, fold_index: int, seed: int, limit: int | None = None) -> KineticsDataset:
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    model_cfg = config.get("model", {})
    needs = _model_needs(model_cfg)
    return KineticsDataset(
        unified_csv=project_path(data_cfg["unified_csv"]),
        feature_index_csv=project_path(data_cfg["feature_index_csv"]),
        cache_root=project_path(data_cfg.get("cache_root", "features_cache")),
        config=config,
        task=str(config.get("task", data_cfg.get("task", "kcat"))),
        split_protocol=str(split_cfg.get("protocol", "official")),
        split_name=split_name,
        fold_index=fold_index,
        seed=seed,
        limit=limit,
        max_protein_length=int(data_cfg.get("max_protein_length", 2048)),
        require_reaction=bool(split_cfg.get("require_reaction", False)),
        require_structure=bool(split_cfg.get("require_structure", False)),
        **needs,
    )


def _loader(dataset: KineticsDataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    sampler = LengthBucketBatchSampler(dataset, batch_size=batch_size, seed=seed, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_kinetics_batch, num_workers=0)


def _forward(model: torch.nn.Module, batch: dict[str, torch.Tensor], model_config: dict[str, Any]) -> torch.Tensor:
    if str(model_config.get("architecture", "fusion")) == "baseline":
        return model(baseline_input(batch, str(model_config.get("variant", "b0"))))
    return model(batch)


def _evaluate_loader(model: torch.nn.Module, loader: DataLoader, model_config: dict[str, Any], device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    preds = []
    labels = []
    meta_keys = ["sample_id", "sequence_id", "uniprot", "official_split", "sequence_40cluster", "sequence_60cluster", "sequence_80cluster", "sequence_99cluster"]
    meta_rows = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = _forward(model, batch, model_config).squeeze(-1).float().cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch["labels"].float().cpu().numpy().tolist())
            for i in range(len(batch["metadata"]["sample_id"])):
                meta_rows.append({key: batch["metadata"][key][i] for key in meta_keys})
    y_true = np.asarray(labels, dtype=np.float64)
    y_pred = np.asarray(preds, dtype=np.float64)
    metrics = compute_metrics(y_true, y_pred)
    meta_df = pd.DataFrame(meta_rows, columns=meta_keys)
    out = meta_df.copy()
    out["label_log10"] = y_true
    out["prediction_log10"] = y_pred
    out["residual"] = y_pred - y_true
    return metrics, out


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train_one(config: dict[str, Any], fold_index: int, seed: int, output_dir: str | Path, *, resume: bool = False, limit: int | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    train_cfg = config.get("training", {})
    device = torch.device(str(train_cfg.get("device", "cpu")))
    set_seed(seed)

    train_ds = _dataset(config, "train", fold_index, seed, limit=limit)
    val_ds = _dataset(config, "val", fold_index, seed, limit=limit)
    micro_batch = int(train_cfg.get("micro_batch_size", 16))
    effective_batch = int(train_cfg.get("effective_batch_size", 128))
    accum_steps = max(1, int(math.ceil(effective_batch / max(1, micro_batch))))
    train_loader = _loader(train_ds, micro_batch, seed, shuffle=True)
    val_loader = _loader(val_ds, max(micro_batch, 32), seed, shuffle=False)

    model_config = config.get("model", {})
    model = build_model(model_config).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(train_cfg.get("scheduler_patience", 5)), threshold=0.0)

    max_epochs = int(train_cfg.get("max_epochs", 200))
    early_patience = int(train_cfg.get("early_stopping_patience", 20))
    clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    checkpoint_dir = output_dir / "checkpoints"
    last_path = checkpoint_dir / "checkpoint_last.pt"
    best_path = checkpoint_dir / "checkpoint_best.pt"

    start_epoch = 0
    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    sampler_epoch = 0
    if resume and last_path.is_file():
        checkpoint = _load_checkpoint(last_path)
        model.load_state_dict(_move_state_to_device(checkpoint["model_state"], device))
        optimizer.load_state_dict(_move_state_to_device(checkpoint["optimizer_state"], device))
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_rmse = float(checkpoint.get("best_val_rmse", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        history = list(checkpoint.get("history", []))
        sampler_epoch = int(checkpoint.get("sampler_epoch", start_epoch))
        random.setstate(checkpoint.get("random_state", random.getstate()))
        if "numpy_state" in checkpoint:
            np.random.set_state(_numpy_state_from_safe(checkpoint["numpy_state"]))
        torch.set_rng_state(checkpoint.get("torch_state", torch.get_rng_state()))
        if device.type == "cuda" and "torch_cuda_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_state"])

    early_counter = 0
    train_start = time.time()
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loader.batch_sampler.epoch = sampler_epoch  # type: ignore[attr-defined]
        sampler_epoch += 1
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = _forward(model, batch, model_config).squeeze(-1).float()
            labels = batch["labels"].float()
            loss = F.mse_loss(pred, labels) / accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch} batch {batch_index}")
            loss.backward()
            total_loss += float(F.mse_loss(pred.detach(), labels).item())
            n_batches += 1
            is_last = batch_index == len(train_loader) - 1
            if (batch_index + 1) % accum_steps == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        train_rmse = math.sqrt(total_loss / max(1, n_batches))

        val_metrics, _ = _evaluate_loader(model, val_loader, model_config, device)
        val_rmse = float(val_metrics["rmse"])
        scheduler.step(val_rmse)
        improved = val_rmse < best_val_rmse - 1e-6
        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            early_counter = 0
            _save_checkpoint(best_path, {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": config,
                "fold_index": fold_index,
                "seed": seed,
                "best_val_rmse": best_val_rmse,
                "best_epoch": best_epoch,
                "history": history,
                "sampler_epoch": sampler_epoch,
                "random_state": random.getstate(),
                "numpy_state": _numpy_state_to_safe(np.random.get_state()),
                "torch_state": torch.get_rng_state(),
                "torch_cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            })
        else:
            early_counter += 1
        record = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "val_metrics": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "improved": bool(improved),
        }
        history.append(record)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _save_checkpoint(last_path, {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "fold_index": fold_index,
            "seed": seed,
            "best_val_rmse": best_val_rmse,
            "best_epoch": best_epoch,
            "history": history,
            "sampler_epoch": sampler_epoch,
            "random_state": random.getstate(),
            "numpy_state": _numpy_state_to_safe(np.random.get_state()),
            "torch_state": torch.get_rng_state(),
            "torch_cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        })
        if early_counter >= early_patience:
            break

    if best_path.is_file():
        checkpoint = _load_checkpoint(best_path)
        model.load_state_dict(_move_state_to_device(checkpoint["model_state"], device))
    val_metrics, val_pred_df = _evaluate_loader(model, val_loader, model_config, device)
    val_pred_df.to_csv(output_dir / "val_predictions.csv", index=False)

    test_metrics = None
    test_pred_df = None
    if bool(train_cfg.get("evaluate_test", False)):
        try:
            test_ds = _dataset(config, "test", fold_index, seed, limit=limit)
            test_loader = _loader(test_ds, max(micro_batch, 32), seed, shuffle=False)
            test_metrics, test_pred_df = _evaluate_loader(model, test_loader, model_config, device)
            test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
        except Exception as exc:
            test_metrics = {"error": f"{type(exc).__name__}: {exc}"}

    summary = {
        "experiment": config.get("experiment", "unnamed"),
        "task": config.get("task", config.get("data", {}).get("task", "kcat")),
        "fold_index": fold_index,
        "seed": seed,
        "output_dir": str(output_dir),
        "device": str(device),
        "n_parameters": n_params,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "micro_batch_size": micro_batch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": accum_steps,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_seconds": time.time() - train_start,
        "resumed": bool(resume and last_path.is_file()),
        "config": config,
    }
    write_json_atomic(output_dir / "train_summary.json", summary)
    return summary


def run_experiment(config_path: str | Path, *, output_root: str | Path | None = None, resume: bool = False, limit: int | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    config["config_path"] = str(config_path)
    output_cfg = config.get("output", {})
    root = project_path(output_root or output_cfg.get("root", "results_abs"))
    experiment = str(config.get("experiment", config_path.stem))
    task = str(config.get("task", config.get("data", {}).get("task", "kcat")))
    experiment_dir = ensure_dir(root / experiment)
    split_cfg = config.get("split", {})
    training_cfg = config.get("training", {})
    folds = split_cfg.get("folds", training_cfg.get("folds", [0]))
    seeds = training_cfg.get("seeds", [42])
    runs = []
    for fold_index in folds:
        for seed in seeds:
            run_dir = experiment_dir / f"fold{int(fold_index)}_seed{int(seed)}"
            summary = train_one(config, int(fold_index), int(seed), run_dir, resume=resume, limit=limit)
            runs.append(summary)
    metric_rows = []
    for run in runs:
        metrics = run.get("best_val_metrics", {})
        metric_rows.append({
            "fold_index": run["fold_index"],
            "seed": run["seed"],
            **{k: metrics.get(k) for k in ["rmse", "mae", "pcc", "scc", "r2"]},
            "n_parameters": run["n_parameters"],
            "best_epoch": run["best_epoch"],
            "output_dir": run["output_dir"],
        })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(experiment_dir / "val_metrics_by_fold_seed.csv", index=False)
    aggregate = {
        "experiment": experiment,
        "task": task,
        "config_path": str(config_path),
        "n_runs": len(runs),
        "metrics_mean": {k: float(metrics_df[k].mean()) for k in ["rmse", "mae", "pcc", "scc", "r2"] if k in metrics_df},
        "metrics_std": {k: float(metrics_df[k].std(ddof=0)) for k in ["rmse", "mae", "pcc", "scc", "r2"] if k in metrics_df},
        "runs": [{k: v for k, v in run.items() if k != "config"} for run in runs],
    }
    write_json_atomic(experiment_dir / "experiment_summary.json", aggregate)
    return aggregate
