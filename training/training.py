import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import DualMLP
from utils import (
    KineticsDataset,
    build_loss,
    build_optimizer,
    build_scheduler,
    calculate_metrics,
    evaluate,
    load_config,
    load_data,
    project_path,
    set_seed,
    split_indices,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("kcat", "Km", "all"), required=True)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--device", default="auto",
                        help="Training device: auto (default), cpu, or cuda:N")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def fold_seed(config, test_fold):
    mode = config["seed_mode"]
    if mode == "fixed":
        return int(config["seed"])
    if mode == "fold_offset":
        return int(config["seed"]) + int(test_fold)
    raise ValueError("Unsupported seed_mode: {}".format(mode))


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(task, config):
    relative_paths = [config["dataset_path"]]
    relative_paths.extend(
        str(Path(config["features_dir"]) / (name + ".npy"))
        for name in ("sample_id", "esmc", "chemberta", "maccs")
    )
    relative_paths.extend(
        "training/" + name for name in ("model.py", "utils.py", "training.py")
    )
    manifest = {
        "task": task,
        "config": config,
        "sha256": {
            relative: file_digest(project_path(relative))
            for relative in relative_paths
        },
    }
    folder = project_path(config["results_dir"])
    path = folder / "run_manifest.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(
                "Inputs or code changed. Use fresh models_dir and results_dir."
            )
    else:
        model_folder = project_path(config["models_dir"])
        if any(folder.glob("fold_*")) or any(model_folder.glob("fold_*")):
            raise RuntimeError("Fold outputs exist without a run manifest")
        folder.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)


def prediction_frame(config, data, test_fold, path):
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = [
        "row_index", "sample_id", "fold", "test_fold", "validation_fold",
        "y_true_log10", "y_pred_log10",
    ]
    if any(column not in frame.columns for column in required):
        raise RuntimeError("Prediction file is missing required columns")
    _, _, expected_rows, _, val_fold = split_indices(
        data["folds"], test_fold, config["split"]
    )
    checks = (
        len(frame) == len(expected_rows),
        np.array_equal(frame["row_index"].to_numpy(), expected_rows),
        np.array_equal(
            frame["sample_id"].to_numpy(), data["sample_ids"][expected_rows]
        ),
        np.array_equal(frame["fold"].to_numpy(), data["folds"][expected_rows]),
        frame["test_fold"].eq(test_fold).all(),
        frame["validation_fold"].eq(val_fold).all(),
    )
    if not all(checks):
        raise RuntimeError("Prediction rows do not match the configured split")
    values = frame[["y_true_log10", "y_pred_log10"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Prediction file contains nonfinite values")
    if not np.allclose(
        values[:, 0], data["target"][expected_rows], rtol=0, atol=1e-6
    ):
        raise RuntimeError("Prediction labels do not match the dataset")
    return frame


def train_one_fold(task, config, data, test_fold, device, overwrite=False):
    model_config = config["model_config"]
    training_config = config["training_config"]
    train_idx, val_idx, test_idx, train_folds, val_fold = split_indices(
        data["folds"], test_fold, config["split"]
    )
    seed = fold_seed(config, test_fold)
    batch_size = int(model_config["batch_size"])
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["early_stop_patience"])
    min_delta = float(training_config["min_delta"])
    if batch_size < 2 or len(train_idx) < batch_size:
        raise ValueError("Training needs at least one batch of two or more rows")
    if max_epochs < 1 or patience < 1 or min_delta < 0:
        raise ValueError("Invalid epoch or early stopping configuration")
    model_dir = project_path(config["models_dir"]) / ("fold_{}".format(test_fold))
    result_dir = project_path(config["results_dir"]) / ("fold_{}".format(test_fold))
    checkpoint_path = model_dir / "best.pt"
    history_path = result_dir / "history.csv"
    prediction_path = result_dir / "test_predictions.csv"
    metrics_path = result_dir / "metrics.json"
    outputs = [checkpoint_path, history_path, prediction_path, metrics_path]
    if not overwrite and all(path.is_file() for path in outputs):
        prediction_frame(config, data, test_fold, prediction_path)
        print("{} fold {}: already complete, skip".format(task, test_fold))
        return
    if not overwrite and any(path.exists() for path in outputs):
        raise RuntimeError("Partial fold outputs exist; inspect or use --overwrite")
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in outputs:
            if path.exists():
                path.unlink()
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        KineticsDataset(train_idx, data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    eval_options = {
        "batch_size": int(training_config["eval_batch_size"]),
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(KineticsDataset(val_idx, data), **eval_options)
    test_loader = DataLoader(KineticsDataset(test_idx, data), **eval_options)
    model = DualMLP(model_config).to(device)
    optimizer = build_optimizer(model, model_config)
    criterion = build_loss(model_config)
    scheduler = build_scheduler(optimizer, model_config)
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    started = time.time()
    print(
        "{} test={} validation={} train={} seed={} N={}/{}/{}".format(
            task, test_fold, val_fold, train_folds, seed,
            len(train_idx), len(val_idx), len(test_idx),
        ),
        flush=True,
    )
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        n_seen = 0
        for esmc, chemberta, maccs, target, _ in train_loader:
            esmc = esmc.to(device, non_blocking=True)
            chemberta = chemberta.to(device, non_blocking=True)
            maccs = maccs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(esmc, chemberta, maccs)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            batch_n = int(target.size(0))
            loss_sum += float(loss.detach().cpu()) * batch_n
            n_seen += batch_n
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        val_rmse = float(val_metrics["RMSE"])
        if scheduler is not None:
            scheduler.step(val_rmse)
        if val_rmse < best_val_rmse - min_delta:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "model": "DualMLP",
                "task": task,
                "split_mode": "811",
                "train_folds": train_folds,
                "validation_fold": val_fold,
                "test_fold": test_fold,
                "seed": seed,
                "epoch": epoch,
                "best_validation_RMSE": best_val_rmse,
                "target_scale": config["target_scale"],
                "original_unit": config["original_unit"],
                "model_config": model_config,
                "training_config": training_config,
                "test_used_for_checkpoint_selection": False,
                "device": str(device),
                "torch_version": str(torch.__version__),
                "model_state_dict": model.state_dict(),
            }
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        else:
            patience_counter += 1
        history.append({
            "epoch": epoch,
            "train_objective": loss_sum / n_seen,
            "validation_RMSE": val_metrics["RMSE"],
            "validation_PCC": val_metrics["PCC"],
            "validation_SCC": val_metrics["SCC"],
            "best_validation_RMSE": best_val_rmse,
            "best_epoch": best_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "patience_counter": patience_counter,
            "seconds": time.time() - epoch_start,
        })
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            "Epoch {:03d} | val RMSE {:.6f} | PCC {:.6f} | SCC {:.6f} | "
            "best {:.6f} @ {:03d} | patience {}/{} | lr {:.3e}".format(
                epoch, val_metrics["RMSE"], val_metrics["PCC"],
                val_metrics["SCC"], best_val_rmse, best_epoch,
                patience_counter, patience, optimizer.param_groups[0]["lr"],
            ),
            flush=True,
        )
        if patience_counter >= patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_metrics, y_true, y_pred, rows = evaluate(model, test_loader, device)
    if not np.array_equal(rows, test_idx):
        raise RuntimeError("Test row order mismatch")
    predictions = pd.DataFrame({
        "row_index": rows,
        "sample_id": data["sample_ids"][rows],
        "fold": data["folds"][rows],
        "test_fold": test_fold,
        "validation_fold": val_fold,
        "y_true_log10": y_true,
        "y_pred_log10": y_pred,
    })
    predictions.to_csv(prediction_path, index=False)
    prediction_frame(config, data, test_fold, prediction_path)
    record = {
        "model": "DualMLP",
        "task": task,
        "split_mode": "811",
        "test_fold": test_fold,
        "validation_fold": val_fold,
        "train_folds": train_folds,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_RMSE": float(checkpoint["best_validation_RMSE"]),
        "target_scale": config["target_scale"],
        "original_unit": config["original_unit"],
        "test_used_for_checkpoint_selection": False,
        "test_metrics": test_metrics,
        "runtime_seconds": float(time.time() - started),
    }
    write_json(metrics_path, record)
    print("{} fold {} test: {}".format(task, test_fold, test_metrics), flush=True)
    del model, optimizer, criterion, scheduler, checkpoint
    del train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate(task, config, data):
    folder = project_path(config["results_dir"])
    model_folder = project_path(config["models_dir"])
    n_folds = int(config["split"]["n_folds"])
    required = []
    for fold in range(n_folds):
        fold_dir = folder / ("fold_{}".format(fold))
        required.extend([
            fold_dir / "test_predictions.csv",
            fold_dir / "metrics.json",
            fold_dir / "history.csv",
            model_folder / ("fold_{}".format(fold)) / "best.pt",
        ])
    if not all(path.is_file() for path in required):
        print("{}: OOF aggregation waits for all folds".format(task), flush=True)
        return False
    frames, records = [], []
    for fold in range(n_folds):
        fold_dir = folder / ("fold_{}".format(fold))
        frame = prediction_frame(
            config, data, fold, fold_dir / "test_predictions.csv"
        )
        metrics = calculate_metrics(frame["y_true_log10"], frame["y_pred_log10"])
        record = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
        if record["test_fold"] != fold or record["seed"] != fold_seed(config, fold):
            raise RuntimeError("Fold metadata does not match configuration")
        for key, value in metrics.items():
            if not np.isclose(
                value, record["test_metrics"][key], rtol=0, atol=1e-6,
                equal_nan=True,
            ):
                raise RuntimeError("Saved metrics differ from saved predictions")
        records.append({
            "test_fold": fold,
            "validation_fold": record["validation_fold"],
            "seed": record["seed"],
            **metrics,
        })
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True).sort_values("row_index")
    n = len(data["target"])
    if len(pooled) != n or not np.array_equal(pooled["row_index"], np.arange(n)):
        raise RuntimeError("OOF predictions must cover every row exactly once")
    if not np.array_equal(pooled["sample_id"], data["sample_ids"]):
        raise RuntimeError("OOF sample IDs are not aligned")
    pooled.to_csv(folder / "oof_predictions.csv", index=False)
    pd.DataFrame(records).to_csv(folder / "fold_metrics.csv", index=False)
    metrics = calculate_metrics(pooled["y_true_log10"], pooled["y_pred_log10"])
    write_json(folder / "oof_metrics.json", metrics)
    print("{} OOF: {}".format(task, metrics), flush=True)
    return True


def main():
    args = parse_args()
    if args.threads < 1 or args.fold not in range(-1, 10):
        raise ValueError("--threads must be positive; --fold must be -1 or 0-9")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    if not args.check_only and not args.aggregate_only:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    print("Resolved device: {}".format(device), flush=True)
    tasks = ("kcat", "Km") if args.task == "all" else (args.task,)
    for task in tasks:
        config = load_config(task)
        data = load_data(config)
        coverage = np.zeros(len(data["target"]), dtype=np.int64)
        for fold in range(int(config["split"]["n_folds"])):
            train, val, test, train_folds, val_fold = split_indices(
                data["folds"], fold, config["split"]
            )
            coverage[test] += 1
            print(
                "{} test={} validation={} train={} seed={} N={}/{}/{}".format(
                    task, fold, val_fold, train_folds, fold_seed(config, fold),
                    len(train), len(val), len(test),
                ),
                flush=True,
            )
        if not np.all(coverage == 1):
            raise RuntimeError("Every sample must occur in exactly one test fold")
        if args.check_only:
            print("{} CHECK-ONLY PASS".format(task), flush=True)
            continue
        verify_run(task, config)
        if args.aggregate_only:
            if not aggregate(task, config, data):
                raise RuntimeError("Cannot aggregate incomplete folds")
            continue
        folds = range(int(config["split"]["n_folds"])) if args.fold == -1 else (
            args.fold,
        )
        for fold in folds:
            train_one_fold(task, config, data, fold, device, args.overwrite)
        aggregate(task, config, data)
        del data
        gc.collect()


if __name__ == "__main__":
    main()
