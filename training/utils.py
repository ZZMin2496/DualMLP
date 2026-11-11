import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import Dataset


def project_path(relative_path):
    root = Path(__file__).resolve().parent.parent
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("Path must remain inside the project") from None
    return path


def load_config(task):
    with project_path("training/configs.json").open(encoding="utf-8") as handle:
        configs = json.load(handle)
    if task not in configs:
        raise ValueError("Unknown task: {}".format(task))
    return configs[task]


def load_data(config):
    id_column = config["sample_id_column"]
    fold_column = config["fold_column"]
    target_column = config["target_column"]
    frame = pd.read_csv(
        project_path(config["dataset_path"]),
        dtype={id_column: str},
        keep_default_na=False,
        low_memory=False,
    )
    for column in (id_column, fold_column, target_column):
        if column not in frame.columns:
            raise ValueError("Missing column: {}".format(column))
    n = len(frame)
    if n != int(config["n_samples"]):
        raise ValueError("Dataset sample count does not match configuration")
    csv_ids = frame[id_column].to_numpy(dtype=str)
    if frame[id_column].eq("").any() or not frame[id_column].is_unique:
        raise ValueError("Sample IDs must be nonempty and unique")
    numeric_folds = pd.to_numeric(frame[fold_column], errors="raise")
    n_folds = int(config["split"]["n_folds"])
    if not numeric_folds.isin(range(n_folds)).all():
        raise ValueError("Invalid fold values")
    folds = numeric_folds.to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(folds), np.arange(n_folds)):
        raise ValueError("Dataset must contain every configured fold")
    target = pd.to_numeric(frame[target_column], errors="raise").to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or infinity")
    folder = project_path(config["features_dir"])
    sample_ids = np.load(folder / "sample_id.npy", allow_pickle=False)
    if sample_ids.shape != (n,) or not np.array_equal(sample_ids, csv_ids):
        raise ValueError("CSV and feature sample IDs are not aligned")
    data = {
        "df": frame,
        "folds": folds,
        "target": target,
        "sample_ids": sample_ids,
        "target_scale": config["target_scale"],
    }
    for name in ("esmc", "chemberta", "maccs"):
        path = project_path(str(Path(config["features_dir"]) / (name + ".npy")))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (n, int(config["model_config"][name + "_dim"]))
        if array.shape != expected or array.dtype.kind not in "fiu":
            raise ValueError("Invalid feature shape or dtype: {}".format(name))
        data[name] = array
    return data


class KineticsDataset(Dataset):
    def __init__(self, indices, data):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        esmc = np.array(self.data["esmc"][index], dtype=np.float32, copy=True)
        chemberta = np.array(
            self.data["chemberta"][index], dtype=np.float32, copy=True
        )
        maccs = np.array(self.data["maccs"][index], dtype=np.float32, copy=True)
        target = np.float32(self.data["target"][index])
        return (
            torch.from_numpy(esmc),
            torch.from_numpy(chemberta),
            torch.from_numpy(maccs),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(folds, test_fold, split_config=None):
    folds = np.asarray(folds)
    n_folds = 10 if split_config is None else int(split_config["n_folds"])
    offset = -1 if split_config is None else int(
        split_config["validation_fold_offset"]
    )
    if test_fold not in range(n_folds):
        raise ValueError("Invalid test fold")
    if split_config is not None:
        expected = (n_folds - 2, 1, 1)
        counts = tuple(int(split_config[key]) for key in (
            "n_train_folds", "n_validation_folds", "n_test_folds"
        ))
        if counts != expected:
            raise ValueError("Expected one validation fold and one test fold")
    validation_fold = (test_fold + offset) % n_folds
    if validation_fold == test_fold:
        raise ValueError("Validation and test folds must differ")
    test_idx = np.where(folds == test_fold)[0]
    val_idx = np.where(folds == validation_fold)[0]
    train_idx = np.where(
        (folds != test_fold) & (folds != validation_fold)
    )[0]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Training, validation and test sets must be nonempty")
    train_folds = [
        fold for fold in range(n_folds)
        if fold not in (test_fold, validation_fold)
    ]
    return train_idx, val_idx, test_idx, train_folds, validation_fold


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        raise ValueError("Metric arrays need equal lengths and at least two rows")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Metric arrays contain NaN or infinity")
    return {
        "n": int(len(y_true)),
        "PCC": float(pearsonr(y_true, y_pred)[0]),
        "SCC": float(spearmanr(y_true, y_pred)[0]),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def build_optimizer(model, config):
    name = config["optimizer"]
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if name not in optimizers:
        raise ValueError("Unsupported optimizer: {}".format(name))
    return optimizers[name](
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(config["weight_decay"]),
    )


def build_loss(config):
    if config["loss"] == "mse":
        return nn.MSELoss()
    if config["loss"] == "smooth_l1":
        return nn.SmoothL1Loss(beta=float(config["smooth_l1_beta"]))
    raise ValueError("Unsupported loss: {}".format(config["loss"]))


def build_scheduler(optimizer, config):
    if config["scheduler"] == "none":
        return None
    if config["scheduler"] != "plateau":
        raise ValueError("Unsupported scheduler: {}".format(config["scheduler"]))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        min_lr=1e-7,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, predictions, indices = [], [], []
    for esmc, chemberta, maccs, target, index in loader:
        prediction = model(
            esmc.to(device, non_blocking=True),
            chemberta.to(device, non_blocking=True),
            maccs.to(device, non_blocking=True),
        )
        ys.append(target.numpy())
        predictions.append(prediction.detach().cpu().numpy())
        indices.append(index.numpy())
    if not ys:
        raise ValueError("Evaluation loader is empty")
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(predictions)
    row_indices = np.concatenate(indices)
    return calculate_metrics(y_true, y_pred), y_true, y_pred, row_indices
