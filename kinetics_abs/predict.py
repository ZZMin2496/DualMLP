from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import KineticsDataset, LengthBucketBatchSampler, collate_kinetics_batch
from .model import baseline_input, build_model
from .train import _forward, _model_needs
from .utils import project_path, read_yaml


def predict_file(
    config_path: str | Path,
    checkpoint_path: str | Path,
    unified_csv: str | Path,
    feature_index_csv: str | Path,
    output_csv: str | Path,
    *,
    cache_root: str | Path | None = None,
    task: str | None = None,
    limit: int | None = None,
    batch_size: int = 16,
    device: str = "cpu",
) -> pd.DataFrame:
    config = read_yaml(config_path)
    checkpoint = torch.load(project_path(checkpoint_path), map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", config)
    model_config = saved_config.get("model", config.get("model", {}))
    data_cfg = dict(config.get("data", {}))
    if cache_root is not None:
        data_cfg["cache_root"] = str(cache_root)
    predict_config = dict(saved_config)
    predict_config["data"] = data_cfg
    predict_config["task"] = task or predict_config.get("task", data_cfg.get("task", "kcat"))
    needs = _model_needs(model_config)
    dataset = KineticsDataset(
        unified_csv=project_path(unified_csv),
        feature_index_csv=project_path(feature_index_csv),
        cache_root=project_path(data_cfg.get("cache_root", "features_cache")),
        config=predict_config,
        task=str(predict_config["task"]),
        split_protocol="all",
        split_name="train",
        need_residue=needs["need_residue"],
        need_reaction=needs["need_reaction"],
        need_structure=needs["need_structure"],
        require_label=False,
        require_reaction=False,
        require_structure=False,
        max_protein_length=int(data_cfg.get("max_protein_length", 2048)),
        limit=limit,
        seed=42,
    )
    sampler = LengthBucketBatchSampler(dataset, batch_size=int(batch_size), seed=42, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_kinetics_batch, num_workers=0)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = _forward(model, batch, model_config).squeeze(-1).float().cpu().numpy()
            for i, sample_id in enumerate(batch["metadata"]["sample_id"]):
                rows.append({
                    "sample_id": sample_id,
                    "sequence_id": batch["metadata"]["sequence_id"][i],
                    "uniprot": batch["metadata"]["uniprot"][i],
                    "prediction_log10": float(pred[i]),
                    "reaction_available": float(batch["reaction_available"][i].item()),
                    "structure_available": float(batch["structure_available"][i].item()),
                })
    frame = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame
