from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .build_features import FEATURE_INDEX_COLUMNS  # noqa: F401  (documented index schema)
from .chemical_features import CHEM_TOTAL_DIM, chemical_config_hash
from .data_adapters import load_unified_csv
from .protein_features import ESMC_DIM, protein_config_hash
from .reaction_features import RXNFP_DIM, reaction_config_hash
from .structure_features import EDGE_ATTR_DIM, EDGE_QUALITY_DIM, NODE_QUALITY_DIM, structure_config_hash
from .utils import project_path

AUX_DIM = 8
SPLIT_COLUMNS = ["official_split", "sequence_40cluster", "sequence_60cluster", "sequence_80cluster", "sequence_99cluster"]


def assign_cluster_folds(frame: pd.DataFrame, n_folds: int = 10, seed: int = 42, cluster_column: str = "sequence_40cluster") -> pd.Series:
    """Deterministically assign sequence clusters to folds with balanced row counts."""
    clusters = frame[cluster_column].fillna("").astype(str)
    clusters = clusters.where(clusters != "", frame["sequence_id"].fillna("").astype(str))
    clusters = clusters.where(clusters != "", frame["sample_id"].fillna("").astype(str))
    work = pd.DataFrame({"cluster": clusters, "sample_id": frame["sample_id"].fillna("").astype(str)})
    counts = work.groupby("cluster").size().rename("n").reset_index()
    # Stable pseudo-random order, then greedy balanced assignment.
    counts["rand"] = counts["cluster"].map(lambda x: _stable_hash_int(f"{seed}:{cluster_column}:{x}"))
    counts = counts.sort_values(["n", "rand"], ascending=[False, True]).reset_index(drop=True)
    fold_counts = np.zeros(int(n_folds), dtype=np.int64)
    assignment: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        fold = int(np.argmin(fold_counts))
        assignment[row.cluster] = fold
        fold_counts[fold] += int(row.n)
    return clusters.map(assignment).fillna(0).astype(int)


def _stable_hash_int(text: str) -> int:
    import hashlib
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def split_mask_for_fold(frame: pd.DataFrame, fold_index: int, n_folds: int = 10, seed: int = 42, cluster_column: str = "sequence_40cluster") -> tuple[pd.Series, pd.Series, pd.Series]:
    folds = assign_cluster_folds(frame, n_folds=n_folds, seed=seed, cluster_column=cluster_column)
    test_fold = int(fold_index) % int(n_folds)
    val_fold = (int(fold_index) + 1) % int(n_folds)
    return (folds != test_fold) & (folds != val_fold), folds == val_fold, folds == test_fold


class KineticsDataset(Dataset):
    def __init__(
        self,
        unified_csv: str | Path,
        feature_index_csv: str | Path,
        cache_root: str | Path,
        config: dict[str, Any],
        *,
        task: str = "kcat",
        split_protocol: str = "official",
        split_name: str = "train",
        fold_index: int = 0,
        need_residue: bool = True,
        need_reaction: bool = True,
        need_structure: bool = True,
        require_label: bool = True,
        require_reaction: bool = False,
        require_structure: bool = False,
        max_protein_length: int = 2048,
        limit: int | None = None,
        seed: int = 42,
    ) -> None:
        self.unified_csv = Path(unified_csv)
        self.feature_index_csv = Path(feature_index_csv)
        self.cache_root = project_path(cache_root)
        self.config = config
        self.task = task
        self.split_protocol = split_protocol
        self.split_name = split_name
        self.fold_index = int(fold_index)
        self.need_residue = bool(need_residue)
        self.need_reaction = bool(need_reaction)
        self.need_structure = bool(need_structure)
        self.require_label = bool(require_label)
        self.require_reaction = bool(require_reaction)
        self.require_structure = bool(require_structure)
        self.max_protein_length = int(max_protein_length)
        self.seed = int(seed)

        unified = load_unified_csv(self.unified_csv)
        index = pd.read_csv(self.feature_index_csv, dtype=str, keep_default_na=False, low_memory=False)
        required_index_cols = ["sample_id", "protein_status", "chemical_status", "reaction_status", "structure_status"]
        missing_index_cols = [c for c in required_index_cols if c not in index.columns]
        if missing_index_cols:
            raise ValueError(f"feature_index missing columns: {missing_index_cols}")
        # The unified table already carries adapter-level reaction_status/structure_status;
        # feature-index build outcomes must stay unambiguous after the merge.
        feature_status_columns = ["protein_status", "chemical_status", "reaction_status", "structure_status"]
        index = index.rename(columns={c: f"feature_{c}" for c in feature_status_columns})
        frame = unified.merge(index, on="sample_id", how="left", suffixes=("", "_idx"))
        frame = frame[frame["task"] == task].copy()
        if frame.empty:
            raise ValueError(f"No rows for task {task} in {self.unified_csv}")
        split_cfg = config.get("split", {})

        # Split assignment is computed on the task-level frame so folds are stable across variants.
        if split_protocol == "official":
            frame["_split_train"] = frame["official_split"].astype(str).eq(split_name)
            frame["_split_val"] = pd.Series(False, index=frame.index)
            frame["_split_test"] = pd.Series(False, index=frame.index)
        elif split_protocol == "all":
            frame["_split_train"] = pd.Series(True, index=frame.index)
            frame["_split_val"] = pd.Series(False, index=frame.index)
            frame["_split_test"] = pd.Series(False, index=frame.index)
        elif split_protocol == "cluster10":
            train_mask, val_mask, test_mask = split_mask_for_fold(
                frame,
                fold_index=fold_index,
                n_folds=int(split_cfg.get("n_folds", config.get("n_folds", 10))),
                seed=seed,
                cluster_column=str(split_cfg.get("cluster_column", config.get("cluster_column", "sequence_40cluster"))),
            )
            frame["_split_train"] = train_mask
            frame["_split_val"] = val_mask
            frame["_split_test"] = test_mask
        else:
            raise ValueError(f"Unsupported split_protocol: {split_protocol}")

        split_col = {"train": "_split_train", "val": "_split_val", "test": "_split_test"}.get(split_name)
        if split_protocol == "all":
            selected = frame["_split_train"].astype(bool)
        elif split_protocol == "official" and split_name == "train":
            selected = frame["_split_train"].astype(bool)
        elif split_protocol == "official" and split_name == "val":
            selected = frame["official_split"].astype(str).eq("val")
        elif split_protocol == "official" and split_name == "test":
            selected = frame["official_split"].astype(str).eq("test")
        elif split_col is not None:
            selected = frame[split_col].astype(bool)
        else:
            raise ValueError(f"Unsupported split_name: {split_name}")

        protein_length = frame["sequence_actual"].fillna("").astype(str).str.len()
        valid = (
            frame["protein_available"].astype(str).eq("1")
            & frame["chem_available"].astype(str).eq("1")
            & frame["feature_protein_status"].astype(str).eq("ok")
            & frame["feature_chemical_status"].astype(str).eq("ok")
            & protein_length.gt(0)
            & protein_length.le(self.max_protein_length)
        )
        if self.require_label:
            valid &= frame["label_available"].astype(str).eq("1") & frame["label_log10"].astype(str).ne("")
        if self.require_reaction:
            valid &= frame["feature_reaction_status"].astype(str).eq("ok")
        if self.require_structure:
            valid &= frame["feature_structure_status"].astype(str).eq("ok")
        selected &= valid
        frame = frame[selected].copy()
        if limit is not None:
            frame = frame.iloc[: int(limit)].copy()
        if frame.empty:
            raise ValueError("No valid samples for the requested split/configuration")

        self.frame = frame.reset_index(drop=True)
        self.records = self.frame.to_dict(orient="records")
        self.lengths = np.array([int(len(r.get("sequence_actual", ""))) for r in self.records], dtype=np.int64)
        self.config_hashes = {
            "protein": protein_config_hash(self.config.get("protein", {})),
            "chemical": chemical_config_hash(self.config.get("chemical", {})),
            "reaction": reaction_config_hash(self.config.get("reaction", {})),
            "structure": structure_config_hash(self.config.get("structure", {})),
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        protein = self._load_protein(record)
        chemical = self._load_chemical(record)
        reaction = self._load_reaction(record) if self.need_reaction else {"reaction_embedding": torch.zeros(RXNFP_DIM, dtype=torch.float32), "available": 0.0}
        structure = self._load_structure(record, int(protein["sequence_length"])) if self.need_structure else self._empty_structure(int(protein["sequence_length"]))
        label_text = str(record.get("label_log10", "")).strip()
        try:
            label_value = float(label_text) if label_text else float("nan")
        except (TypeError, ValueError):
            label_value = float("nan")
        label_available = float(str(record.get("label_available", "")) == "1" and np.isfinite(label_value))
        item = {
            "sample_id": record.get("sample_id", ""),
            "sequence_id": record.get("sequence_id", ""),
            "uniprot": record.get("uniprot", ""),
            "official_split": record.get("official_split", ""),
            "sequence_40cluster": record.get("sequence_40cluster", ""),
            "sequence_60cluster": record.get("sequence_60cluster", ""),
            "sequence_80cluster": record.get("sequence_80cluster", ""),
            "sequence_99cluster": record.get("sequence_99cluster", ""),
            "protein_global": protein["global_mean"],
            "residue_embeddings": protein["residue_embeddings"],
            "need_residue": self.need_residue,
            "protein_length": int(protein["sequence_length"]),
            "protein_available": 1.0,
            "chemical": chemical["representation"],
            "chemical_available": float(chemical["available"]),
            "reaction": reaction["reaction_embedding"],
            "reaction_available": float(reaction["available"]),
            "coordinate_mask": structure["coordinate_mask"],
            "node_quality": structure["node_quality"],
            "edge_index": structure["edge_index"],
            "edge_attr": structure["edge_attr"],
            "edge_quality": structure["edge_quality"],
            "structure_available": float(structure["available"]),
            "aux": self._aux_vector(chemical["available"], reaction["available"], structure),
            "label": float(label_value),
            "label_available": label_available,
        }
        return item

    def _load_protein(self, record: dict[str, Any]) -> dict[str, Any]:
        from .protein_features import load_protein_features
        seq_id = record.get("sequence_id", "")
        feat = load_protein_features(self.cache_root, seq_id, self.config.get("protein", {}), need_residue=self.need_residue)
        if feat.get("status") != "ok":
            raise RuntimeError(f"Protein features unavailable for {record.get('sample_id')}: {feat.get('status')}")
        length = int(feat.get("sequence_length", len(record.get("sequence_actual", ""))))
        if self.need_residue:
            residue = np.asarray(feat["residue_embeddings"], dtype=np.float32)
        else:
            residue = np.zeros((0, ESMC_DIM), dtype=np.float32)
        return {
            "global_mean": torch.from_numpy(np.asarray(feat["global_mean"], dtype=np.float32)),
            "residue_embeddings": torch.from_numpy(residue),
            "sequence_length": length,
        }

    def _load_chemical(self, record: dict[str, Any]) -> dict[str, Any]:
        from .chemical_features import load_chemical_features
        cid = record.get("chemical_cache_key") or record.get("chemical_id") or ""
        feat = load_chemical_features(self.cache_root, cid, self.config.get("chemical", {}))
        if feat.get("status") != "ok":
            raise RuntimeError(f"Chemical features unavailable for {record.get('sample_id')}: {feat.get('status')}")
        return {"representation": torch.from_numpy(np.asarray(feat["representation"], dtype=np.float32)), "available": 1.0}

    def _load_reaction(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self.need_reaction:
            return {"reaction_embedding": torch.zeros(RXNFP_DIM, dtype=torch.float32), "available": 0.0}
        rid = record.get("reaction_cache_key") or record.get("reaction_id") or ""
        if not rid:
            return {"reaction_embedding": torch.zeros(RXNFP_DIM, dtype=torch.float32), "available": 0.0}
        from .reaction_features import load_reaction_features
        feat = load_reaction_features(self.cache_root, rid, self.config.get("reaction", {}))
        if feat.get("status") == "ok":
            return {"reaction_embedding": torch.from_numpy(np.asarray(feat["reaction_embedding"], dtype=np.float32)), "available": 1.0}
        return {"reaction_embedding": torch.zeros(RXNFP_DIM, dtype=torch.float32), "available": 0.0}

    def _load_structure(self, record: dict[str, Any], length: int) -> dict[str, Any]:
        if not self.need_structure:
            return self._empty_structure(length)
        key = record.get("structure_key", "")
        if not key:
            return self._empty_structure(length)
        from .structure_features import load_structure_features
        feat = load_structure_features(self.cache_root, key, self.config.get("structure", {}), sequence_length=length)
        if feat.get("status") != "ok":
            return self._empty_structure(length)
        coordinate_mask = np.asarray(feat["coordinate_mask"], dtype=np.bool_)
        if coordinate_mask.shape[0] != length:
            return self._empty_structure(length)
        return {
            "coordinate_mask": torch.from_numpy(coordinate_mask.copy()),
            "node_quality": torch.from_numpy(np.asarray(feat["node_quality"], dtype=np.float32)),
            "edge_index": torch.from_numpy(np.asarray(feat["edge_index"], dtype=np.int64).copy()),
            "edge_attr": torch.from_numpy(np.asarray(feat["edge_attr"], dtype=np.float32)),
            "edge_quality": torch.from_numpy(np.asarray(feat["edge_quality"], dtype=np.float32)),
            "available": float(coordinate_mask.any()),
            "coordinate_coverage": float(feat.get("coordinate_coverage", coordinate_mask.mean())),
            "n_undirected_edges": int(feat.get("n_undirected_edges", 0)),
        }

    def _empty_structure(self, length: int) -> dict[str, Any]:
        return {
            "coordinate_mask": torch.zeros(length, dtype=torch.bool),
            "node_quality": torch.zeros(length, NODE_QUALITY_DIM, dtype=torch.float32),
            "edge_index": torch.zeros(2, 0, dtype=torch.int64),
            "edge_attr": torch.zeros(0, EDGE_ATTR_DIM, dtype=torch.float32),
            "edge_quality": torch.zeros(0, EDGE_QUALITY_DIM, dtype=torch.float32),
            "available": 0.0,
            "coordinate_coverage": 0.0,
            "n_undirected_edges": 0,
        }

    def _aux_vector(self, chemical_available: float, reaction_available: float, structure: dict[str, Any]) -> torch.Tensor:
        coord = structure["coordinate_mask"].float()
        coverage = float(coord.mean().item()) if coord.numel() else 0.0
        node_quality = structure["node_quality"].float()
        if node_quality.numel():
            plddt = node_quality[:, 0]
            plddt_avail = node_quality[:, 1]
            mean_plddt = float((plddt * plddt_avail).sum().item() / max(1.0, float(plddt_avail.sum().item())))
            mean_plddt_available = float(plddt_avail.mean().item())
        else:
            mean_plddt = 0.0
            mean_plddt_available = 0.0
        n_edges = float(structure.get("n_undirected_edges", 0))
        aux = np.array([
            1.0,  # protein_available
            float(chemical_available),
            float(reaction_available),
            float(structure["available"]),
            coverage,
            mean_plddt,
            mean_plddt_available,
            float(np.log1p(n_edges) / 10.0),
        ], dtype=np.float32)
        return torch.from_numpy(aux)


class LengthBucketBatchSampler:
    def __init__(self, dataset: KineticsDataset, batch_size: int, seed: int = 42, shuffle: bool = True, bucket_multiplier: int = 50) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.bucket_multiplier = int(bucket_multiplier)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = np.argsort(self.dataset.lengths, kind="mergesort")
        mega_size = max(self.batch_size, self.batch_size * self.bucket_multiplier)
        batches = []
        for start in range(0, len(order), mega_size):
            chunk = order[start:start + mega_size]
            if self.shuffle:
                chunk = chunk[rng.permutation(len(chunk))]
            for batch_start in range(0, len(chunk), self.batch_size):
                batch = chunk[batch_start:batch_start + self.batch_size]
                if len(batch):
                    batches.append(batch.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))


def collate_kinetics_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    lengths = np.array([int(item["protein_length"]) for item in batch], dtype=np.int64)
    max_len = int(lengths.max()) if batch_size else 0
    any_residue = any(bool(item.get("need_residue", True)) for item in batch)
    residue_len = max_len if any_residue else 0
    residue_embeddings = torch.zeros(batch_size, residue_len, ESMC_DIM, dtype=torch.float32)
    residue_mask = torch.zeros(batch_size, residue_len, dtype=torch.bool)
    coordinate_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    node_quality = torch.zeros(batch_size, max_len, NODE_QUALITY_DIM, dtype=torch.float32)
    protein_global = torch.stack([item["protein_global"] for item in batch], dim=0)
    chemical = torch.stack([item["chemical"] for item in batch], dim=0)
    reaction = torch.stack([item["reaction"] for item in batch], dim=0)
    aux = torch.stack([item["aux"] for item in batch], dim=0)
    labels = torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32)
    label_available = torch.tensor([float(item["label_available"]) for item in batch], dtype=torch.float32)
    chemical_available = torch.tensor([float(item["chemical_available"]) for item in batch], dtype=torch.float32)
    reaction_available = torch.tensor([float(item["reaction_available"]) for item in batch], dtype=torch.float32)
    structure_available = torch.tensor([float(item["structure_available"]) for item in batch], dtype=torch.float32)

    edge_index_parts = []
    edge_attr_parts = []
    edge_quality_parts = []
    offset = 0
    for i, item in enumerate(batch):
        length = int(lengths[i])
        if any_residue:
            residue_embeddings[i, :length] = item["residue_embeddings"]
            residue_mask[i, :length] = True
        else:
            residue_mask[i, :0] = True
        coordinate_mask[i, :length] = item["coordinate_mask"]
        node_quality[i, :length] = item["node_quality"]
        edge_index = item["edge_index"]
        if edge_index.numel():
            edge_index_parts.append(edge_index + int(offset))
            edge_attr_parts.append(item["edge_attr"])
            edge_quality_parts.append(item["edge_quality"])
        offset += length

    if edge_index_parts:
        graph_edge_index = torch.cat(edge_index_parts, dim=1)
        graph_edge_attr = torch.cat(edge_attr_parts, dim=0)
        graph_edge_quality = torch.cat(edge_quality_parts, dim=0)
    else:
        graph_edge_index = torch.zeros(2, 0, dtype=torch.int64)
        graph_edge_attr = torch.zeros(0, EDGE_ATTR_DIM, dtype=torch.float32)
        graph_edge_quality = torch.zeros(0, EDGE_QUALITY_DIM, dtype=torch.float32)

    metadata = {
        "sample_id": [item["sample_id"] for item in batch],
        "sequence_id": [item["sequence_id"] for item in batch],
        "uniprot": [item["uniprot"] for item in batch],
        "official_split": [item["official_split"] for item in batch],
        "sequence_40cluster": [item["sequence_40cluster"] for item in batch],
        "sequence_60cluster": [item["sequence_60cluster"] for item in batch],
        "sequence_80cluster": [item["sequence_80cluster"] for item in batch],
        "sequence_99cluster": [item["sequence_99cluster"] for item in batch],
    }
    return {
        "residue_embeddings": residue_embeddings,
        "residue_mask": residue_mask,
        "protein_global": protein_global,
        "chemical": chemical,
        "reaction": reaction,
        "coordinate_mask": coordinate_mask,
        "node_quality": node_quality,
        "graph_edge_index": graph_edge_index,
        "graph_edge_attr": graph_edge_attr,
        "graph_edge_quality": graph_edge_quality,
        "graph_node_count": int(offset),
        "protein_available": torch.ones(batch_size, dtype=torch.float32),
        "chemical_available": chemical_available,
        "reaction_available": reaction_available,
        "structure_available": structure_available,
        "aux": aux,
        "labels": labels,
        "label_available": label_available,
        "metadata": metadata,
    }
