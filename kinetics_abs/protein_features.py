from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .utils import ensure_dir, project_path, sequence_id, sha256_json, write_json_atomic

ESMC_DIM = 1152
DEFAULT_ESMC_MODEL = "esmc_600m"
DEFAULT_MAX_LEN = 2048
DEFAULT_OVERLAP = 128


def protein_config_hash(config: dict[str, Any]) -> str:
    payload = {
        "model_name": config.get("model_name", DEFAULT_ESMC_MODEL),
        "max_len": int(config.get("max_len", DEFAULT_MAX_LEN)),
        "overlap": int(config.get("overlap", DEFAULT_OVERLAP)),
        "dim": ESMC_DIM,
        "extract": "last_hidden_residue_mean_windows_v1",
    }
    return sha256_json(payload)[:16]


def _window_bounds(length: int, max_len: int, overlap: int) -> list[tuple[int, int]]:
    if length <= max_len:
        return [(0, length)]
    stride = max(1, max_len - overlap)
    starts = list(range(0, length, stride))
    if starts[-1] + max_len < length:
        starts.append(max(0, length - max_len))
    bounds = []
    for start in starts:
        end = min(length, start + max_len)
        bounds.append((start, end))
        if end == length:
            break
    # Deduplicate while preserving order.
    return list(dict.fromkeys(bounds))


def _residue_embeddings_from_output(output: Any, seq_len: int) -> np.ndarray:
    import torch

    embedding = output.embeddings
    if embedding is None:
        raise RuntimeError("ESMC returned no embeddings")
    if embedding.dim() == 3:
        embedding = embedding[0]
    if embedding.size(0) >= seq_len + 2:
        embedding = embedding[1:seq_len + 1]
    elif embedding.size(0) >= seq_len + 1:
        embedding = embedding[1:seq_len + 1]
    elif embedding.size(0) == seq_len:
        pass
    else:
        raise RuntimeError(f"Unexpected ESMC embedding shape {tuple(embedding.shape)} for length {seq_len}")
    if embedding.shape[0] != seq_len:
        raise RuntimeError(f"ESMC residue count mismatch: {embedding.shape[0]} != {seq_len}")
    return embedding.detach().float().cpu().numpy()


def encode_sequence_windows(model: Any, sequence: str, device: Any, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    max_len = int(config.get("max_len", DEFAULT_MAX_LEN))
    overlap = int(config.get("overlap", DEFAULT_OVERLAP))
    bounds = _window_bounds(len(sequence), max_len, overlap)
    total = np.zeros((len(sequence), ESMC_DIM), dtype=np.float32)
    counts = np.zeros(len(sequence), dtype=np.float32)
    window_records = []
    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
    for start, end in bounds:
        subseq = sequence[start:end]
        protein_tensor = model.encode(ESMProtein(sequence=subseq))
        if isinstance(protein_tensor, ESMProteinError):
            raise RuntimeError(f"ESMProteinError: {protein_tensor}")
        with torch.no_grad():
            if str(device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model.logits(protein_tensor, logits_config)
            else:
                output = model.logits(protein_tensor, logits_config)
        emb = _residue_embeddings_from_output(output, len(subseq))
        total[start:end] += emb
        counts[start:end] += 1.0
        window_records.append({"start": start, "end": end, "length": len(subseq)})
    if not np.all(counts > 0):
        raise RuntimeError("Window coverage incomplete")
    residue_embeddings = (total / counts[:, None]).astype(np.float16)
    global_mean = residue_embeddings.astype(np.float32).mean(axis=0)
    meta = {
        "n_windows": len(bounds),
        "windows": window_records,
        "coverage_min": float(counts.min()),
        "coverage_max": float(counts.max()),
    }
    return residue_embeddings, global_mean.astype(np.float32), meta


def _patch_esmc_tokenizer_for_transformers() -> None:
    # esm's EsmSequenceTokenizer overrides special-token properties without setters,
    # which breaks transformers' PreTrainedTokenizerBase.__init__ setattr loop.
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        from transformers.tokenization_utils_base import SpecialTokensMixin
    except Exception:
        return
    for name in ["cls_token", "eos_token", "mask_token", "pad_token"]:
        base_prop = getattr(SpecialTokensMixin, name, None)
        current = getattr(EsmSequenceTokenizer, name, None)
        if isinstance(base_prop, property) and base_prop.fset is not None and (not isinstance(current, property) or current.fset is None):
            setattr(EsmSequenceTokenizer, name, base_prop)


def load_esmc_model(config: dict[str, Any], device: str) -> tuple[Any, str]:
    import os
    import torch
    from esm.models.esmc import ESMC

    _patch_esmc_tokenizer_for_transformers()
    model_name = config.get("model_name", DEFAULT_ESMC_MODEL)
    cache_dir = project_path(config.get("hf_cache_dir", "models_abs/encoders/esmc"))
    ensure_dir(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    model = ESMC.from_pretrained(model_name, device=torch.device(device)).eval()
    model = model.float()
    return model, str(cache_dir)


def _cache_dir(cache_root: str | Path, seq_id: str, config: dict[str, Any]) -> Path:
    return ensure_dir(Path(cache_root) / seq_id / protein_config_hash(config))


def cache_is_valid(directory: Path, sequence: str, config: dict[str, Any]) -> bool:
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = {
        "sequence": sequence,
        "sequence_length": len(sequence),
        "config_hash": protein_config_hash(config),
        "status": "ok",
    }
    return all(meta.get(k) == v for k, v in expected.items()) and (directory / "residue_embeddings.npy").is_file() and (directory / "global_mean.npy").is_file()


def encode_and_cache_sequence(sequence: str, cache_root: str | Path, model: Any, device: str, config: dict[str, Any]) -> dict[str, Any]:
    seq_id = sequence_id(sequence)
    directory = _cache_dir(cache_root, seq_id, config)
    if cache_is_valid(directory, sequence, config):
        return {"sequence_id": seq_id, "status": "cache_hit", "path": str(directory)}
    try:
        residue_embeddings, global_mean, window_meta = encode_sequence_windows(model, sequence, device, config)
    except Exception as exc:  # record failure, continue with other sequences
        failure = {
            "sequence_id": seq_id,
            "sequence_length": len(sequence),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(directory / "failure.json", failure)
        return failure
    np.save(directory / "residue_embeddings.npy", residue_embeddings, allow_pickle=False)
    np.save(directory / "global_mean.npy", global_mean, allow_pickle=False)
    np.save(directory / "residue_indices.npy", np.arange(len(sequence), dtype=np.int32), allow_pickle=False)
    meta = {
        "sequence_id": seq_id,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "config_hash": protein_config_hash(config),
        "config": config,
        "status": "ok",
        "encoder": config.get("model_name", DEFAULT_ESMC_MODEL),
        "dim": ESMC_DIM,
        **window_meta,
    }
    write_json_atomic(directory / "meta.json", meta)
    return {"sequence_id": seq_id, "status": "ok", "path": str(directory), "n_windows": window_meta["n_windows"]}


def build_protein_cache(records: list[dict[str, str]], cache_root: str | Path, config: dict[str, Any], device: str = "cpu", limit: int | None = None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    unique = {}
    for record in records:
        sid = record.get("sequence_id", "")
        seq = record.get("sequence_actual", "")
        if sid and seq and sid not in unique:
            unique[sid] = seq
    if limit is not None:
        unique = dict(list(unique.items())[: int(limit)])
    model = None
    statuses = {}
    failures = []
    for sid, seq in tqdm(unique.items(), desc="protein-esmc"):
        directory = _cache_dir(cache_root, sid, config)
        if cache_is_valid(directory, seq, config):
            statuses[sid] = "ok"
            continue
        if model is None:
            model, _ = load_esmc_model(config, device)
        result = encode_and_cache_sequence(seq, cache_root, model, device, config)
        statuses[sid] = result.get("status", "failed")
        if result.get("status") != "ok" and result.get("status") != "cache_hit":
            failures.append(result)
    return statuses, failures


def load_protein_features(cache_root: str | Path, seq_id: str, config: dict[str, Any], need_residue: bool = False) -> dict[str, Any]:
    directory = _cache_dir(cache_root, seq_id, config)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return {"status": "missing", "sequence_id": seq_id}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        return {"status": meta.get("status", "failed"), "sequence_id": seq_id}
    global_mean = np.load(directory / "global_mean.npy", allow_pickle=False).astype(np.float32)
    out = {
        "status": "ok",
        "sequence_id": seq_id,
        "global_mean": global_mean,
        "sequence_length": int(meta.get("sequence_length", global_mean.shape[0] if global_mean.ndim == 1 else -1)),
        "n_windows": int(meta.get("n_windows", 1)),
    }
    if need_residue:
        residue_embeddings = np.load(directory / "residue_embeddings.npy", mmap_mode="r", allow_pickle=False)
        residue_indices = np.load(directory / "residue_indices.npy", allow_pickle=False)
        out["residue_embeddings"] = residue_embeddings
        out["residue_indices"] = residue_indices
    return out
