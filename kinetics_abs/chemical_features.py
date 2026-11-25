from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .utils import chemical_id, ensure_dir, sha256_json, write_json_atomic

CHEMBERTA_DIM = 384
MACCS_DIM = 166
CHEM_TOTAL_DIM = CHEMBERTA_DIM + MACCS_DIM
DEFAULT_CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"
DEFAULT_MAX_LEN = 512


def chemical_config_hash(config: dict[str, Any]) -> str:
    return sha256_json({
        "chemberta_model": config.get("chemberta_model", DEFAULT_CHEMBERTA_MODEL),
        "max_len": int(config.get("max_len", DEFAULT_MAX_LEN)),
        "pooling": "mean_valid_non_special_v1",
        "maccs": "bits_1_to_166_or_v1",
        "canonicalization": "rdkit_isomeric_sorted_components_v1",
        "dim": [CHEMBERTA_DIM, MACCS_DIM],
    })[:16]


def canonicalize_chemical_context(smiles: str) -> tuple[str, str]:
    if not smiles:
        return "", "missing"
    try:
        from rdkit import Chem
    except Exception as exc:
        return smiles, f"rdkit_unavailable:{type(exc).__name__}"
    components = []
    for part in smiles.split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return "", "parse_error"
        components.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    if not components:
        return "", "empty_after_parse"
    components = sorted(set(components))
    return ".".join(components), "ok"


def chemical_cache_id(smiles: str, scope: str = "") -> tuple[str, str, str]:
    canonical, canon_status = canonicalize_chemical_context(smiles)
    if canon_status == "ok":
        return chemical_id(canonical, scope), canonical, canon_status
    return chemical_id(smiles, scope), canonical, canon_status


def maccs_bits_for_smiles(smiles: str) -> tuple[np.ndarray, str]:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    bits = np.zeros(167, dtype=np.uint8)
    parsed_any = False
    for part in smiles.split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return np.zeros(MACCS_DIM, dtype=np.uint8), "parse_error"
        parsed_any = True
        key_bits = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        arr = np.fromiter((1 if c == "1" else 0 for c in key_bits), dtype=np.uint8, count=167)
        bits = np.maximum(bits, arr)
    if not parsed_any:
        return np.zeros(MACCS_DIM, dtype=np.uint8), "missing"
    return bits[1:167].astype(np.uint8), "ok"


def load_chemberta(config: dict[str, Any], device: str) -> tuple[Any, Any, str]:
    import os
    from transformers import AutoModel, AutoTokenizer

    model_name = config.get("chemberta_model", DEFAULT_CHEMBERTA_MODEL)
    cache_dir = config.get("hf_cache_dir") or config.get("chemberta_cache_dir") or "models_abs/encoders/chemberta"
    cache_path = ensure_dir(cache_dir) if not Path(cache_dir).exists() else Path(cache_dir)
    os.environ.setdefault("HF_HOME", str(cache_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_path), local_files_only=False)
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_path), local_files_only=False)
    if int(model.config.hidden_size) != CHEMBERTA_DIM:
        raise RuntimeError(f"ChemBERTa hidden size {model.config.hidden_size} != {CHEMBERTA_DIM}")
    import torch
    model = model.to(device).eval()
    return tokenizer, model, model_name


def _mean_pool(last_hidden: Any, attention_mask: Any, input_ids: Any, tokenizer: Any) -> Any:
    import torch

    mask = attention_mask.bool().clone()
    special = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    special = {x for x in special if x is not None}
    if special:
        special_tensor = torch.tensor(sorted(special), device=input_ids.device)
        mask &= ~torch.isin(input_ids, special_tensor)
    mask = mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return summed / counts


def encode_chemical_context(smiles: str, scope: str, config: dict[str, Any], tokenizer: Any = None, model: Any = None, device: str = "cpu") -> dict[str, Any]:
    import torch

    canonical, canon_status = canonicalize_chemical_context(smiles)
    cid = chemical_id(canonical if canon_status == "ok" else smiles, scope)
    max_len = int(config.get("max_len", DEFAULT_MAX_LEN))
    base = {
        "chemical_id": cid,
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "chemical_scope": scope,
        "canonical_status": canon_status,
        "chemberta_token_length": 0,
        "status": "ok",
    }
    if canon_status != "ok":
        base["status"] = canon_status
        return base
    maccs, maccs_status = maccs_bits_for_smiles(canonical)
    base["maccs_status"] = maccs_status
    if maccs_status != "ok":
        base["status"] = maccs_status
        return base

    if tokenizer is None or model is None:
        tokenizer, model, _ = load_chemberta(config, device)
    encoded = tokenizer(canonical, add_special_tokens=True)["input_ids"]
    token_length = len(encoded)
    base["chemberta_token_length"] = int(token_length)
    if token_length > max_len:
        base["status"] = "chemberta_too_long"
        return base
    inputs = tokenizer([canonical], return_tensors="pt", padding=True, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    pooled = _mean_pool(output.last_hidden_state, inputs["attention_mask"], inputs["input_ids"], tokenizer)
    embedding = pooled[0].detach().float().cpu().numpy().astype(np.float32)
    if embedding.shape != (CHEMBERTA_DIM,) or not np.isfinite(embedding).all():
        base["status"] = "chemberta_invalid_output"
        return base
    base["chemberta_embedding"] = embedding
    base["maccs"] = maccs
    base["representation"] = np.concatenate([embedding, maccs.astype(np.float32)], axis=0).astype(np.float32)
    return base


def _cache_dir(cache_root: str | Path, cid: str, config: dict[str, Any]) -> Path:
    return ensure_dir(Path(cache_root) / cid / chemical_config_hash(config))


def cache_is_valid(directory: Path, config: dict[str, Any]) -> bool:
    meta_path = directory / "meta.json"
    npz_path = directory / "features.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(npz_path)
    except Exception:
        return False
    ok = (
        meta.get("config_hash") == chemical_config_hash(config)
        and meta.get("status") == "ok"
        and data["chemberta_embedding"].shape == (CHEMBERTA_DIM,)
        and data["maccs"].shape == (MACCS_DIM,)
        and data["representation"].shape == (CHEM_TOTAL_DIM,)
    )
    return bool(ok)


def encode_and_cache_chemical(record: dict[str, str], cache_root: str | Path, config: dict[str, Any], tokenizer: Any = None, model: Any = None, device: str = "cpu") -> dict[str, Any]:
    smiles = record.get("chem_context_smiles", "")
    scope = record.get("chemical_scope", "")
    cid, _, _ = chemical_cache_id(smiles, scope)
    directory = _cache_dir(cache_root, cid, config)
    if cache_is_valid(directory, config):
        return {"chemical_id": cid, "status": "ok", "path": str(directory)}
    try:
        result = encode_chemical_context(smiles, scope, config, tokenizer=tokenizer, model=model, device=device)
    except Exception as exc:
        result = {"chemical_id": cid, "status": f"failed:{type(exc).__name__}:{exc}"}
    if result.get("status") == "ok" and "representation" in result:
        np.savez_compressed(
            directory / "features.npz",
            chemberta_embedding=result["chemberta_embedding"].astype(np.float32),
            maccs=result["maccs"].astype(np.uint8),
            representation=result["representation"].astype(np.float32),
        )
        meta = {k: v for k, v in result.items() if k not in {"chemberta_embedding", "maccs", "representation"}}
        meta["config_hash"] = chemical_config_hash(config)
        meta["config"] = config
        write_json_atomic(directory / "meta.json", meta)
        return {"chemical_id": cid, "status": "ok", "path": str(directory)}
    failure = {k: v for k, v in result.items() if k not in {"chemberta_embedding", "maccs", "representation"}}
    failure["config_hash"] = chemical_config_hash(config)
    write_json_atomic(directory / "failure.json", failure)
    return {"chemical_id": cid, "status": failure.get("status", "failed"), "path": str(directory)}


def build_chemical_cache(records: list[dict[str, str]], cache_root: str | Path, config: dict[str, Any], device: str = "cpu", limit: int | None = None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    unique = {}
    for record in records:
        smiles = record.get("chem_context_smiles", "")
        scope = record.get("chemical_scope", "")
        if smiles:
            cid, _, _ = chemical_cache_id(smiles, scope)
            unique.setdefault(cid, record)
    if limit is not None:
        unique = dict(list(unique.items())[: int(limit)])
    tokenizer = model = None
    statuses = {}
    failures = []
    for cid, record in tqdm(unique.items(), desc="chemical"):
        directory = _cache_dir(cache_root, cid, config)
        if cache_is_valid(directory, config):
            statuses[cid] = "ok"
            continue
        if tokenizer is None or model is None:
            tokenizer, model, _ = load_chemberta(config, device)
        result = encode_and_cache_chemical(record, cache_root, config, tokenizer=tokenizer, model=model, device=device)
        statuses[cid] = result.get("status", "failed")
        if result.get("status") != "ok":
            failures.append(result)
    return statuses, failures


def load_chemical_features(cache_root: str | Path, cid: str, config: dict[str, Any]) -> dict[str, Any]:
    directory = _cache_dir(cache_root, cid, config)
    meta_path = directory / "meta.json"
    npz_path = directory / "features.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        return {"status": "missing", "chemical_id": cid}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        return {"status": meta.get("status", "failed"), "chemical_id": cid}
    data = np.load(npz_path)
    return {
        "status": "ok",
        "chemical_id": cid,
        "representation": data["representation"].astype(np.float32),
        "chemberta_embedding": data["chemberta_embedding"].astype(np.float32),
        "maccs": data["maccs"].astype(np.uint8),
        "canonical_smiles": meta.get("canonical_smiles", ""),
        "chemberta_token_length": int(meta.get("chemberta_token_length", 0)),
    }
