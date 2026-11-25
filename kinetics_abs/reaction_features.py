from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .utils import ensure_dir, reaction_id, sha256_json, write_json_atomic

RXNFP_DIM = 256
DEFAULT_RXNFP_MODEL = "bert_pretrained"


def reaction_config_hash(config: dict[str, Any]) -> str:
    return sha256_json({
        "rxnfp_model": config.get("rxnfp_model", DEFAULT_RXNFP_MODEL),
        "max_len": int(config.get("max_len", 512)),
        "pooling": "cls_v1",
        "canonicalization": "rxnfp_process_reaction_v1",
        "dim": RXNFP_DIM,
    })[:16]


def _validate_side(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        from rdkit import Chem
    except Exception:
        return True
    for part in smiles.split("."):
        part = part.strip()
        if not part:
            continue
        if Chem.MolFromSmiles(part) is None:
            return False
    return True


def normalize_reaction(reaction_smiles: str) -> dict[str, Any]:
    text = "" if reaction_smiles is None else str(reaction_smiles).strip()
    base = {
        "input_reaction_smiles": text,
        "canonical_reaction_smiles": "",
        "status": "ok",
        "reactants": "",
        "products": "",
        "notes": [],
    }
    if not text:
        base["status"] = "missing"
        return base
    if text.count(">>") != 1 and text.count(">") < 2:
        # RXNFP/RDKit convention accepts both A>>B and A>B>C; require a product side.
        base["status"] = "invalid_separator"
        return base
    # Normalize to RXNFP's three-field reaction format when possible.
    if ">>" in text:
        left, right = text.split(">>", 1)
        normalized_input = f"{left.strip()}>>{right.strip()}"
    else:
        parts = text.split(">")
        if len(parts) != 3:
            base["status"] = "invalid_separator"
            return base
        reactants, reagents, products = [p.strip() for p in parts]
        precursor = ".".join([p for p in [reactants, reagents] if p])
        normalized_input = f"{precursor}>>{products}"
    left, right = normalized_input.split(">>", 1)
    if not left.strip() or not right.strip():
        base["status"] = "incomplete_side"
        base["reactants"] = left.strip()
        base["products"] = right.strip()
        return base
    if not _validate_side(left) or not _validate_side(right):
        base["status"] = "smiles_parse_error"
        base["reactants"] = left.strip()
        base["products"] = right.strip()
        return base
    try:
        from rxnfp.tokenization import process_reaction
        canonical = process_reaction(normalized_input)
    except Exception as exc:
        canonical = ""
        base["notes"].append(f"process_reaction_error:{type(exc).__name__}")
    if not canonical:
        base["status"] = "canonicalization_failed"
        base["reactants"] = left.strip()
        base["products"] = right.strip()
        return base
    if ">>" not in canonical:
        base["status"] = "canonicalization_failed"
        return base
    c_left, c_right = canonical.split(">>", 1)
    base.update({
        "canonical_reaction_smiles": canonical,
        "reactants": c_left.strip(),
        "products": c_right.strip(),
    })
    return base


def load_rxnfp_generator(config: dict[str, Any], device: str) -> tuple[Any, Any, str, str]:
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer

    model_name = config.get("rxnfp_model", DEFAULT_RXNFP_MODEL)
    force_no_cuda = not str(device).startswith("cuda")
    model, tokenizer = get_default_model_and_tokenizer(model_name, force_no_cuda=force_no_cuda)
    generator = RXNBERTFingerprintGenerator(model, tokenizer, force_no_cuda=force_no_cuda)
    hidden = int(model.config.hidden_size)
    if hidden != RXNFP_DIM:
        raise RuntimeError(f"RXNFP hidden size {hidden} != {RXNFP_DIM}")
    return generator, tokenizer, model_name, str(model.config.max_position_embeddings)


def _token_length(tokenizer: Any, reaction_smiles: str) -> int:
    encoded = tokenizer.encode_plus(reaction_smiles, add_special_tokens=True)["input_ids"]
    return int(len(encoded))


def encode_reactions(records: list[dict[str, str]], cache_root: str | Path, config: dict[str, Any], device: str = "cpu", limit: int | None = None, batch_size: int = 32) -> tuple[dict[str, str], list[dict[str, Any]]]:
    unique: dict[str, dict[str, str]] = {}
    for record in records:
        rxn = record.get("reaction_smiles", "")
        if rxn:
            rid = reaction_id(rxn)
            unique.setdefault(rid, record)
    if limit is not None:
        unique = dict(list(unique.items())[: int(limit)])

    generator = tokenizer = model_name = max_position_embeddings = None
    statuses: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    pending: list[tuple[str, dict[str, Any], str]] = []

    for rid, record in tqdm(unique.items(), desc="reaction-prepare"):
        rxn = record.get("reaction_smiles", "")
        directory = _cache_dir(cache_root, rid, config)
        if cache_is_valid(directory, config):
            statuses[rid] = "ok"
            continue
        normalized = normalize_reaction(rxn)
        if normalized["status"] != "ok":
            _write_failure(directory, rid, rxn, normalized, config, "reaction_invalid")
            statuses[rid] = normalized["status"]
            failures.append({"reaction_id": rid, "status": normalized["status"], "reaction_smiles": rxn})
            continue
        canonical = normalized["canonical_reaction_smiles"]
        if generator is None:
            generator, tokenizer, model_name, max_position_embeddings = load_rxnfp_generator(config, device)
        max_len = int(config.get("max_len", int(max_position_embeddings)))
        try:
            token_length = _token_length(tokenizer, canonical)
        except Exception as exc:
            _write_failure(directory, rid, rxn, normalized, config, f"tokenize_error:{type(exc).__name__}:{exc}")
            statuses[rid] = "tokenize_error"
            failures.append({"reaction_id": rid, "status": "tokenize_error", "reaction_smiles": rxn})
            continue
        if token_length > min(max_len, int(max_position_embeddings)):
            normalized["status"] = "reaction_too_long"
            _write_failure(directory, rid, rxn, normalized, config, "reaction_too_long", token_length=token_length)
            statuses[rid] = "reaction_too_long"
            failures.append({"reaction_id": rid, "status": "reaction_too_long", "reaction_smiles": rxn, "token_length": token_length})
            continue
        pending.append((rid, normalized, canonical))

    for start in tqdm(range(0, len(pending), int(batch_size)), desc="reaction-rxnfp"):
        batch = pending[start:start + int(batch_size)]
        smiles_batch = [item[2] for item in batch]
        try:
            embeddings = generator.convert_batch(smiles_batch)
        except Exception as exc:
            for rid, normalized, canonical in batch:
                _write_failure(_cache_dir(cache_root, rid, config), rid, normalized["input_reaction_smiles"], normalized, config, f"rxnfp_error:{type(exc).__name__}:{exc}")
                statuses[rid] = "rxnfp_error"
                failures.append({"reaction_id": rid, "status": "rxnfp_error", "reaction_smiles": normalized["input_reaction_smiles"], "error": str(exc)})
            continue
        for (rid, normalized, canonical), emb in zip(batch, embeddings):
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.shape != (RXNFP_DIM,) or not np.isfinite(arr).all():
                _write_failure(_cache_dir(cache_root, rid, config), rid, normalized["input_reaction_smiles"], normalized, config, "rxnfp_invalid_output")
                statuses[rid] = "rxnfp_invalid_output"
                failures.append({"reaction_id": rid, "status": "rxnfp_invalid_output", "reaction_smiles": normalized["input_reaction_smiles"]})
                continue
            directory = _cache_dir(cache_root, rid, config)
            np.save(directory / "reaction_embedding.npy", arr, allow_pickle=False)
            meta = {
                "reaction_id": rid,
                "input_reaction_smiles": normalized["input_reaction_smiles"],
                "canonical_reaction_smiles": canonical,
                "reactants": normalized["reactants"],
                "products": normalized["products"],
                "status": "ok",
                "config_hash": reaction_config_hash(config),
                "config": config,
                "encoder": model_name,
                "dim": RXNFP_DIM,
                "token_length": _safe_token_length(tokenizer, canonical),
                "max_position_embeddings": int(max_position_embeddings),
            }
            write_json_atomic(directory / "meta.json", meta)
            statuses[rid] = "ok"
    return statuses, failures


def _safe_token_length(tokenizer: Any, reaction_smiles: str) -> int:
    try:
        return _token_length(tokenizer, reaction_smiles)
    except Exception:
        return -1


def _cache_dir(cache_root: str | Path, rid: str, config: dict[str, Any]) -> Path:
    return ensure_dir(Path(cache_root) / rid / reaction_config_hash(config))


def cache_is_valid(directory: Path, config: dict[str, Any]) -> bool:
    meta_path = directory / "meta.json"
    emb_path = directory / "reaction_embedding.npy"
    if not meta_path.is_file() or not emb_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = np.load(emb_path, allow_pickle=False)
    except Exception:
        return False
    return (
        meta.get("status") == "ok"
        and meta.get("config_hash") == reaction_config_hash(config)
        and emb.shape == (RXNFP_DIM,)
        and np.isfinite(emb).all()
    )


def _write_failure(directory: Path, rid: str, raw_rxn: str, normalized: dict[str, Any], config: dict[str, Any], reason: str, token_length: int | None = None) -> None:
    payload = {
        "reaction_id": rid,
        "input_reaction_smiles": raw_rxn,
        "canonical_reaction_smiles": normalized.get("canonical_reaction_smiles", ""),
        "reactants": normalized.get("reactants", ""),
        "products": normalized.get("products", ""),
        "status": normalized.get("status", "failed"),
        "failure_reason": reason,
        "token_length": token_length,
        "config_hash": reaction_config_hash(config),
        "config": config,
    }
    write_json_atomic(directory / "failure.json", payload)


def load_reaction_features(cache_root: str | Path, rid: str, config: dict[str, Any]) -> dict[str, Any]:
    directory = _cache_dir(cache_root, rid, config)
    meta_path = directory / "meta.json"
    emb_path = directory / "reaction_embedding.npy"
    if not meta_path.is_file() or not emb_path.is_file():
        return {"status": "missing", "reaction_id": rid}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        return {"status": meta.get("status", "failed"), "reaction_id": rid}
    emb = np.load(emb_path, allow_pickle=False).astype(np.float32)
    return {
        "status": "ok",
        "reaction_id": rid,
        "reaction_embedding": emb,
        "canonical_reaction_smiles": meta.get("canonical_reaction_smiles", ""),
        "token_length": int(meta.get("token_length", -1)),
        "encoder": meta.get("encoder", DEFAULT_RXNFP_MODEL),
    }
