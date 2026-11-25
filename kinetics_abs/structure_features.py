from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .utils import ensure_dir, sha256_json, write_json_atomic

RBF_DIM = 16
GAP_BUCKET_EDGES = [1, 2, 3, 4, 8, 16, 32, 64, 128]
GAP_BUCKET_DIM = len(GAP_BUCKET_EDGES) + 1
EDGE_TYPE_DIM = 2  # flags: sequence edge, spatial edge
EDGE_ATTR_DIM = RBF_DIM + GAP_BUCKET_DIM + EDGE_TYPE_DIM
NODE_QUALITY_DIM = 4  # plddt_norm, plddt_available, coordinate_valid, source_predicted
EDGE_QUALITY_DIM = 4  # endpoint_quality, endpoint_quality_available, pae_quality, pae_available
DEFAULT_MAX_KNN = 16
DEFAULT_MAX_DISTANCE = 12.0


def structure_config_hash(config: dict[str, Any]) -> str:
    return sha256_json({
        "parser": "pdb_atom_ca_v1",
        "max_knn": int(config.get("max_knn", DEFAULT_MAX_KNN)),
        "max_distance": float(config.get("max_distance", DEFAULT_MAX_DISTANCE)),
        "rbf_dim": RBF_DIM,
        "rbf_centers": [float(x) for x in config.get("rbf_centers", np.linspace(0.0, DEFAULT_MAX_DISTANCE, RBF_DIM).tolist())],
        "rbf_width": float(config.get("rbf_width", DEFAULT_MAX_DISTANCE / RBF_DIM)),
        "gap_bucket_edges": GAP_BUCKET_EDGES,
        "assume_bfactor_is_plddt": bool(config.get("assume_bfactor_is_plddt", True)),
        "structure_source": config.get("structure_source", "unknown"),
        "chain_selection": config.get("chain_id", "most_ca"),
    })[:16]


def structure_key(sequence_id_value: str, structure_ref: str) -> str:
    return "struct_" + sha256_json({"sequence_id": sequence_id_value, "structure_ref": structure_ref})[:24]


def _candidate_paths(structure_ref: str, roots: list[str | Path]) -> list[Path]:
    ref = structure_ref.strip()
    candidates: list[Path] = []
    path = Path(ref)
    if path.is_absolute():
        candidates.append(path)
    filename = path.name
    for root in roots:
        root_path = Path(root)
        candidates.extend([
            root_path / ref,
            root_path / ref.lstrip("./"),
            root_path / "datasets" / "raw" / "alphafold" / filename,
            root_path / "data" / "raw" / "alphafold" / filename,
            root_path / "structures" / filename,
            root_path / filename,
        ])
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def find_structure_file(structure_ref: str, roots: list[str | Path]) -> Path | None:
    for candidate in _candidate_paths(structure_ref, roots):
        if candidate.is_file():
            return candidate
    return None


def _safe_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def _parse_pdb_ca(path: Path, chain_id: str | None = None) -> dict[str, Any]:
    residues: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "A"
            res_seq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, res_seq, icode)
            if key not in residues:
                residues[key] = {
                    "chain": chain,
                    "res_seq": res_seq,
                    "icode": icode,
                    "resname": resname,
                    "coord": np.array([
                        _safe_float(line[30:38]),
                        _safe_float(line[38:46]),
                        _safe_float(line[46:54]),
                    ], dtype=np.float32),
                    "bfactor": _safe_float(line[60:66]),
                }
                order.append(key)
    if not order:
        raise ValueError("No CA atoms found")
    # Select chain with most CA residues unless an explicit one-character chain ID is requested.
    chain_selection = "" if chain_id is None else str(chain_id).strip()
    if chain_selection.lower() in {"", "most_ca", "auto", "all"}:
        counts: dict[str, int] = {}
        for key in order:
            counts[residues[key]["chain"]] = counts.get(residues[key]["chain"], 0) + 1
        best_chain = max(counts, key=counts.get)  # type: ignore[arg-type]
        selected = [key for key in order if residues[key]["chain"] == best_chain]
    else:
        selected = [key for key in order if residues[key]["chain"] == chain_selection]
        if not selected:
            raise ValueError(f"Requested chain {chain_selection} not found")
    records = [residues[key] for key in selected]
    return {"chain": records[0]["chain"], "residues": records}


def _residue_to_one_letter(resname: str) -> str:
    try:
        from Bio.SeqUtils import seq1
        aa = seq1(resname.upper())
        return aa if aa else "X"
    except Exception:
        return "X"


def _gap_bucket(gap: int) -> int:
    for idx, edge in enumerate(GAP_BUCKET_EDGES):
        if gap <= edge:
            return idx
    return GAP_BUCKET_DIM - 1


def _rbf(distance: float, centers: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-(((distance - centers) / max(width, 1e-6)) ** 2)).astype(np.float32)


def _knn_pairs(coords: np.ndarray, valid_positions: np.ndarray, max_knn: int, max_distance: float) -> list[tuple[int, int, float]]:
    if len(valid_positions) == 0:
        return []
    points = coords[valid_positions]
    n = len(points)
    k = min(int(max_knn) + 1, n)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        pairs = []
        for local_i, neighbors in enumerate(indices):
            for dist, local_j in zip(distances[local_i], neighbors):
                if local_i == int(local_j):
                    continue
                if float(dist) <= float(max_distance) + 1e-6:
                    pairs.append((int(valid_positions[local_i]), int(valid_positions[int(local_j)]), float(dist)))
        return pairs
    except Exception:
        pairs = []
        # Chunked brute-force fallback; avoids materializing the full LxL matrix.
        chunk = 256
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            diff = points[start:end, None, :] - points[None, :, :]
            dist = np.sqrt((diff * diff).sum(axis=-1))
            for row in range(end - start):
                order = np.argsort(dist[row])[: int(max_knn) + 1]
                for local_j in order:
                    if int(local_j) == start + row:
                        continue
                    if float(dist[row, local_j]) <= float(max_distance) + 1e-6:
                        pairs.append((int(valid_positions[start + row]), int(valid_positions[int(local_j)]), float(dist[row, local_j])))
        return pairs


def build_graph(sequence: str, coordinates: np.ndarray, coordinate_mask: np.ndarray, node_quality: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    length = len(sequence)
    if coordinates.shape != (length, 3):
        raise ValueError("coordinates shape mismatch")
    if coordinate_mask.shape != (length,):
        raise ValueError("coordinate_mask shape mismatch")
    if node_quality.shape != (length, NODE_QUALITY_DIM):
        raise ValueError("node_quality shape mismatch")

    valid_positions = np.flatnonzero(coordinate_mask.astype(bool))
    pair_info: dict[tuple[int, int], dict[str, Any]] = {}

    # Sequence edges: adjacent sequence positions, both with valid coordinates.
    for i in range(length - 1):
        j = i + 1
        if coordinate_mask[i] and coordinate_mask[j]:
            pair_info[(i, j)] = {"sequence": True, "spatial": False}

    # Spatial kNN edges among valid coordinates.
    knn_pairs = _knn_pairs(coordinates.astype(np.float32), valid_positions, int(config.get("max_knn", DEFAULT_MAX_KNN)), float(config.get("max_distance", DEFAULT_MAX_DISTANCE)))
    spatial_dist = {}
    for i, j, dist in knn_pairs:
        key = (min(i, j), max(i, j))
        spatial_dist[key] = dist
        info = pair_info.setdefault(key, {"sequence": False, "spatial": False})
        info["spatial"] = True
    # Sequence edges may have a different orientation than the sorted pair key; normalize all to sorted pairs.
    normalized: dict[tuple[int, int], dict[str, Any]] = {}
    for (i, j), info in pair_info.items():
        key = (min(i, j), max(i, j))
        merged = normalized.setdefault(key, {"sequence": False, "spatial": False})
        merged["sequence"] = merged["sequence"] or info["sequence"]
        merged["spatial"] = merged["spatial"] or info["spatial"]
    pair_info = normalized

    centers = np.asarray(config.get("rbf_centers", np.linspace(0.0, DEFAULT_MAX_DISTANCE, RBF_DIM).tolist()), dtype=np.float32)
    width = float(config.get("rbf_width", DEFAULT_MAX_DISTANCE / RBF_DIM))
    edge_index = []
    edge_attr = []
    edge_quality = []
    for (i, j), info in sorted(pair_info.items()):
        delta = coordinates[i].astype(np.float32) - coordinates[j].astype(np.float32)
        distance = float(np.linalg.norm(delta))
        gap = abs(int(j) - int(i))
        bucket = _gap_bucket(gap)
        gap_onehot = np.zeros(GAP_BUCKET_DIM, dtype=np.float32)
        gap_onehot[bucket] = 1.0
        type_flags = np.array([1.0 if info["sequence"] else 0.0, 1.0 if info["spatial"] else 0.0], dtype=np.float32)
        attr = np.concatenate([_rbf(distance, centers, width), gap_onehot, type_flags]).astype(np.float32)
        # Endpoint quality: use available pLDDT values only; missing values are not treated as low quality.
        p_i = float(node_quality[i, 0]) if node_quality[i, 1] > 0 else np.nan
        p_j = float(node_quality[j, 0]) if node_quality[j, 1] > 0 else np.nan
        available = [p for p in [p_i, p_j] if np.isfinite(p)]
        endpoint_quality = float(min(available)) if available else 0.0
        endpoint_available = 1.0 if available else 0.0
        quality = np.array([endpoint_quality, endpoint_available, 1.0, 0.0], dtype=np.float32)  # PAE unavailable in PDB cache v1.
        for a, b in ((i, j), (j, i)):
            edge_index.append((a, b))
            edge_attr.append(attr)
            edge_quality.append(quality)
    if edge_index:
        edge_index_arr = np.asarray(edge_index, dtype=np.int32).T
        edge_attr_arr = np.asarray(edge_attr, dtype=np.float32)
        edge_quality_arr = np.asarray(edge_quality, dtype=np.float32)
    else:
        edge_index_arr = np.zeros((2, 0), dtype=np.int32)
        edge_attr_arr = np.zeros((0, EDGE_ATTR_DIM), dtype=np.float32)
        edge_quality_arr = np.zeros((0, EDGE_QUALITY_DIM), dtype=np.float32)
    return {
        "edge_index": edge_index_arr,
        "edge_attr": edge_attr_arr,
        "edge_quality": edge_quality_arr,
        "n_undirected_edges": len(pair_info),
    }


def _structure_key_dir(cache_root: str | Path, key: str, config: dict[str, Any]) -> Path:
    return ensure_dir(Path(cache_root) / key / structure_config_hash(config))


def cache_is_valid(directory: Path, config: dict[str, Any]) -> bool:
    meta_path = directory / "meta.json"
    required = ["coordinates.npy", "coordinate_mask.npy", "node_quality.npy", "edge_index.npy", "edge_attr.npy", "edge_quality.npy"]
    if not meta_path.is_file() or any(not (directory / name).is_file() for name in required):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        length = int(meta.get("sequence_length", -1))
        coordinates = np.load(directory / "coordinates.npy", mmap_mode="r")
        coordinate_mask = np.load(directory / "coordinate_mask.npy", mmap_mode="r")
        node_quality = np.load(directory / "node_quality.npy", mmap_mode="r")
        edge_index = np.load(directory / "edge_index.npy", mmap_mode="r")
        edge_attr = np.load(directory / "edge_attr.npy", mmap_mode="r")
        edge_quality = np.load(directory / "edge_quality.npy", mmap_mode="r")
    except Exception:
        return False
    ok = (
        meta.get("status") == "ok"
        and meta.get("config_hash") == structure_config_hash(config)
        and coordinates.shape == (length, 3)
        and coordinate_mask.shape == (length,)
        and node_quality.shape == (length, NODE_QUALITY_DIM)
        and edge_index.shape[0] == 2
        and edge_attr.shape[1] == EDGE_ATTR_DIM
        and edge_quality.shape[1] == EDGE_QUALITY_DIM
    )
    return bool(ok)


def encode_structure_record(record: dict[str, str], cache_root: str | Path, config: dict[str, Any], structure_roots: list[str | Path]) -> dict[str, Any]:
    sequence = record.get("sequence_actual", "")
    seq_id = record.get("sequence_id", "")
    ref = record.get("structure_ref", "")
    key = structure_key(seq_id, ref)
    directory = _structure_key_dir(cache_root, key, config)
    if cache_is_valid(directory, config):
        return {"structure_key": key, "status": "ok", "path": str(directory)}
    path = find_structure_file(ref, structure_roots)
    if path is None:
        failure = {"structure_key": key, "sequence_id": seq_id, "structure_ref": ref, "status": "structure_file_missing", "config_hash": structure_config_hash(config)}
        write_json_atomic(directory / "failure.json", failure)
        return {"structure_key": key, "status": "structure_file_missing", "path": str(directory)}
    try:
        parsed = _parse_pdb_ca(path, config.get("chain_id"))
        structure_sequence = "".join(_residue_to_one_letter(r["resname"]) for r in parsed["residues"])
        coords_struct = np.stack([r["coord"] for r in parsed["residues"]]).astype(np.float32)
        bfactor = np.array([r["bfactor"] for r in parsed["residues"]], dtype=np.float32)
        # Map target sequence positions to structure residue positions.
        mapping = np.full(len(sequence), -1, dtype=np.int32)
        matcher = difflib.SequenceMatcher(None, structure_sequence, sequence, autojunk=False)
        for block in matcher.get_matching_blocks():
            struct_start, target_start, size = block.a, block.b, block.size
            for offset in range(size):
                mapping[target_start + offset] = struct_start + offset
        coordinates = np.zeros((len(sequence), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(sequence), dtype=np.bool_)
        plddt = np.zeros(len(sequence), dtype=np.float32)
        plddt_available = np.zeros(len(sequence), dtype=np.float32)
        assume_bfactor = bool(config.get("assume_bfactor_is_plddt", True))
        for target_pos, struct_pos in enumerate(mapping):
            if struct_pos < 0:
                continue
            coord = coords_struct[struct_pos]
            if not np.isfinite(coord).all():
                continue
            coordinates[target_pos] = coord
            coordinate_mask[target_pos] = True
            if assume_bfactor and np.isfinite(bfactor[struct_pos]):
                plddt[target_pos] = float(np.clip(bfactor[struct_pos] / 100.0, 0.0, 1.0))
                plddt_available[target_pos] = 1.0
        source_predicted = 1.0 if str(config.get("structure_source", "unknown")).lower() in {"alphafold", "alphafold_plddt", "predicted"} else 0.0
        node_quality = np.stack([
            plddt,
            plddt_available,
            coordinate_mask.astype(np.float32),
            np.full(len(sequence), source_predicted, dtype=np.float32),
        ], axis=1).astype(np.float32)
        graph = build_graph(sequence, coordinates, coordinate_mask, node_quality, config)
        coverage = float(coordinate_mask.mean()) if len(sequence) else 0.0
        if coverage <= 0.0:
            raise ValueError("No target sequence positions mapped to structure coordinates")
        for name, array in [
            ("coordinates.npy", coordinates), ("coordinate_mask.npy", coordinate_mask), ("node_quality.npy", node_quality),
            ("edge_index.npy", graph["edge_index"]), ("edge_attr.npy", graph["edge_attr"]), ("edge_quality.npy", graph["edge_quality"]),
        ]:
            np.save(directory / name, array, allow_pickle=False)
        mapping_records = []
        for target_pos, struct_pos in enumerate(mapping):
            if struct_pos >= 0:
                rec = parsed["residues"][struct_pos]
                mapping_records.append({
                    "target_position": int(target_pos),
                    "structure_chain": rec["chain"],
                    "structure_res_seq": rec["res_seq"],
                    "structure_icode": rec["icode"],
                    "structure_resname": rec["resname"],
                    "bfactor": float(rec["bfactor"]) if np.isfinite(rec["bfactor"]) else None,
                })
        mapping_path = directory / "residue_mapping.json"
        write_json_atomic(mapping_path, {"sequence_id": seq_id, "structure_ref": ref, "chain": parsed["chain"], "mapping": mapping_records})
        meta = {
            "structure_key": key,
            "sequence_id": seq_id,
            "structure_ref": ref,
            "structure_file": str(path),
            "sequence_length": int(len(sequence)),
            "structure_residue_count": int(len(parsed["residues"])),
            "structure_chain": parsed["chain"],
            "mapped_positions": int((mapping >= 0).sum()),
            "coordinate_coverage": coverage,
            "n_valid_coordinates": int(coordinate_mask.sum()),
            "n_undirected_edges": int(graph["n_undirected_edges"]),
            "status": "ok",
            "config_hash": structure_config_hash(config),
            "config": config,
            "edge_attr_dim": EDGE_ATTR_DIM,
            "node_quality_dim": NODE_QUALITY_DIM,
            "edge_quality_dim": EDGE_QUALITY_DIM,
        }
        write_json_atomic(directory / "meta.json", meta)
        return {"structure_key": key, "status": "ok", "path": str(directory), "coordinate_coverage": coverage}
    except Exception as exc:
        failure = {
            "structure_key": key,
            "sequence_id": seq_id,
            "structure_ref": ref,
            "structure_file": str(path),
            "status": "structure_parse_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "config_hash": structure_config_hash(config),
        }
        write_json_atomic(directory / "failure.json", failure)
        return {"structure_key": key, "status": "structure_parse_failed", "path": str(directory), "error": failure["error"]}


def build_structure_cache(records: list[dict[str, str]], cache_root: str | Path, config: dict[str, Any], structure_roots: list[str | Path], limit: int | None = None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    unique = {}
    for record in records:
        ref = record.get("structure_ref", "")
        seq_id = record.get("sequence_id", "")
        if ref and seq_id:
            key = structure_key(seq_id, ref)
            unique.setdefault(key, record)
    if limit is not None:
        unique = dict(list(unique.items())[: int(limit)])
    statuses = {}
    failures = []
    for key, record in tqdm(unique.items(), desc="structure"):
        result = encode_structure_record(record, cache_root, config, structure_roots)
        statuses[key] = result.get("status", "failed")
        if result.get("status") not in {"ok"}:
            failures.append({k: v for k, v in result.items() if k != "path"})
    return statuses, failures


def load_structure_features(cache_root: str | Path, key: str, config: dict[str, Any], sequence_length: int | None = None) -> dict[str, Any]:
    directory = _structure_key_dir(cache_root, key, config)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return {"status": "missing", "structure_key": key}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        return {"status": meta.get("status", "failed"), "structure_key": key}
    length = int(sequence_length if sequence_length is not None else meta.get("sequence_length", -1))
    coordinates = np.load(directory / "coordinates.npy", mmap_mode="r")
    coordinate_mask = np.load(directory / "coordinate_mask.npy", mmap_mode="r")
    node_quality = np.load(directory / "node_quality.npy", mmap_mode="r")
    edge_index = np.load(directory / "edge_index.npy", mmap_mode="r")
    edge_attr = np.load(directory / "edge_attr.npy", mmap_mode="r")
    edge_quality = np.load(directory / "edge_quality.npy", mmap_mode="r")
    if length >= 0 and coordinates.shape[0] != length:
        return {"status": "length_mismatch", "structure_key": key}
    return {
        "status": "ok",
        "structure_key": key,
        "coordinates": coordinates,
        "coordinate_mask": coordinate_mask,
        "node_quality": node_quality,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_quality": edge_quality,
        "coordinate_coverage": float(meta.get("coordinate_coverage", 0.0)),
        "n_undirected_edges": int(meta.get("n_undirected_edges", 0)),
        "structure_file": meta.get("structure_file", ""),
        "structure_chain": meta.get("structure_chain", ""),
    }
