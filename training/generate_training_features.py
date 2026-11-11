#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARRAYS = {
    "esmc": ("esmc_dim", np.dtype("float16")),
    "chemberta": ("chemberta_dim", np.dtype("float32")),
    "maccs": ("maccs_dim", np.dtype("uint8")),
}


def validate_arrays(folder, expected_ids, model_config):
    ids = np.load(folder / "sample_id.npy", allow_pickle=False)
    if ids.dtype.kind != "U" or not np.array_equal(ids, expected_ids):
        raise ValueError(f"Sample IDs or row order differ: {folder}")
    arrays = {"sample_id": ids}
    for name, (dimension_key, dtype) in ARRAYS.items():
        path = folder / f"{name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = (len(ids), int(model_config[dimension_key]))
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{path}: expected {shape} {dtype}, got {array.shape} {array.dtype}")
        for start in range(0, len(ids), 4096):
            chunk = array[start:start + 4096]
            if not np.isfinite(chunk).all() or (name == "maccs" and (chunk > 1).any()):
                raise ValueError(f"Invalid feature values: {path}")
        arrays[name] = array
    return arrays


def compare_arrays(generated, reference):
    for name in ARRAYS:
        current, old = generated[name], reference[name]
        changed = 0
        max_difference = 0.0
        for start in range(0, len(current), 4096):
            a = current[start:start + 4096]
            b = old[start:start + 4096]
            changed += int(np.count_nonzero(a != b))
            max_difference = max(max_difference, float(np.max(
                np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        print(f"{name}: differing elements={changed}; max absolute difference={max_difference:.8g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("kcat", "Km", "all"))
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "features/regenerated")
    parser.add_argument("--esmc-python", default=sys.executable)
    parser.add_argument("--esmc-cache", type=Path, default=ROOT / "models/encoders/esmc")
    parser.add_argument("--chemberta-model", type=Path, default=ROOT / "models/encoders/chemberta")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--esmc-device")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chemberta-batch-size", type=int, default=8)
    parser.add_argument("--compare-existing", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.chemberta_batch_size) < 1:
        parser.error("Thread count and ChemBERTa batch size must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tasks = ("kcat", "Km") if args.task == "all" else (args.task,)
    output_root = args.output_root.resolve()
    destinations = {task: output_root / task for task in tasks}
    for task, destination in destinations.items():
        reference_dir = (ROOT / config[task]["features_dir"]).resolve()
        if destination.exists() or destination.is_symlink() or destination == reference_dir:
            parser.error(f"Choose a new output directory: {destination}")

    extractor = ROOT / "inference/generate_features.py"
    if not extractor.is_file():
        parser.error(f"Missing project feature extractor: {extractor}")
    for encoder in (args.esmc_cache, args.chemberta_model):
        if not encoder.is_dir() or not any(encoder.iterdir()):
            parser.error(f"Encoder directory missing or empty: {encoder}")

    output_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_config = config[task]
        source = ROOT / task_config["dataset_path"]
        frame = pd.read_csv(source, dtype=str, keep_default_na=False, low_memory=False)
        id_column = task_config["sample_id_column"]
        required = {"Sequence", "Smiles", id_column}
        if not required.issubset(frame.columns) or len(frame) != task_config["n_samples"]:
            raise ValueError(f"CSV columns or row count do not match training configuration: {source}")
        ids = frame[id_column]
        if ids.str.strip().eq("").any() or not ids.is_unique:
            raise ValueError(f"Sample IDs must be nonempty and unique: {source}")
        expected_ids = ids.to_numpy(dtype=str)
        model_config = task_config["model_config"]
        reference = None
        if args.compare_existing:
            reference = validate_arrays(ROOT / task_config["features_dir"],
                                        expected_ids, model_config)

        with tempfile.TemporaryDirectory(prefix=f".{task}-", dir=output_root) as temp:
            work = Path(temp)
            snapshot = work / "input.csv"
            staging = work / "arrays"
            staging.mkdir()
            frame[["Sequence", "Smiles"]].rename(
                columns={"Sequence": "sequence", "Smiles": "smiles"}
            ).to_csv(snapshot, index=False)
            for stage, interpreter, encoder, device in (
                ("esmc", args.esmc_python, args.esmc_cache, args.esmc_device or args.device),
                ("chemberta_maccs", sys.executable, args.chemberta_model, args.device),
            ):
                print(f"{task}: generating {stage} on {device}", flush=True)
                subprocess.run(
                    [interpreter, str(extractor), "--input", str(snapshot),
                     "--stage", stage, "--encoder-path", str(encoder.resolve()),
                     "--output-dir", str(staging), "--device", device,
                     "--threads", str(args.threads),
                     "--chemberta-batch-size", str(args.chemberta_batch_size)],
                    check=True,
                )
            np.save(staging / "sample_id.npy", expected_ids, allow_pickle=False)
            generated = validate_arrays(staging, expected_ids, model_config)
            if reference is not None:
                compare_arrays(generated, reference)
            del generated
            destination = destinations[task]
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            os.rename(staging, destination)
            print(f"{task}: wrote {len(expected_ids)} rows to {destination}", flush=True)


if __name__ == "__main__":
    main()
