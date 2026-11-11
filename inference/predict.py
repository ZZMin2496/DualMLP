"""Predict enzyme kinetics from a CSV containing sequence and smiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from generate_features import read_input, resolve_device

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCRIPT = Path(__file__).with_name("generate_features.py").resolve()
sys.path.insert(0, str(ROOT))
FEATURES = (("esmc", 1152, np.float16), ("chemberta", 384, np.float32),
            ("maccs", 166, np.uint8))
RAW_COLUMNS = {"kcat": "pred_kcat_s_inv", "Km": "pred_Km_mM"}
RATIO_COLUMN = "pred_kcat_over_Km_s_inv_mM_inv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_task(task, paths, config, arrays, batch_size, device):
    import torch
    from training.model import DualMLP

    batches = []
    for fold, path in enumerate(paths):
        checkpoint = torch.load(str(path), map_location="cpu")
        if (checkpoint.get("task"), checkpoint.get("test_fold"),
                str(checkpoint.get("split_mode"))) != (task, fold, "811"):
            raise ValueError(f"Unexpected checkpoint task, fold or split: {path}")
        model_config = dict(checkpoint["model_config"])
        model_config.setdefault("activation", config["activation"])
        for key in ("esmc_dim", "chemberta_dim", "maccs_dim", "total_dim",
                    "hidden_dims", "activation", "bn_eps"):
            if model_config[key] != config[key]:
                raise ValueError(f"Unexpected {key} in {path}")
        model = DualMLP(model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        with torch.no_grad():
            values = np.concatenate([
                model(*(array[start:start + batch_size].to(device)
                        for array in arrays)).cpu().numpy()
                for start in range(0, len(arrays[0]), batch_size)
            ]).astype(np.float64)
        if values.shape != (len(arrays[0]),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid predictions: {path}")
        batches.append(values)
    return np.column_stack(batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models",
                        help="Checkpoint root (default: project models/)")
    parser.add_argument("--model-config", type=Path, default=ROOT / "training/configs.json",
                        help="Model config (default: project training/configs.json)")
    parser.add_argument("--task", choices=("all", "kcat", "Km"), default="all")
    parser.add_argument("--esmc-python", default=sys.executable)
    parser.add_argument("--esmc-cache", type=Path,
                        help="ESMC cache (default: encoders/esmc under model directory)")
    parser.add_argument("--chemberta-model", type=Path,
                        help="ChemBERTa directory (default: encoders/chemberta under model directory)")
    parser.add_argument("--device", default="auto", help="Device: auto (default), cpu, cuda, or cuda:N")
    parser.add_argument("--esmc-device", help="Override the device for ESMC only")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chemberta-batch-size", type=int, default=8)
    parser.add_argument("--save-fold-predictions", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.chemberta_batch_size) < 1:
        parser.error("Thread count and batch sizes must be positive")
    output = args.output.resolve()
    manifest = output.with_suffix(".manifest.json")
    if output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")
    if output.exists() or manifest.exists():
        parser.error("Output or manifest already exists. Choose a new output name.")

    tasks = ("kcat", "Km") if args.task == "all" else (args.task,)
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    paths = {task: [args.model_dir / task / f"fold_{fold}" / "best.pt"
                    for fold in range(10)] for task in tasks}
    for path in (path for task_paths in paths.values() for path in task_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    esmc_cache = (args.esmc_cache or args.model_dir / "encoders/esmc").resolve()
    chemberta = (args.chemberta_model or args.model_dir / "encoders/chemberta").resolve()
    for directory in (esmc_cache, chemberta):
        if not directory.is_dir() or not any(directory.iterdir()):
            raise FileNotFoundError(f"Encoder directory missing or empty: {directory}")

    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    esmc_device = args.esmc_device or (args.device if args.device == "auto" else str(device))

    with tempfile.TemporaryDirectory(prefix="dualmlp-") as temp:
        work = Path(temp)
        snapshot = work / "input.csv"
        snapshot.write_bytes(args.input.read_bytes())
        result = read_input(snapshot)
        input_hash = sha256(snapshot)
        generated = {RATIO_COLUMN} if len(tasks) == 2 else set()
        for task in tasks:
            generated.update((RAW_COLUMNS[task], f"pred_{task}_log10"))
            if args.save_fold_predictions:
                generated.add(f"fold_std_{task}_log10")
                generated.update(f"pred_{task}_log10_fold_{fold}" for fold in range(10))
        collisions = generated.intersection(result.columns)
        if collisions:
            parser.error(f"Input already contains output columns: {sorted(collisions)}")
        if "sample_id" not in result:
            result["sample_id"] = [f"sample_{i:08d}" for i in range(len(result))]
        for stage, interpreter, encoder, stage_device in (
            ("esmc", args.esmc_python, esmc_cache, esmc_device),
            ("chemberta_maccs", sys.executable, chemberta, str(device)),
        ):
            print(f"Generating {stage} (requested device: {stage_device})", flush=True)
            subprocess.run([interpreter, str(FEATURE_SCRIPT), "--stage", stage,
                            "--input", str(snapshot), "--output-dir", str(work),
                            "--encoder-path", str(encoder), "--device", stage_device, "--threads", str(args.threads),
                            "--chemberta-batch-size", str(args.chemberta_batch_size),
                            "--device-report", str(work / f"{stage}.device")],
                           check=True)
        esmc_device = (work / "esmc.device").read_text(encoding="utf-8").strip()

        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        arrays = []
        for name, dim, dtype in FEATURES:
            feature = np.load(work / f"{name}.npy", allow_pickle=False)
            if (feature.shape != (len(result), dim) or feature.dtype != dtype
                    or not np.isfinite(feature).all()):
                raise ValueError(f"Invalid {name} feature shape, dtype or values")
            arrays.append(torch.from_numpy(feature.astype(np.float32)))

        for task in tasks:
            matrix = predict_task(task, paths[task], config[task]["model_config"],
                                  arrays, args.batch_size, device)
            mean = matrix.mean(axis=1)
            result[f"pred_{task}_log10"] = mean
            result[RAW_COLUMNS[task]] = np.power(10.0, mean)
            if args.save_fold_predictions:
                for fold in range(10):
                    result[f"pred_{task}_log10_fold_{fold}"] = matrix[:, fold]
                result[f"fold_std_{task}_log10"] = matrix.std(axis=1)
        if len(tasks) == 2:
            result[RATIO_COLUMN] = np.power(10.0, result["pred_kcat_log10"] -
                                           result["pred_Km_log10"])
        original_units = [column for column in generated if "log10" not in column]
        values = result[original_units].to_numpy()
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Predictions outside the representable positive range")

    provenance = {
        "input_sha256": input_hash, "n_samples": len(result), "tasks": list(tasks),
        "esmc_cache": str(esmc_cache), "chemberta_model": str(chemberta),
        "feature_script_sha256": sha256(FEATURE_SCRIPT),
        "model_config_path": str(args.model_config.resolve()),
        "chemberta_batch_size": args.chemberta_batch_size,
        "device": str(device), "esmc_device": esmc_device,
        "ensemble": "Mean in log10 space, then exponentiate base 10",
        "checkpoints": {task: [{"path": str(path.resolve()), "sha256": sha256(path)}
                               for path in paths[task]] for task in tasks},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dualmlp-", dir=output.parent) as temp:
        temp_csv, temp_json = Path(temp) / "output.csv", Path(temp) / "manifest.json"
        result.to_csv(temp_csv, index=False)
        temp_json.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        os.link(temp_csv, output)
        try:
            os.link(temp_json, manifest)
        except OSError:
            output.unlink()
            raise
    print(f"Saved {len(result)} predictions to {output}")


if __name__ == "__main__":
    main()
