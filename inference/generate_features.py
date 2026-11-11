import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ESMC_ALIAS = "esmc_600m"
ESMC_DIM = 1152
ESMC_MAX_LEN = 2048
CHEMBERTA_DIM = 384
CHEMBERTA_MAX_LEN = 512

def log(message=""):
    print(message, flush=True)

def resolve_device(value):
    import torch

    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device {value!r}; use auto, cpu, cuda, or cuda:N") from exc
    if device.type == "cpu" and device.index is None:
        return device
    if device.type != "cuda":
        raise ValueError(f"Unsupported device {value!r}; use auto, cpu, cuda, or cuda:N")
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this Python environment")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is not visible to this process")
    return torch.device(f"cuda:{index}")


def load_esmc(device):
    os.environ["HF_HOME"] = str(ESMC_CACHE)
    os.environ["HF_HUB_CACHE"] = str(ESMC_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from esm.models.esmc import ESMC
    log("Loading ESMC: {}".format(ESMC_ALIAS))
    model = ESMC.from_pretrained(ESMC_ALIAS).to(device)
    model.eval()
    return model

def esmc_embed_one(model, sequence, device):
    import torch
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    protein_tensor = model.encode(ESMProtein(sequence=sequence))
    if isinstance(protein_tensor, ESMProteinError):
        raise RuntimeError("ESMProteinError: {}".format(protein_tensor))
    config = LogitsConfig(sequence=True, return_embeddings=True)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.logits(protein_tensor, config)
        else:
            output = model.logits(protein_tensor, config)
    embedding = output.embeddings
    if embedding is None:
        raise RuntimeError("No ESMC embeddings returned")
    if embedding.dim() == 3:
        embedding = embedding[0]
    if embedding.size(0) >= len(sequence) + 2:
        residue_embedding = embedding[1:len(sequence) + 1]
    elif embedding.size(0) >= 3:
        residue_embedding = embedding[1:-1]
    else:
        residue_embedding = embedding
    vector = (
        residue_embedding.mean(dim=0).detach().float().cpu().numpy()
        .astype(np.float16, copy=False)
    )
    if vector.shape != (ESMC_DIM,) or not np.all(np.isfinite(vector)):
        raise RuntimeError("Invalid ESMC vector: {}".format(vector.shape))
    return vector

def load_chemberta(device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not CHEMBERTA_MODEL_PATH.exists():
        raise FileNotFoundError(str(CHEMBERTA_MODEL_PATH))
    tokenizer = AutoTokenizer.from_pretrained(
        str(CHEMBERTA_MODEL_PATH), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(CHEMBERTA_MODEL_PATH), local_files_only=True
    )
    if int(model.config.hidden_size) != CHEMBERTA_DIM:
        raise RuntimeError("ChemBERTa hidden size mismatch")
    model = model.to(device)
    model.eval()
    return tokenizer, model

def chemberta_encode_batch(smiles, tokenizer, model, device):
    import torch

    inputs = tokenizer(
        list(smiles),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CHEMBERTA_MAX_LEN,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    vector = (
        output.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        .astype(np.float32, copy=False)
    )
    if vector.shape != (len(smiles), CHEMBERTA_DIM):
        raise RuntimeError("ChemBERTa output shape mismatch: {}".format(vector.shape))
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("ChemBERTa output contains NaN/Inf")
    return vector

def rdkit_maccs(smiles, remove_first_bit):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("RDKit cannot parse SMILES: {}".format(smiles))
    bits = np.fromiter(
        (1 if value == "1" else 0 for value in MACCSkeys.GenMACCSKeys(molecule).ToBitString()),
        dtype=np.uint8,
        count=167,
    )
    if bits.shape != (167,):
        raise RuntimeError("Unexpected MACCS shape: {}".format(bits.shape))
    return bits[1:] if remove_first_bit else bits


def read_input(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty or not {"sequence", "smiles"}.issubset(frame.columns):
        raise ValueError("Expected a nonempty CSV with sequence and smiles columns")
    for column in ("sequence", "smiles"):
        empty = frame[column].str.strip().eq("")
        if empty.any():
            raise ValueError(f"Empty {column} in data row {int(np.flatnonzero(empty)[0]) + 1}")
    return frame


def generate(frame, stage, encoder_path, output, chemberta_batch_size, device):
    global ESMC_CACHE, CHEMBERTA_MODEL_PATH
    if stage == "esmc":
        ESMC_CACHE = encoder_path
        keys = ["".join(seq.split()).upper()[:ESMC_MAX_LEN] for seq in frame.sequence]
        model = load_esmc(device).float().eval()
        vectors = {key: esmc_embed_one(model, key, device) for key in dict.fromkeys(keys)}
        np.save(output / "esmc.npy", np.stack([vectors[key] for key in keys]))
        return

    CHEMBERTA_MODEL_PATH = encoder_path
    keys = frame.smiles.str.strip().tolist()
    unique = list(dict.fromkeys(keys))
    fingerprints = {key: rdkit_maccs(key, remove_first_bit=True) for key in unique}
    tokenizer, model = load_chemberta(device)
    vectors = {}
    for start in range(0, len(unique), chemberta_batch_size):
        batch = unique[start:start + chemberta_batch_size]
        try:
            encoded = chemberta_encode_batch(batch, tokenizer, model, device)
        except RuntimeError:
            encoded = [chemberta_encode_batch([key], tokenizer, model, device)[0] for key in batch]
        vectors.update(zip(batch, encoded))
    np.save(output / "chemberta.npy", np.stack([vectors[key] for key in keys]))
    np.save(output / "maccs.npy", np.stack([fingerprints[key] for key in keys]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stage", choices=("esmc", "chemberta_maccs"), required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chemberta-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.chemberta_batch_size) < 1:
        parser.error("Thread count and ChemBERTa batch size must be positive")
    if not args.encoder_path.is_dir():
        parser.error(f"Encoder directory missing: {args.encoder_path}")
    # Set the offline cache before importing any model libraries.
    if args.stage == "esmc":
        for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            os.environ[name] = str(args.encoder_path.resolve())
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"{args.stage} resolved device: {device}")
    generate(read_input(args.input), args.stage, args.encoder_path.resolve(),
             args.output_dir, args.chemberta_batch_size, device)
    if args.device_report is not None:
        args.device_report.write_text(str(device), encoding="utf-8")


if __name__ == "__main__":
    main()
