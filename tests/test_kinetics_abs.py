import numpy as np
import torch

from kinetics_abs.dataset import AUX_DIM
from kinetics_abs.evaluate import compute_metrics
from kinetics_abs.model import build_model
from kinetics_abs.chemical_features import CHEM_TOTAL_DIM
from kinetics_abs.protein_features import ESMC_DIM
from kinetics_abs.reaction_features import RXNFP_DIM
from kinetics_abs.structure_features import EDGE_ATTR_DIM, EDGE_QUALITY_DIM, NODE_QUALITY_DIM, _parse_pdb_ca, build_graph


def test_compute_metrics_basic():
    metrics = compute_metrics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 4.0, 4.0]))
    assert metrics["n"] == 4
    assert abs(metrics["rmse"] - np.sqrt(0.25)) < 1e-6
    assert abs(metrics["mae"] - 0.25) < 1e-6
    assert np.isfinite(metrics["pcc"])


def test_structure_build_graph_synthetic():
    sequence = "ACDE"
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    coordinate_mask = np.array([True, True, True, False], dtype=bool)
    node_quality = np.zeros((4, NODE_QUALITY_DIM), dtype=np.float32)
    graph = build_graph(sequence, coordinates, coordinate_mask, node_quality, {"max_knn": 4, "max_distance": 5.0})
    assert graph["edge_index"].shape[0] == 2
    assert graph["n_undirected_edges"] == 2
    assert graph["edge_attr"].shape == (4, EDGE_ATTR_DIM)
    assert graph["edge_quality"].shape == (4, EDGE_QUALITY_DIM)
    assert graph["edge_index"].max() < 4


def test_fusion_model_graph_modes_forward():
    torch.manual_seed(0)
    batch_size, length = 2, 5
    batch = {
        "residue_embeddings": torch.randn(batch_size, length, ESMC_DIM),
        "residue_mask": torch.ones(batch_size, length, dtype=torch.bool),
        "chemical": torch.randn(batch_size, CHEM_TOTAL_DIM),
        "reaction": torch.randn(batch_size, RXNFP_DIM),
        "chemical_available": torch.ones(batch_size),
        "reaction_available": torch.ones(batch_size),
        "aux": torch.randn(batch_size, AUX_DIM),
        "coordinate_mask": torch.ones(batch_size, length, dtype=torch.bool),
        "node_quality": torch.rand(batch_size, length, NODE_QUALITY_DIM),
    }
    edges = []
    for i in range(batch_size):
        offset = i * length
        for j in range(length - 1):
            edges.extend([[offset + j, offset + j + 1], [offset + j + 1, offset + j]])
    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
    n_edges = edge_index.shape[1]
    batch.update({
        "graph_edge_index": edge_index,
        "graph_edge_attr": torch.randn(n_edges, EDGE_ATTR_DIM),
        "graph_edge_quality": torch.rand(n_edges, EDGE_QUALITY_DIM),
    })
    for graph_mode in ["none", "uncontrolled", "quality"]:
        model = build_model({"architecture": "fusion", "graph_mode": graph_mode, "n_queries": 2, "num_heads": 2})
        output = model(batch)
        assert output.shape == (batch_size, 1)
        assert torch.isfinite(output).all()


def _pdb_atom_line(chain: str, res_seq: int, resname: str, x: float, y: float, z: float, bfactor: float) -> str:
    line = [" "] * 80
    line[0:6] = "ATOM  "
    line[6:11] = f"{1:5d}"
    line[12:16] = " CA "
    line[17:20] = resname
    line[21] = chain
    line[22:26] = f"{res_seq:4d}"
    line[30:38] = f"{x:8.3f}"
    line[38:46] = f"{y:8.3f}"
    line[46:54] = f"{z:8.3f}"
    line[60:66] = f"{bfactor:6.2f}"
    return "".join(line) + "\n"


def test_parse_pdb_ca_most_ca_chain(tmp_path):
    path = tmp_path / "two_chains.pdb"
    rows = []
    for i in range(3):
        rows.append(_pdb_atom_line("A", i + 1, "ALA", float(i), 0.0, 0.0, 90.0))
    rows.append(_pdb_atom_line("B", 1, "GLY", 10.0, 0.0, 0.0, 50.0))
    path.write_text("".join(rows), encoding="utf-8")
    auto = _parse_pdb_ca(path, "most_ca")
    assert auto["chain"] == "A"
    assert len(auto["residues"]) == 3
    explicit = _parse_pdb_ca(path, "B")
    assert explicit["chain"] == "B"
    assert len(explicit["residues"]) == 1
