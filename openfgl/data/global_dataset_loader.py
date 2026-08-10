from os import path as osp

from openfgl.config import SUPPORTED_DATASETS


def load_global_dataset(root, dataset):
    """Load one of the six datasets used by the FedGMoE main experiment."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported FedGMoE dataset: {dataset}")

    if dataset in {"Cora", "CiteSeer", "PubMed"}:
        from torch_geometric.datasets import Planetoid

        return Planetoid(root=osp.join(root, "subgraph_fl"), name=dataset)

    if dataset in {"Roman-empire", "Minesweeper"}:
        from torch_geometric.datasets import HeterophilousGraphDataset

        return HeterophilousGraphDataset(
            root=osp.join(root, "subgraph_fl"),
            name=dataset,
        )

    from torch_geometric.datasets import Actor

    return Actor(root=osp.join(root, "subgraph_fl", "Actor"))
