import json
import os
from os import path as osp

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import remove_self_loops, to_undirected

from openfgl.config import SUPPORTED_DATASETS
from openfgl.data.global_dataset_loader import load_global_dataset
from openfgl.data.simulation import subgraph_fl_louvain


class FGLDataset(Dataset):
    """Prepare/load the Louvain client graphs used by the main experiment."""

    def __init__(self, args):
        if args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported FedGMoE dataset: {args.dataset}")
        if args.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        self.args = args
        super().__init__(args.root)
        self.load_data()

    @property
    def global_root(self):
        return osp.join(self.root, "global")

    @property
    def distrib_root(self):
        return osp.join(self.root, "distrib")

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        simulation = f"subgraph_fl_louvain_{self.args.louvain_resolution:g}"
        return osp.join(
            self.distrib_root,
            f"{simulation}_{self.args.dataset}_client_{self.args.num_clients}",
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"data_{client_id}.pt" for client_id in range(self.args.num_clients)]

    @staticmethod
    def _normalize_graph(data):
        data.x = data.x.to(torch.float32)
        data.y = data.y.squeeze()
        if getattr(data, "edge_attr", None) is not None:
            data.edge_index, data.edge_attr = remove_self_loops(
                *to_undirected(data.edge_index, data.edge_attr)
            )
        else:
            data.edge_index = remove_self_loops(to_undirected(data.edge_index))[0]
        data.edge_index = data.edge_index.to(torch.int64)
        return data

    def get_client_data(self, client_id):
        path = osp.join(self.processed_dir, f"data_{client_id}.pt")
        try:
            data = torch.load(path, weights_only=False)
        except TypeError:
            data = torch.load(path)
        return self._normalize_graph(data)

    def process(self):
        global_dataset = load_global_dataset(self.global_root, self.args.dataset)
        os.makedirs(self.processed_dir, exist_ok=True)
        local_data = subgraph_fl_louvain(self.args, global_dataset)
        for client_id, data in enumerate(local_data):
            torch.save(data, osp.join(self.processed_dir, f"data_{client_id}.pt"))
        with open(
            osp.join(self.processed_dir, "description.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(vars(self.args), file, indent=2)

    def load_data(self):
        self.local_data = [
            self.get_client_data(client_id)
            for client_id in range(self.args.num_clients)
        ]
        global_dataset = load_global_dataset(self.global_root, self.args.dataset)
        self.global_data = self._normalize_graph(global_dataset[0])
        self.global_data.num_global_classes = global_dataset.num_classes
