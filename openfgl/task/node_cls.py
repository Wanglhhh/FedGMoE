import os
from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F

from openfgl.flcore.fedgmoe.models import FedGMoEModel
from openfgl.task.base import BaseTask
from openfgl.utils.basic_utils import idx_to_mask_tensor, mask_tensor_to_idx


class NodeClsTask(BaseTask):
    """The only task in this repository: FedGMoE node classification."""

    @property
    def default_model(self):
        return FedGMoEModel(
            input_dim=self.num_feats,
            hid_dim=self.args.hid_dim,
            output_dim=self.num_global_classes,
            dropout=self.args.dropout,
            smooth_hops=self.args.smooth_hops,
            bwgnn_order=self.args.bwgnn_order,
            gate_temperature=self.args.gate_temperature,
            expert_reliability_floor=self.args.expert_reliability_floor,
            calibration_base_temperature=self.args.calibration_base_temperature,
            client_calibration_shrinkage=self.args.client_calibration_shrinkage,
            calibration_mirror_steps=self.args.calibration_mirror_steps,
            calibration_mirror_lr=self.args.calibration_mirror_lr,
            calibration_mirror_prior=self.args.calibration_mirror_prior,
            calibration_evidence_floor=self.args.calibration_evidence_floor,
            calibration_precision_ridge=self.args.calibration_precision_ridge,
            calibration_consensus_steps=self.args.calibration_consensus_steps,
            calibration_consensus_lr=self.args.calibration_consensus_lr,
            calibration_consensus_prior=self.args.calibration_consensus_prior,
        )

    @property
    def num_samples(self):
        return self.data.x.size(0)

    @property
    def num_feats(self):
        return self.data.x.size(1)

    @property
    def num_global_classes(self):
        return int(self.data.num_global_classes)

    @staticmethod
    def default_loss_fn(logits, labels):
        return F.cross_entropy(logits, labels)

    def loss_fn(self, embedding, logits, labels, mask):
        return self.default_loss_fn(logits[mask], labels[mask])

    def train(self):
        self.model.train()
        for _ in range(self.args.num_epochs):
            self.optim.zero_grad()
            embedding, logits = self.model(self.data)
            loss = self.loss_fn(embedding, logits, self.data.y, self.train_mask)
            loss.backward()
            self.optim.step()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            embedding, logits = self.model(self.data)
            result = {}
            for split, mask in (
                ("train", self.train_mask),
                ("val", self.val_mask),
                ("test", self.test_mask),
            ):
                loss = self.loss_fn(embedding, logits, self.data.y, mask)
                accuracy = (logits[mask].argmax(dim=-1) == self.data.y[mask]).float().mean()
                result[f"loss_{split}"] = float(loss.detach().cpu())
                result[f"{split}_accuracy"] = float(accuracy.detach().cpu())

        info = "".join(f"\t{key}: {value:.4f}" for key, value in result.items())
        print(f"[client {self.client_id}]" + info)
        return result

    @property
    def default_split(self):
        if self.args.dataset in {"Cora", "CiteSeer", "PubMed"}:
            return 0.2, 0.4, 0.4
        return 0.5, 0.25, 0.25

    @property
    def split_dir(self):
        return osp.join(self.data_dir, "node_cls", "default_split")

    def load_train_val_test_split(self):
        paths = {
            split: osp.join(self.split_dir, f"{split}_{self.client_id}.pt")
            for split in ("train", "val", "test")
        }
        if all(osp.exists(path) for path in paths.values()):
            masks = {split: torch.load(path) for split, path in paths.items()}
        else:
            masks = dict(
                zip(
                    ("train", "val", "test"),
                    self._stratified_split(),
                )
            )
            os.makedirs(self.split_dir, exist_ok=True)
            for split, mask in masks.items():
                torch.save(mask.cpu(), paths[split])

        self.train_mask = masks["train"].to(self.device).bool()
        self.val_mask = masks["val"].to(self.device).bool()
        self.test_mask = masks["test"].to(self.device).bool()

    def _stratified_split(self):
        train_ratio, val_ratio, _ = self.default_split
        train_mask = idx_to_mask_tensor([], self.num_samples)
        val_mask = idx_to_mask_tensor([], self.num_samples)
        test_mask = idx_to_mask_tensor([], self.num_samples)

        for class_id in range(self.num_global_classes):
            indices = mask_tensor_to_idx(self.data.y == class_id)
            np.random.shuffle(indices)
            train_end = int(train_ratio * len(indices))
            val_end = int((train_ratio + val_ratio) * len(indices))
            train_mask |= idx_to_mask_tensor(indices[:train_end], self.num_samples)
            val_mask |= idx_to_mask_tensor(indices[train_end:val_end], self.num_samples)
            test_mask |= idx_to_mask_tensor(indices[val_end:], self.num_samples)
        return train_mask, val_mask, test_mask
