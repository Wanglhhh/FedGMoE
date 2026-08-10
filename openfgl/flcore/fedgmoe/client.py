import torch

from openfgl.flcore.base import BaseClient


class FedGMoEClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)
        self.lambda_aux = args.lambda_aux
        self.calibration_warmup_rounds = args.calibration_warmup_rounds
        self.calibration_route = None
        self.calibration_information = 0.0
        self.calibration_precision = None
        self.task.loss_fn = self._loss_fn

    def _loss_fn(self, embedding, logits, labels, mask):
        classification = self.task.default_loss_fn(logits[mask], labels[mask])
        auxiliary = self.task.model.moe_regularization_loss(
            labels,
            mask,
            lambda_aux=self.lambda_aux,
        )
        return classification + auxiliary

    def _calibrate_on_validation_set(self):
        if self.message_pool["round"] < self.calibration_warmup_rounds:
            return
        self.task.model.eval()
        with torch.no_grad():
            self.task.model(self.task.data)
        (
            self.calibration_route,
            self.calibration_information,
            self.calibration_precision,
        ) = self.task.model.estimate_calibrated_route(
            self.task.data.y,
            self.task.val_mask,
        )
        if self.calibration_route is not None:
            self.task.model.set_local_calibrated_gate(
                self.calibration_route,
                int(self.task.val_mask.sum().item()),
                information=self.calibration_information,
                precision=self.calibration_precision,
            )

    def execute(self):
        server = self.message_pool["server"]
        self.task.model.load_state_dict(server["state_dict"])
        self.task.model.set_calibrated_global_gate(server["calibrated_global_gate"])
        self.task.model.set_calibrated_global_information(
            server["calibrated_global_information"]
        )
        self.task.model.set_calibrated_global_precision(
            server["calibrated_global_precision"]
        )
        self.calibration_route = None
        self.calibration_information = 0.0
        self.calibration_precision = None
        self.task.train()
        self._calibrate_on_validation_set()

    def send_message(self):
        message = {
            "num_samples": self.task.num_samples,
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.task.model.state_dict().items()
            },
        }
        if self.calibration_route is not None:
            message.update(
                {
                    "calibration_reliability": self.calibration_route.detach().cpu().clone(),
                    "calibration_samples": int(self.task.val_mask.sum().item()),
                    "calibration_information": self.calibration_information,
                    "calibration_precision": self.calibration_precision.detach().cpu().clone(),
                }
            )
        self.message_pool[f"client_{self.client_id}"] = message
