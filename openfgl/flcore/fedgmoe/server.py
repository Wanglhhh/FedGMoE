import torch

from openfgl.flcore.base import BaseServer


class FedGMoEServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGMoEServer, self).__init__(args, global_data, data_dir, message_pool, device)

    @staticmethod
    def _weighted_average_state_dicts(state_dicts, weights):
        averaged = {}
        for key in state_dicts[0].keys():
            value = None
            for state_dict, weight in zip(state_dicts, weights):
                weighted = state_dict[key].to(dtype=torch.float32, device=weights[0].device) * weight
                value = weighted.clone() if value is None else value + weighted
            averaged[key] = value
        return averaged

    @staticmethod
    def _precision_consensus(routes, precisions, prior, steps, lr, prior_strength):
        """Find a simplex route that respects directional client precision."""
        if not routes or not precisions:
            return prior
        device = prior.device
        dtype = prior.dtype
        routes = [route.to(device=device, dtype=dtype) for route in routes]
        precisions = [precision.to(device=device, dtype=dtype) for precision in precisions]
        total_precision = sum(precisions)
        # Route Hessians are scaled by validation-set size. Normalize their
        # common magnitude before optimization so the mirror step is stable
        # across datasets while retaining every directional trade-off.
        mean_scale = torch.diagonal(total_precision).sum().clamp_min(1e-12)
        mean_scale = mean_scale / max(prior.numel() - 1, 1)
        precisions = [precision / mean_scale for precision in precisions]
        prior_weight = float(prior_strength)
        logits = prior.clamp_min(1e-12).log().detach()
        for _ in range(max(1, int(steps))):
            logits = logits.detach().requires_grad_(True)
            route = torch.softmax(logits, dim=0)
            objective = sum(
                0.5 * torch.dot(route - local_route, precision @ (route - local_route))
                for local_route, precision in zip(routes, precisions)
            )
            if prior_weight > 0:
                objective = objective + prior_weight * (
                    route.clamp_min(1e-12)
                    * (route.clamp_min(1e-12).log() - prior.clamp_min(1e-12).log())
                ).sum()
            gradient = torch.autograd.grad(objective, logits)[0]
            with torch.no_grad():
                logits = logits - float(lr) * gradient
        return torch.softmax(logits.detach(), dim=0)

    def execute(self):
        client_ids = range(self.args.num_clients)
        total = sum(self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in client_ids)
        weights = torch.tensor(
            [self.message_pool[f"client_{client_id}"]["num_samples"] / total for client_id in client_ids],
            device=self.device,
            dtype=torch.float32,
        )
        states = [
            self.message_pool[f"client_{client_id}"]["state_dict"]
            for client_id in client_ids
        ]
        averaged = self._weighted_average_state_dicts(states, weights)
        self.task.model.load_state_dict(averaged)

        calibration_messages = [
            self.message_pool[f"client_{client_id}"]
            for client_id in client_ids
            if "calibration_reliability" in self.message_pool[f"client_{client_id}"]
        ]
        if calibration_messages:
            information_weights = torch.tensor(
                [message["calibration_samples"] for message in calibration_messages],
                device=self.device,
                dtype=torch.float32,
            )
            information = torch.tensor(
                [message.get("calibration_information", 0.0) for message in calibration_messages],
                device=self.device,
                dtype=torch.float32,
            )
            global_information = (
                (information_weights * information).sum()
                / information_weights.sum().clamp_min(1e-12)
            )
            precision_messages = [
                message
                for message in calibration_messages
                if message.get("calibration_precision") is not None
            ]
            if len(precision_messages) != len(calibration_messages):
                raise RuntimeError("FedGMoE calibration requires client precision matrices.")
            routes = [
                message["calibration_reliability"].to(self.device)
                for message in precision_messages
            ]
            precisions = [
                message["calibration_precision"].to(self.device)
                for message in precision_messages
            ]
            prior = self.task.model.calibrated_global_gate.detach().to(self.device)
            calibration_gate = self._precision_consensus(
                routes,
                precisions,
                prior,
                self.task.model.calibration_consensus_steps,
                self.task.model.calibration_consensus_lr,
                self.task.model.calibration_consensus_prior,
            )
            global_precision = sum(precisions) / len(precisions)
            self.task.model.set_calibrated_global_precision(global_precision)
            self.task.model.set_calibrated_global_gate(calibration_gate)
            self.task.model.set_calibrated_global_information(global_information)
    def send_message(self):
        self.message_pool["server"] = {
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.task.model.state_dict().items()
            },
            "calibrated_global_gate": self.task.model.calibrated_global_gate.detach().cpu().clone(),
            "calibrated_global_information": self.task.model.calibrated_global_information.detach().cpu().clone(),
            "calibrated_global_precision": self.task.model.calibrated_global_precision.detach().cpu().clone(),
        }
