import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
import math


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.5):
        super(MLPBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def mean_aggregate(x, edge_index):
    row, col = edge_index
    out = torch.zeros_like(x)
    deg = x.new_zeros((x.size(0), 1))
    out.index_add_(0, row, x[col])
    deg.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device, dtype=x.dtype))
    out = out + x
    deg = deg + 1.0
    return out / deg.clamp_min(1.0)


def sym_norm_aggregate(x, edge_index):
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    row, col = edge_index
    deg = x.new_zeros((x.size(0), 1))
    deg.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device, dtype=x.dtype))
    deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    msg = x[col] * deg_inv_sqrt[col]
    out = torch.zeros_like(x)
    out.index_add_(0, row, msg)
    return out * deg_inv_sqrt


def normalized_laplacian_apply(x, edge_index):
    return x - sym_norm_aggregate(x, edge_index)


def calculate_bwgnn_theta(order):
    order = max(1, int(order))
    scale = order + 1
    thetas = []
    for band in range(order + 1):
        band_scale = scale * math.comb(order, band)
        coeffs = []
        for power in range(order + 1):
            if power < band:
                coeffs.append(0.0)
                continue
            remain = power - band
            if remain > order - band:
                coeffs.append(0.0)
                continue
            coeff = band_scale * math.comb(order - band, remain)
            coeff *= (0.5 ** band) * ((-0.5) ** remain)
            coeffs.append(float(coeff))
        thetas.append(coeffs)
    return thetas


class BWGNNPolyConv(nn.Module):
    def __init__(self, theta):
        super(BWGNNPolyConv, self).__init__()
        self.register_buffer("theta", torch.tensor(theta, dtype=torch.float32), persistent=False)

    def forward(self, x, edge_index):
        theta = self.theta.to(device=x.device, dtype=x.dtype)
        out = theta[0] * x
        power_x = x
        for power in range(1, theta.numel()):
            power_x = normalized_laplacian_apply(power_x, edge_index)
            out = out + theta[power] * power_x
        return out


def select_bwgnn_thetas(order, band):
    thetas = calculate_bwgnn_theta(order)
    split = max(1, len(thetas) // 2)
    if band == "low":
        selected = thetas[:split]
    elif band == "high":
        selected = thetas[split:]
    else:
        raise ValueError(f"Unsupported BWGNN band: {band}")
    return selected or thetas


class ContentExpert(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.5):
        super(ContentExpert, self).__init__()
        self.encoder = MLPBlock(input_dim, hid_dim, dropout)

    def forward(self, data):
        return self.encoder(data.x)


class SmoothExpert(nn.Module):
    def __init__(self, input_dim, hid_dim, hops=2, dropout=0.5, decay=0.7, backbone="gcn", bwgnn_order=2):
        super(SmoothExpert, self).__init__()
        self.backbone = backbone
        if self.backbone not in {"mean", "gcn", "bwgnn_low"}:
            raise ValueError(f"Unsupported smooth_backbone: {self.backbone}")
        self.encoder = MLPBlock(input_dim, hid_dim, dropout)
        self.hops = max(1, int(hops))
        self.decay = float(decay)
        self.dropout = dropout
        self.out = nn.Linear(hid_dim, hid_dim)
        self.convs = nn.ModuleList()
        if self.backbone == "gcn":
            self.convs.append(GCNConv(input_dim, hid_dim))
            for _ in range(max(1, self.hops) - 1):
                self.convs.append(GCNConv(hid_dim, hid_dim))
        elif self.backbone == "bwgnn_low":
            thetas = select_bwgnn_thetas(bwgnn_order, "low")
            self.convs = nn.ModuleList([BWGNNPolyConv(theta) for theta in thetas])
            self.out = nn.Linear(hid_dim * len(self.convs), hid_dim)

    def forward(self, data):
        if self.backbone == "gcn":
            h = data.x
            for conv in self.convs:
                h = conv(h, data.edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            return h

        if self.backbone == "bwgnn_low":
            h = self.encoder(data.x)
            bands = [conv(h, data.edge_index) for conv in self.convs]
            h = torch.cat(bands, dim=-1)
            h = F.dropout(h, p=self.dropout, training=self.training)
            return F.relu(self.out(h))

        h0 = self.encoder(data.x)
        h = h0
        mixed = h0
        coef_sum = 1.0
        coef = 1.0
        for _ in range(self.hops):
            h = mean_aggregate(h, data.edge_index)
            coef *= self.decay
            mixed = mixed + coef * h
            coef_sum += coef
        mixed = mixed / max(coef_sum, 1e-12)
        mixed = F.dropout(mixed, p=self.dropout, training=self.training)
        return F.relu(self.out(mixed))


class BWGNNHeteroExpert(nn.Module):
    def __init__(self, input_dim, hid_dim, order=2, dropout=0.5):
        super(BWGNNHeteroExpert, self).__init__()
        self.encoder = MLPBlock(input_dim, hid_dim, dropout)
        self.dropout = dropout
        thetas = select_bwgnn_thetas(order, "high")
        self.convs = nn.ModuleList([BWGNNPolyConv(theta) for theta in thetas])
        self.out = nn.Linear(hid_dim * len(self.convs), hid_dim)

    def forward(self, data):
        h = self.encoder(data.x)
        bands = [conv(h, data.edge_index) for conv in self.convs]
        h = torch.cat(bands, dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.relu(self.out(h))


class FedGMoEModel(nn.Module):
    """Four-expert spatial-spectral MoE with calibrated routing."""

    EXPERT_NAMES = ("content", "spatial", "low", "high")

    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        dropout=0.5,
        smooth_hops=2,
        bwgnn_order=2,
        gate_temperature=1.0,
        expert_reliability_floor=0.02,
        calibration_base_temperature=1.0,
        client_calibration_shrinkage=50.0,
        calibration_mirror_steps=40,
        calibration_mirror_lr=0.5,
        calibration_mirror_prior=0.02,
        calibration_evidence_floor=0.05,
        calibration_precision_ridge=1e-4,
        calibration_consensus_steps=40,
        calibration_consensus_lr=0.5,
        calibration_consensus_prior=0.02,
    ):
        super(FedGMoEModel, self).__init__()
        self.num_experts = len(self.EXPERT_NAMES)
        self.gate_temperature = float(gate_temperature)
        self.expert_reliability_floor = float(expert_reliability_floor)
        self.calibration_base_temperature = float(calibration_base_temperature)
        self.client_calibration_shrinkage = max(
            0.0, float(client_calibration_shrinkage)
        )
        self.calibration_mirror_steps = max(1, int(calibration_mirror_steps))
        self.calibration_mirror_lr = max(0.0, float(calibration_mirror_lr))
        self.calibration_mirror_prior = max(0.0, float(calibration_mirror_prior))
        self.calibration_evidence_floor = max(1e-8, float(calibration_evidence_floor))
        self.calibration_precision_ridge = max(0.0, float(calibration_precision_ridge))
        self.calibration_consensus_steps = max(1, int(calibration_consensus_steps))
        self.calibration_consensus_lr = max(0.0, float(calibration_consensus_lr))
        self.calibration_consensus_prior = max(0.0, float(calibration_consensus_prior))

        hops = max(1, int(smooth_hops))
        self.content_expert = ContentExpert(input_dim, hid_dim, dropout)
        self.spatial_expert = SmoothExpert(
            input_dim,
            hid_dim,
            hops,
            dropout,
            backbone="gcn",
            bwgnn_order=bwgnn_order,
        )
        self.low_expert = SmoothExpert(
            input_dim,
            hid_dim,
            hops,
            dropout,
            backbone="bwgnn_low",
            bwgnn_order=bwgnn_order,
        )
        self.high_expert = BWGNNHeteroExpert(
            input_dim,
            hid_dim,
            bwgnn_order,
            dropout,
        )

        self.content_classifier = nn.Linear(hid_dim, output_dim)
        self.spatial_classifier = nn.Linear(hid_dim, output_dim)
        self.low_classifier = nn.Linear(hid_dim, output_dim)
        self.high_classifier = nn.Linear(hid_dim, output_dim)

        uniform = torch.full(
            (self.num_experts,),
            1.0 / self.num_experts,
            dtype=torch.float32,
        )
        for name in [
            "calibrated_global_gate",
            "local_calibrated_gate",
        ]:
            self.register_buffer(name, uniform.clone(), persistent=False)
        self.register_buffer(
            "local_calibration_samples",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "local_calibration_information",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "calibrated_global_information",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "local_calibration_precision",
            torch.zeros(self.num_experts, self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "calibrated_global_precision",
            torch.eye(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.last_output = {}

    def _normalize_active(self, gate):
        gate = gate.clamp_min(0.0)
        total = gate.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(gate, 1.0 / self.num_experts)
        return torch.where(total > 0, gate / total.clamp_min(1e-12), uniform)

    def set_calibrated_global_gate(self, gate):
        gate = torch.as_tensor(
            gate,
            device=self.calibrated_global_gate.device,
            dtype=self.calibrated_global_gate.dtype,
        )
        if gate.numel() != self.num_experts:
            raise ValueError("fedgmoe calibrated global gate requires four weights")
        self.calibrated_global_gate.copy_(
            self._normalize_active(gate.reshape(self.num_experts))
        )

    def set_local_calibrated_gate(
        self,
        gate,
        samples,
        information=None,
        precision=None,
    ):
        gate = torch.as_tensor(
            gate,
            device=self.local_calibrated_gate.device,
            dtype=self.local_calibrated_gate.dtype,
        )
        if gate.numel() != self.num_experts:
            raise ValueError("fedgmoe local calibrated gate requires four weights")
        self.local_calibrated_gate.copy_(
            self._normalize_active(gate.reshape(self.num_experts))
        )
        self.local_calibration_samples.fill_(max(0.0, float(samples)))
        if information is not None:
            self.local_calibration_information.fill_(max(0.0, float(information)))
        if precision is not None:
            precision = torch.as_tensor(
                precision,
                device=self.local_calibration_precision.device,
                dtype=self.local_calibration_precision.dtype,
            )
            if precision.shape != self.local_calibration_precision.shape:
                raise ValueError("local calibration precision has an invalid shape")
            precision = 0.5 * (precision + precision.transpose(0, 1))
            self.local_calibration_precision.copy_(precision)

    def set_calibrated_global_information(self, information):
        self.calibrated_global_information.fill_(max(0.0, float(information)))

    def set_calibrated_global_precision(self, precision):
        precision = torch.as_tensor(
            precision,
            device=self.calibrated_global_precision.device,
            dtype=self.calibrated_global_precision.dtype,
        )
        if precision.shape != self.calibrated_global_precision.shape:
            raise ValueError("global calibration precision has an invalid shape")
        precision = 0.5 * (precision + precision.transpose(0, 1))
        self.calibrated_global_precision.copy_(precision)

    def _hierarchical_calibration_base(self, reference):
        global_base = self._normalize_active(
            self.calibrated_global_gate.to(reference)
        )
        local_base = self._normalize_active(
            self.local_calibrated_gate.to(reference)
        )
        fallback_samples = self.local_calibration_samples.to(reference).clamp_min(0.0)
        local_information = self.local_calibration_information.to(reference).clamp_min(0.0)
        global_information = self.calibrated_global_information.to(reference).clamp_min(
            self.calibration_evidence_floor
        )
        relative_information = (local_information / global_information).clamp(0.25, 4.0)
        information_evidence = fallback_samples * relative_information
        local_evidence = torch.where(
            local_information > 0,
            information_evidence,
            fallback_samples,
        )
        local_weight = local_evidence / (
            local_evidence + self.client_calibration_shrinkage + 1e-12
        )
        local_precision = self.local_calibration_precision.to(reference)
        global_precision = self.calibrated_global_precision.to(reference)
        local_trace = torch.diagonal(local_precision).sum().clamp_min(0.0)
        global_trace = torch.diagonal(global_precision).sum().clamp_min(0.0)
        if local_trace > 0 and global_trace > 0:
            precision_scale = (local_trace / global_trace).clamp(0.25, 4.0)
            local_evidence = fallback_samples * precision_scale
            local_weight = local_evidence / (
                local_evidence + self.client_calibration_shrinkage + 1e-12
            )
        base = local_weight * local_base + (1.0 - local_weight) * global_base
        return self._normalize_active(base), local_weight

    def _gate_weight(self, h_content):
        num_nodes = h_content.size(0)
        base, _ = self._hierarchical_calibration_base(h_content)
        base_logits = base.clamp_min(1e-12).log()
        base_logits = base_logits / max(self.calibration_base_temperature, 1e-6)
        gate = F.softmax(
            base_logits / max(self.gate_temperature, 1e-6),
            dim=-1,
        )
        gate = self._normalize_active(gate)
        return gate.view(1, -1).expand(num_nodes, -1)

    def forward(self, data):
        expert_embeddings = [
            self.content_expert(data),
            self.spatial_expert(data),
            self.low_expert(data),
            self.high_expert(data),
        ]
        expert_logits = [
            self.content_classifier(expert_embeddings[0]),
            self.spatial_classifier(expert_embeddings[1]),
            self.low_classifier(expert_embeddings[2]),
            self.high_classifier(expert_embeddings[3]),
        ]
        gate = self._gate_weight(expert_embeddings[0])
        logits = (
            gate.unsqueeze(-1) * torch.stack(expert_logits, dim=1)
        ).sum(dim=1)
        embedding = (
            gate.unsqueeze(-1) * torch.stack(expert_embeddings, dim=1)
        ).sum(dim=1)
        self.last_output = {}
        for name, expert_logit in zip(
            self.EXPERT_NAMES,
            expert_logits,
        ):
            self.last_output[f"logits_{name}"] = expert_logit
        return embedding, logits

    def _expert_logits(self, mask=None, detach=False):
        logits = [
            self.last_output[f"logits_{name}"]
            for name in self.EXPERT_NAMES
        ]
        if mask is not None:
            logits = [value[mask] for value in logits]
        if detach:
            logits = [value.detach() for value in logits]
        return logits

    def _bilevel_route_objective(self, logits, target, route, prior=None):
        mixed_logits = torch.einsum("e,nec->nc", route, logits)
        objective = F.cross_entropy(mixed_logits, target)
        if prior is not None and self.calibration_mirror_prior > 0:
            objective = objective + self.calibration_mirror_prior * (
                route.clamp_min(1e-12)
                * (
                    route.clamp_min(1e-12).log()
                    - prior.clamp_min(1e-12).log()
                )
            ).sum()
        return objective

    def _estimate_bilevel_mirror_route(self, labels, mask):
        """Optimize the held-out route continuously by entropic mirror descent."""
        if mask is None or mask.sum().item() == 0:
            return None

        logits = torch.stack(self._expert_logits(mask, detach=True), dim=1)
        target = labels[mask]
        active = torch.arange(self.num_experts, device=logits.device)
        active_count = int(active.numel())
        if active_count == 0:
            return None

        with torch.no_grad():
            prior, _ = self._hierarchical_calibration_base(logits)
            prior = self._normalize_active(prior).detach()
            route = prior.index_select(0, active).clamp_min(1e-12)
            route = route / route.sum().clamp_min(1e-12)

        selector = F.one_hot(active, num_classes=self.num_experts).to(
            dtype=logits.dtype,
            device=logits.device,
        )
        def expand_route(active_route):
            return torch.matmul(active_route, selector)

        for _ in range(self.calibration_mirror_steps):
            route = route.detach().requires_grad_(True)
            full_route = expand_route(route)
            objective = self._bilevel_route_objective(
                logits,
                target,
                full_route,
                prior,
            )
            gradient = torch.autograd.grad(objective, route)[0]
            with torch.no_grad():
                route = F.softmax(
                    route.clamp_min(1e-12).log()
                    - self.calibration_mirror_lr * gradient,
                    dim=0,
                )

        route = route.detach()
        full_route = expand_route(route)
        # The validation-risk Hessian encodes which expert trade-offs are locally
        # identifiable, not just one scalar confidence value.
        def validation_risk(active_route):
            mixed_logits = torch.einsum(
                "e,nec->nc",
                expand_route(active_route),
                logits,
            )
            return F.cross_entropy(mixed_logits, target)

        hessian = torch.autograd.functional.hessian(validation_risk, route)
        with torch.no_grad():
            tangent = torch.eye(active_count, device=logits.device, dtype=logits.dtype)
            tangent = tangent - 1.0 / active_count
            information = torch.trace(tangent @ hessian @ tangent)
            information = (information / max(active_count - 1, 1)).clamp_min(0.0)
            samples = float(target.numel())
            precision = selector.transpose(0, 1) @ hessian @ selector
            active_mask = torch.ones_like(full_route)
            full_tangent = torch.diag(active_mask) - torch.outer(
                active_mask,
                active_mask,
            ) / max(active_count, 1)
            precision = full_tangent @ precision @ full_tangent
            precision = samples * 0.5 * (precision + precision.transpose(0, 1))
            precision = precision + self.calibration_precision_ridge * full_tangent
            floor = min(
                max(self.expert_reliability_floor, 0.0),
                1.0 / max(active_count, 1) - 1e-6,
            )
            full_route = full_route * (1.0 - active_count * floor)
            full_route = full_route + floor * active_mask
            full_route = self._normalize_active(full_route)
        return full_route, information, precision

    def estimate_calibrated_route(self, labels, mask):
        estimate = self._estimate_bilevel_mirror_route(labels, mask)
        if estimate is None:
            return None, 0.0, None
        route, information, precision = estimate
        return (
            route,
            float(information.item()),
            precision,
        )

    def moe_regularization_loss(
        self,
        labels,
        mask,
        lambda_aux=0.0,
    ):
        if not self.last_output:
            return labels.new_tensor(0.0, dtype=torch.float32)
        expert_losses = torch.stack(
            [
                F.cross_entropy(logits[mask], labels[mask])
                for logits in self._expert_logits()
            ]
        )
        return float(lambda_aux) * expert_losses.mean()
