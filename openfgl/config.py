import argparse


SUPPORTED_DATASETS = (
    "Cora",
    "CiteSeer",
    "PubMed",
    "Actor",
    "Minesweeper",
    "Roman-empire",
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Run one FedGMoE main experiment.")

# Runtime and the single experiment seed.
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--seed", type=int, default=3412)

# Dataset and federated partition.
parser.add_argument("--root", type=str, default="dataset")
parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS, default="Cora")
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_rounds", type=int, default=100)
parser.add_argument("--louvain_resolution", type=float, default=1.0)
parser.add_argument("--louvain_delta", type=float, default=20.0)

# Local training.
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--hid_dim", type=int, default=64)
parser.add_argument("--smooth_hops", type=int, default=2)
parser.add_argument("--bwgnn_order", type=int, default=2)
parser.add_argument("--lambda_aux", type=float, default=0.1)

# Fixed-main-method calibration hyperparameters.
parser.add_argument("--gate_temperature", type=float, default=1.0)
parser.add_argument("--expert_reliability_floor", type=float, default=0.02)
parser.add_argument("--calibration_warmup_rounds", type=int, default=5)
parser.add_argument("--calibration_base_temperature", type=float, default=1.0)
parser.add_argument("--client_calibration_shrinkage", type=float, default=50.0)
parser.add_argument("--calibration_mirror_steps", type=int, default=40)
parser.add_argument("--calibration_mirror_lr", type=float, default=0.5)
parser.add_argument("--calibration_mirror_prior", type=float, default=0.02)
parser.add_argument("--calibration_evidence_floor", type=float, default=0.05)
parser.add_argument("--calibration_precision_ridge", type=float, default=1e-4)
parser.add_argument("--calibration_consensus_steps", type=int, default=40)
parser.add_argument("--calibration_consensus_lr", type=float, default=0.5)
parser.add_argument("--calibration_consensus_prior", type=float, default=0.02)

# Output only; these options do not change the experiment method.
parser.add_argument("--debug", type=str2bool, default=True)
parser.add_argument("--log_root", type=str, default=None)
parser.add_argument("--log_name", type=str, default=None)
parser.add_argument("--plot_curve", type=str2bool, default=True)


args = parser.parse_args()
