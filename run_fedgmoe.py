import sys

from openfgl.config import args
from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.basic_utils import seed_everything


DATASET_TRAINING_PROFILES = {
    "Cora": {
        "num_rounds": 60,
        "lr": 0.004,
        "weight_decay": 0.0008,
        "calibration_base_temperature": 0.7,
        "client_calibration_shrinkage": 100.0,
    },
    "CiteSeer": {
        "num_rounds": 60,
        "lr": 0.0025,
        "calibration_base_temperature": 0.4,
    },
    "PubMed": {"num_rounds": 150, "lr": 0.005},
    "Actor": {"num_rounds": 60, "lr": 0.005},
    "Minesweeper": {"num_rounds": 120},
    "Roman-empire": {"num_rounds": 150, "lr": 0.005},
}


def _arg_supplied(name):
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def _apply_dataset_profile():
    applied = []
    for name, value in DATASET_TRAINING_PROFILES[args.dataset].items():
        if not _arg_supplied(f"--{name}"):
            setattr(args, name, value)
            applied.append(name)
    if applied:
        print(f"Dataset profile: {args.dataset} (applied: {','.join(applied)})")


def main():
    _apply_dataset_profile()
    seed_everything(args.seed)
    FGLTrainer(args).train()


if __name__ == "__main__":
    main()
