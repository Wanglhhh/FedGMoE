import csv
import datetime
import os
import pickle
import time


def _safe(value):
    text = str(value)
    for token in '<>:"/\\|?*':
        text = text.replace(token, "-")
    return text.replace(" ", "")


class Logger:
    """Save only the round accuracy/loss results of the main run."""

    def __init__(self, args):
        self.args = args
        self.enabled = args.debug
        self.rows = []
        self.start_time = time.time()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = "_".join(
            [
                "fedgmoe",
                _safe(args.dataset),
                f"s{args.seed}",
                f"lr{args.lr}",
                f"r{args.num_rounds}",
                timestamp,
            ]
        )
        if args.log_name:
            stem += f"_{_safe(args.log_name)}"
        root = args.log_root or "logs"
        self.output_dir = os.path.join(root, args.dataset)
        self.csv_path = os.path.join(self.output_dir, stem + ".csv")
        self.pkl_path = os.path.join(self.output_dir, stem + ".pkl")
        self.summary_path = os.path.join(self.output_dir, stem + ".txt")
        self.figure_path = os.path.join(self.output_dir, stem + ".png")

    def add_log(self, result):
        if not self.enabled:
            return
        self.rows.append(dict(result))
        self._save_csv()

    def _save_csv(self):
        if not self.rows:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def _save_curve(self):
        if not self.rows or not self.args.plot_curve:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        figure, axis = plt.subplots(figsize=(7, 4))
        rounds = [row["round"] for row in self.rows]
        axis.plot(rounds, [row["val_accuracy"] for row in self.rows], label="validation")
        axis.plot(rounds, [row["test_accuracy"] for row in self.rows], label="test")
        axis.set_xlabel("Round")
        axis.set_ylabel("Accuracy")
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.figure_path, dpi=160)
        plt.close(figure)

    def save(self):
        if not self.enabled or not self.rows:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        best = max(self.rows, key=lambda row: row["val_accuracy"])
        with open(self.summary_path, "w", encoding="utf-8") as file:
            file.write(f"dataset: {self.args.dataset}\n")
            file.write(f"seed: {self.args.seed}\n")
            file.write(f"best_round: {best['round']}\n")
            file.write(f"best_val_accuracy: {best['val_accuracy']:.6f}\n")
            file.write(f"best_test_accuracy: {best['test_accuracy']:.6f}\n")
        with open(self.pkl_path, "wb") as file:
            pickle.dump(
                {
                    "args": vars(self.args),
                    "elapsed_seconds": time.time() - self.start_time,
                    "rounds": self.rows,
                },
                file,
            )
        self._save_csv()
        self._save_curve()
