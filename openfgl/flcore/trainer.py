import torch

from openfgl.data.distributed_dataset_loader import FGLDataset
from openfgl.flcore.fedgmoe.client import FedGMoEClient
from openfgl.flcore.fedgmoe.server import FedGMoEServer
from openfgl.utils.logger import Logger


class FGLTrainer:
    """Run one all-client FedGMoE experiment and report local accuracy."""

    def __init__(self, args):
        self.args = args
        self.message_pool = {}
        dataset = FGLDataset(args)
        self.device = torch.device(
            f"cuda:{args.gpuid}"
            if args.use_cuda and torch.cuda.is_available()
            else "cpu"
        )
        self.clients = [
            FedGMoEClient(
                args,
                client_id,
                dataset.local_data[client_id],
                dataset.processed_dir,
                self.message_pool,
                self.device,
            )
            for client_id in range(args.num_clients)
        ]
        self.server = FedGMoEServer(
            args,
            dataset.global_data,
            dataset.processed_dir,
            self.message_pool,
            self.device,
        )
        self.best = {
            "round": 0,
            "val_accuracy": 0.0,
            "test_accuracy": 0.0,
        }
        self.logger = Logger(args)

    def train(self):
        all_clients = list(range(self.args.num_clients))
        for round_id in range(self.args.num_rounds):
            print(f"round # {round_id}\t\tclients: {all_clients}")
            self.message_pool["round"] = round_id

            self.server.send_message()
            for client in self.clients:
                client.execute()
                client.send_message()
            self.server.execute()
            result = self.evaluate()
            self.logger.add_log(result)
            print("-" * 50)

        self.logger.save()

    def evaluate(self):
        totals = {
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "test_accuracy": 0.0,
            "loss_train": 0.0,
            "loss_val": 0.0,
            "loss_test": 0.0,
        }
        total_samples = sum(client.task.num_samples for client in self.clients)
        for client in self.clients:
            local = client.task.evaluate()
            weight = client.task.num_samples / total_samples
            for key in totals:
                totals[key] += local[key] * weight

        result = {"round": self.message_pool["round"], **totals}
        if result["val_accuracy"] > self.best["val_accuracy"]:
            self.best = {
                "round": result["round"],
                "val_accuracy": result["val_accuracy"],
                "test_accuracy": result["test_accuracy"],
            }

        print(
            f"round: {result['round']}\t"
            f"val_accuracy: {result['val_accuracy']:.4f}\t"
            f"test_accuracy: {result['test_accuracy']:.4f}"
        )
        print(
            f"best_round: {self.best['round']}\t"
            f"best_val_accuracy: {self.best['val_accuracy']:.4f}\t"
            f"best_test_accuracy: {self.best['test_accuracy']:.4f}"
        )
        return result
