from torch.optim import Adam


class BaseTask:
    """Minimal task state shared by FedGMoE clients and server."""

    def __init__(self, args, client_id, data, data_dir, device):
        self.args = args
        self.client_id = client_id
        self.data_dir = data_dir
        self.device = device
        self.data = data.to(device)
        self.model = self.default_model.to(device)
        self.optim = Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if client_id is not None:
            self.load_train_val_test_split()
