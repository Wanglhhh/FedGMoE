from openfgl.task.node_cls import NodeClsTask


class BaseClient:
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        self.args = args
        self.client_id = client_id
        self.message_pool = message_pool
        self.device = device
        self.task = NodeClsTask(args, client_id, data, data_dir, device)

    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError


class BaseServer:
    def __init__(self, args, global_data, data_dir, message_pool, device):
        self.args = args
        self.message_pool = message_pool
        self.device = device
        self.task = NodeClsTask(args, None, global_data, data_dir, device)

    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError
