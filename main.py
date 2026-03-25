import flwr as fl
import torch
from data_utils import get_federated_dataloaders
from model import NextWordLSTM
from client import FLClient
from flwr.common import Context


def main():
    print("=== 启动最新版 Flower 联邦仿真实验 (Simulation API) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_CLIENTS = 2  # 这里可以随时改成 5 个或 10 个进行复杂实验

    # 1. 获取 Non-IID 划分后的异构数据集
    client_loaders, valid_loader, vocab = get_federated_dataloaders(num_clients=NUM_CLIENTS)
    actual_vocab_size = max(vocab.values()) + 1

    # 2. 客户端工厂函数：使用最新版的 Context 传参
    def client_fn(context: Context) -> fl.client.Client:
        # 从上下文中提取当前虚拟客户端的 ID
        client_id = int(context.node_config["partition-id"])

        # 客户端只获取属于自己的那份“偏好数据”
        train_loader = client_loaders[client_id]
        model = NextWordLSTM(vocab_size=actual_vocab_size).to(device)
        return FLClient(model, train_loader, epochs=1).to_client()

    # 3. 配置联邦学习策略
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # 4. 启动一键仿真！
    # client_resources 告诉框架每个节点可以调用多少硬件资源
    # num_gpus: 1.0 表示框架会自动排队调度，让每个虚拟客户端独占你的 4070Ti 跑完一轮再换下一个，安全且高效
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0},
    )


if __name__ == "__main__":
    main()