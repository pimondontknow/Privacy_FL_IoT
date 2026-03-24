import flwr as fl


def main():
    print("=== 启动联邦学习中央服务器 ===")

    # 1. 配置联邦学习策略 (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # 每一轮抽取 100% 的可用客户端参与训练
        fraction_evaluate=1.0,  # 每一轮抽取 100% 的可用客户端参与评估
        min_fit_clients=2,  # 至少需要 2 个客户端连接后，才正式开始训练
        min_evaluate_clients=2,  # 至少需要 2 个客户端连接后，才正式开始评估
        min_available_clients=2,  # 系统总共至少需要 2 个在线客户端
    )

    # 2. 启动 Flower 服务器，监听本地 8080 端口
    print("服务器正在 0.0.0.0:8080 监听客户端连接...")
    print("请开启新的终端运行 client.py...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # 我们先测试跑 3 轮联邦通信
        strategy=strategy,
    )


if __name__ == "__main__":
    main()