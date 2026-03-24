import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from model import NextWordLSTM
from data_utils import get_dataloaders

# ==========================================
# 1. 基础配置
# ==========================================
# 自动检测并使用你刚配置好的 4070Ti 显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Flower 框架的参数转换辅助函数
# ==========================================
# Flower 框架在网络中传输的是 NumPy 数组，而我们的模型用的是 PyTorch 张量。
# 所以我们需要写两个“翻译官”函数来进行转换。

def get_parameters(model):
    """将 PyTorch 模型参数打包成 NumPy 数组列表，发给服务器"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """接收服务器发来的 NumPy 数组列表，覆盖本地的 PyTorch 模型"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ==========================================
# 3. 核心：定义 Flower 客户端
# ==========================================
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, epochs):
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        """响应服务器的请求，上传本地参数"""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """核心训练逻辑：接收全局参数 -> 本地训练 -> 返回更新后的参数"""
        print("\n--- 收到服务器下发的全局模型，开始本地训练 ---")

        # 1. 用全局模型覆盖本地模型
        set_parameters(self.model, parameters)

        # 2. 在本地数据上训练指定的 Epoch 数
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"本地 Epoch {epoch + 1}/{self.epochs} 完成 | 平均 Loss: {total_loss / len(self.train_loader):.4f}")

        # 3. 返回更新后的模型参数、本地数据样本量（用于服务器加权平均）、以及其他自定义指标字典
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """本地评估逻辑（为了简化第一阶段，我们暂时把评估任务交给服务器）"""
        return float(0.0), len(self.train_loader.dataset), {"accuracy": 0.0}


# ==========================================
# 4. 客户端启动入口
# ==========================================
def main():
    print(f"客户端启动中... 使用的计算设备: {device}")

    # 1. 客户端加载自己的本地数据 (目前为了跑通流程，先全量加载)
    print("正在加载数据管道...")
    train_loader, _, _, vocab = get_dataloaders(batch_size=32, seq_len=35)
    actual_vocab_size = max(vocab.values()) + 1

    # 2. 初始化本地骨干网络
    model = NextWordLSTM(vocab_size=actual_vocab_size).to(device)

    # 3. 实例化并启动 Flower 客户端，尝试连接服务器 (默认端口 8080)
    print("正在连接中央服务器 (127.0.0.1:8080)...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FLClient(model, train_loader, epochs=1).to_client()
    )


if __name__ == "__main__":
    main()