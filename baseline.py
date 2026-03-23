import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_dataloaders
from model import NextWordLSTM


def train_standalone():
    print("=== 启动集中式基线模型训练 ===")

    # 1. 设备配置 (如果有 NVIDIA 显卡会自动使用 GPU 加速，否则使用 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 2. 超参数设置
    BATCH_SIZE = 32
    SEQ_LEN = 35
    VOCAB_SIZE = 10000
    EPOCHS = 2  # 为了快速测试，我们先只跑 2 个轮次
    LEARNING_RATE = 0.001

    # 3. 加载数据
    print("正在加载数据集...")
    train_loader, valid_loader, test_loader, vocab = get_dataloaders(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_vocab_size=VOCAB_SIZE
    )

    # 因为实际语料库提取的词表可能小于设定的 max_vocab_size，我们要以实际的词表大小为准
    # 取字典中最大的索引值加 1，确保 Embedding 层绝对不会越界
    actual_vocab_size = max(vocab.values()) + 1
    print(f"实际词表大小: {actual_vocab_size}")

    # 4. 初始化模型、损失函数和优化器
    model = NextWordLSTM(vocab_size=actual_vocab_size).to(device)
    # 使用交叉熵损失函数，这是分类任务（预测词表中哪一个词）的标准配置
    criterion = nn.CrossEntropyLoss()
    # 使用 Adam 优化器自动调整学习率
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 核心训练循环
    for epoch in range(EPOCHS):
        model.train()  # 开启训练模式
        total_loss = 0.0

        print(f"\n--- 开始第 {epoch + 1}/{EPOCHS} 轮训练 ---")

        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据推送到对应的设备上 (CPU 或 GPU)
            data, target = data.to(device), target.to(device)

            # 第一步：清空上一步的残余梯度
            optimizer.zero_grad()

            # 第二步：前向传播，得到预测打分
            output = model(data)

            # 第三步：计算损失 (预测值与真实标签的差距)
            loss = criterion(output, target)

            # 第四步：反向传播，计算每个参数的梯度
            loss.backward()

            # 第五步：根据梯度更新模型参数
            optimizer.step()

            total_loss += loss.item()

            # 每隔 100 个 batch 打印一次进度，让你看到模型正在学习
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = total_loss / 100
                print(f"Epoch: {epoch + 1} | Batch: {batch_idx}/{len(train_loader)} | 平均 Loss: {avg_loss:.4f}")
                total_loss = 0.0  # 重置累加器

        # ==========================================
        # 验证环节 (每个 Epoch 结束后，在未见过的数据上测试性能)
        # ==========================================
        model.eval()  # 开启评估模式 (关闭 Dropout 等)
        val_loss = 0.0
        correct = 0
        total_samples = 0

        # 验证时不计算梯度，节省显存并加速
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # 累加验证集 Loss
                val_loss += criterion(output, target).item()

                # 计算准确率 (选出打分最高的那个词的索引作为预测结果)
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == target).sum().item()
                total_samples += target.size(0)

        avg_val_loss = val_loss / len(valid_loader)
        val_accuracy = correct / total_samples * 100

        print(f"\n✅ 第 {epoch + 1} 轮结束 | 验证集平均 Loss: {avg_val_loss:.4f} | 准确率: {val_accuracy:.2f}%")


if __name__ == "__main__":
    train_standalone()