import torch
import torch.nn as nn


# ==========================================
# 核心 LSTM 模型定义
# ==========================================
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        """
        初始化下一个词预测的 LSTM 模型
        :param vocab_size: 词表大小 (模型的最终输出维度需要和词表大小一致)
        :param embedding_dim: 词向量的维度 (将离散的单词映射为连续的稠密向量)
        :param hidden_dim: LSTM 隐藏层的维度 (决定了模型的记忆容量)
        :param num_layers: LSTM 的层数 (层数越多，提取复杂特征的能力越强)
        """
        super(NextWordLSTM, self).__init__()

        # 1. 词嵌入层 (Embedding Layer)
        # 作用：将输入的单词索引 (如整数 5) 转换为长度为 embedding_dim 的浮点数向量
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 2. LSTM 层 (Long Short-Term Memory Layer)
        # 作用：处理序列数据，提取上下文的特征
        # batch_first=True 表示输入张量的形状必须是 (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 3. 全连接层 (Fully Connected Layer)
        # 作用：将 LSTM 提取的隐藏特征映射回词表维度，输出每个词的概率打分
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x):
        """
        前向传播函数，定义数据在网络中的流向
        :param x: 输入的文本序列张量，形状为 (batch_size, seq_len)
        """
        # 形状变化: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # 将词向量输入给 LSTM
        # lstm_out 形状: (batch_size, seq_len, hidden_dim)
        # 我们不需要保留中间状态的 hidden 变量，所以用 _ 忽略
        lstm_out, _ = self.lstm(embeds)

        # 关键步骤：因为我们是预测“下一个词”，所以只需要利用 LSTM 在序列最后一个时间步的输出特征
        # 取出每个样本在序列最后一个位置 (索引为 -1) 的隐藏状态
        # 形状变化: (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        last_out = lstm_out[:, -1, :]

        # 传入全连接层进行打分预测
        # 形状变化: (batch_size, hidden_dim) -> (batch_size, vocab_size)
        out = self.fc(last_out)

        return out


# ==========================================
# 本地测试代码（验证网络结构与张量维度是否匹配）
# ==========================================
if __name__ == "__main__":
    # 模拟 data_utils.py 中的输出参数
    BATCH_SIZE = 4
    SEQ_LEN = 10
    VOCAB_SIZE = 10000  # 假设词表大小为 10000

    # 1. 实例化模型
    print("正在初始化 LSTM 模型...")
    model = NextWordLSTM(vocab_size=VOCAB_SIZE)
    print(model)

    # 2. 生成一个模拟的输入张量 (代表一个 batch 的单词索引序列)
    # torch.randint 会生成 0 到 VOCAB_SIZE-1 之间的随机整数
    dummy_input = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    print(f"\n模拟输入张量的形状: {dummy_input.shape} (Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN})")

    # 3. 将模拟数据输入模型，进行一次前向传播
    output = model(dummy_input)

    # 4. 检查输出形状
    print(f"\n模型输出张量的形状: {output.shape}")
    print(
        f"预期形状应为: ({BATCH_SIZE}, {VOCAB_SIZE}) -> 代表这 {BATCH_SIZE} 个样本里，每个样本下个词属于 {VOCAB_SIZE} 个词中某一个的打分。")

    if output.shape == (BATCH_SIZE, VOCAB_SIZE):
        print("\n✅ 测试通过！LSTM 模型结构搭建完全正确！")
    else:
        print("\n❌ 形状不匹配，请检查代码。")