import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import requests
from torch.utils.data import Subset

# ==========================================
# 1. 数据下载与解压模块
# ==========================================
def download_wikitext2(root_dir='./data'):
    """直接从 GitHub 下载纯文本文件，彻底避开压缩包损坏和反爬虫问题"""
    target_dir = os.path.join(root_dir, "wikitext-2")
    os.makedirs(target_dir, exist_ok=True)

    # 直接使用 PyTorch 官方 GitHub 仓库中托管的纯文本文件
    urls = {
        "wiki.train.tokens": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
        "wiki.valid.tokens": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
        "wiki.test.tokens": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    }

    paths = []
    for filename, url in urls.items():
        file_path = os.path.join(target_dir, filename)
        paths.append(file_path)

        # 如果文件不存在，则直接拉取文本写入
        if not os.path.exists(file_path):
            print(f"正在从 PyTorch 官方 GitHub 下载 {filename} ...")
            # 添加基础的 User-Agent 即可
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"{filename} 下载成功！")

    print("所有数据文件准备完毕！")
    return paths[0], paths[1], paths[2]


# ==========================================
# 2. 词表（Vocabulary）构建模块
# ==========================================
def build_vocab(file_path, max_vocab_size=10000):
    """统计词频，建立单词到数字索引的映射字典"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 简单的按空格分词
    tokens = text.split()
    counter = Counter(tokens)

    # 选取最常见的前 max_vocab_size - 1 个词，保留 0 给未知词 <unk>
    common_words = counter.most_common(max_vocab_size - 1)

    vocab = {'<unk>': 0}
    for idx, (word, _) in enumerate(common_words, start=1):
        vocab[word] = idx

    return vocab


# ==========================================
# 3. PyTorch 自定义 Dataset
# ==========================================
class NextWordDataset(Dataset):
    """为下一个词预测任务定制的 Dataset"""

    def __init__(self, file_path, vocab, seq_len=35):
        self.seq_len = seq_len
        self.vocab = vocab

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = text.split()

        # 将文本单词转化为对应的数字索引，遇到生词统一标记为 <unk> 的索引 0
        self.data = [self.vocab.get(word, self.vocab['<unk>']) for word in tokens]

    def __len__(self):
        # 减去序列长度，确保我们总是有“下一个词”可以作为标签
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 提取长度为 seq_len 的输入序列 X
        x = self.data[idx: idx + self.seq_len]
        # 紧接着的那个词作为预测目标 Y
        y = self.data[idx + self.seq_len]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ==========================================
# 4. 暴露给外部调用的主接口
# ==========================================
def get_dataloaders(batch_size=32, seq_len=35, max_vocab_size=10000):
    """集成方法：一步获取所有 DataLoader 和词表"""
    train_path, valid_path, test_path = download_wikitext2()

    print("正在构建词表 (Vocabulary)...")
    vocab = build_vocab(train_path, max_vocab_size)

    print("正在构建 Dataset...")
    train_dataset = NextWordDataset(train_path, vocab, seq_len)
    valid_dataset = NextWordDataset(valid_path, vocab, seq_len)
    test_dataset = NextWordDataset(test_path, vocab, seq_len)

    print("正在构建 DataLoader...")
    # drop_last=True 确保每个 batch 的大小完全一致，避免 LSTM 在处理最后一个残缺 batch 时报错
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print("数据准备就绪！")
    return train_loader, valid_loader, test_loader, vocab


# ==========================================
# 5. 本地测试代码（仅在直接运行此脚本时触发）
# ==========================================
if __name__ == "__main__":
    # 测试一下数据流是否正常
    train_loader, _, _, vocab = get_dataloaders(batch_size=4, seq_len=10)
    print(f"\n词表大小: {len(vocab)}")

    # 抓取第一个 batch 看看
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\n第一个 Batch 的输入 X 形状: {data.shape}")  # 应该是 [4, 10]
        print(f"第一个 Batch 的标签 Y 形状: {target.shape}")  # 应该是 [4]
        print(f"\nX 的具体数据 (数字索引):\n{data}")
        print(f"Y 的具体数据 (数字索引):\n{target}")
        break  # 只看第一个 batch 就退出


def get_federated_dataloaders(num_clients=2, batch_size=32, seq_len=35, max_vocab_size=10000):
    """专为联邦学习设计的 Non-IID 数据划分引擎"""
    train_path, valid_path, test_path = download_wikitext2()

    print("正在构建联邦词表...")
    vocab = build_vocab(train_path, max_vocab_size)

    print("正在加载并切分训练数据...")
    full_train_dataset = NextWordDataset(train_path, vocab, seq_len)

    # 【Non-IID 核心逻辑】：顺序切块 (Block Partitioning)
    dataset_size = len(full_train_dataset)
    chunk_size = dataset_size // num_clients

    client_loaders = []
    for i in range(num_clients):
        # 给每个客户端分配一段连续且完全不重合的文本块，模拟不同主题偏好的边缘设备
        indices = list(range(i * chunk_size, (i + 1) * chunk_size))
        subset = Subset(full_train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
        client_loaders.append(loader)

    print("正在准备全局验证集...")
    valid_dataset = NextWordDataset(valid_path, vocab, seq_len)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"成功将数据划分为 {num_clients} 份异构客户端数据！")
    return client_loaders, valid_loader, vocab