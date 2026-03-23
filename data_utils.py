import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import requests


# ==========================================
# 1. 数据下载与解压模块
# ==========================================
def download_wikitext2(root_dir='./data'):
    """自动下载并解压 WikiText-2 数据集"""
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, "wikitext-2.zip")

    # 如果压缩包不存在，则使用 requests 下载
    if not os.path.exists(zip_path):
        print("正在下载 WikiText-2 数据集，请稍候 (这可能需要几分钟)...")
        # requests.get 会自动处理 301 重定向
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查是否下载成功

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("下载完成，正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        print("解压完毕！")

    # 返回训练集、验证集和测试集的文件路径
    train_path = os.path.join(root_dir, "wikitext-2", "wiki.train.tokens")
    valid_path = os.path.join(root_dir, "wikitext-2", "wiki.valid.tokens")
    test_path = os.path.join(root_dir, "wikitext-2", "wiki.test.tokens")

    return train_path, valid_path, test_path


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