import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将图书分块为最大长度的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") # 分词器初始化
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # 创建数据集
    # 创建加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

#测试
with open("small-text-sample.txt", "r", encoding="utf-8") as f: # 读取文本文件
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2") # 初始化分词器
encoded_text = tokenizer.encode(raw_text) # 对原始文本进行编码

# 定义词汇表大小、输出维度、最大长度和块大小
vocab_size = 50257
output_dim = 256
max_len = 1024
block_size = max_len

# 创建词嵌入层和位置嵌入层
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(block_size, output_dim)

max_length = 4 # 设置最大长度为 4
# 创建数据加载器
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=5)

# 遍历数据加载器中的每个批次
for batch in dataloader:
    x, y = batch # 从当前批次中解包输入和目标数据

    # 使用词嵌入层计算输入序列的词嵌入
    token_embeddings = token_embedding_layer(x)
    # 使用位置嵌入层计算位置嵌入，这里使用 torch.arange 创建一个与 max_length 相同长度的序列
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    # 将词嵌入和位置嵌入相加，得到最终的输入嵌入
    input_embeddings = token_embeddings + pos_embeddings

    # 跳出循环，这里可能是为了演示目的，实际训练中通常会继续循环直到遍历完所有数据
    break

print(input_embeddings.shape) #输出：torch.Size([8, 4, 256])