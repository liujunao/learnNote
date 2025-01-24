import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#方案A：简单实现 -- 定义因果自注意力模块
class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        super().__init__() # 调用父类构造函数
        self.d_out = d_out # 输出维度
        # 查询、键和值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # Dropout层，用于正则化
        # 注册一个缓冲区，用于存储上三角掩码，用于因果自注意力
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, n_tokens, d_in = x.shape  # 获取输入张量的批次大小、序列长度和输入维度
        # 分别计算查询、键和值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算注意力分数，这里使用了转置操作
        attn_scores = queries @ keys.transpose(1, 2)
        # 使用掩码将未来位置的注意力分数置为负无穷，实现因果自注意力
        attn_scores.masked_fill_(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)
        # 归一化注意力分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
        # 应用dropout
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = attn_weights @ values
        return context_vec

#定义多头注意力包装器模块
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__() # 调用父类构造函数
        # 创建多个因果自注意力模块
        self.heads = nn.ModuleList([
            CausalSelfAttention(d_in, d_out, block_size, dropout, qkv_bias) for _ in range(num_heads)
        ])
        # 输出投影层，用于将多头的输出合并
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        # 将所有头的输出沿着最后一个维度拼接
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        # 通过输出投影层
        return self.out_proj(context_vec)

#测试
torch.manual_seed(123) # 设置随机种子以确保结果的可重复性
block_size = max_length # 定义块大小，这里与最大长度相同
d_in = output_dim # 输入维度

num_heads = 2 # 定义多头注意力中每个头的输出维度
d_out = d_in // num_heads # 输出维度是输入维度除以头数

# 初始化多头注意力包装器
mha = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads)

# 假设 input_embeddings 是之前准备好的输入数据
batch = input_embeddings
# 使用多头注意力模块处理输入数据
context_vecs = mha(batch)

# 打印上下文向量的维度
print("context_vecs.shape:", context_vecs.shape) #context_vecs.shape: torch.Size([8, 4, 256])
#--------------------------

#方式B：替代实现 -- 定义多头自注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__() # 调用父类构造函数
        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        # 初始化模块的属性
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 计算每个头的维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # 键线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值线性层
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        # 注册一个缓冲区，用于存储上三角掩码，用于因果自注意力
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        # 获取输入张量的批次大小、序列长度和输入维度
        b, num_tokens, d_in = x.shape

        # 分别计算查询、键和值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将矩阵按头数分割，并添加一个维度
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置以匹配多头注意力的维度
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力分数，并应用因果掩码
        attn_scores = queries @ keys.transpose(2, 3)  # 对每个头进行点积
        # 将掩码截断到与序列长度相匹配，并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 扩展掩码以匹配维度
        mask_unsqueezed = mask_bool.unsqueeze(0).unsqueeze(0)
        # 使用扩展的掩码填充注意力分数
        attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)
        
        # 归一化注意力分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        # 合并头的输出
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 可选的输出投影
        context_vec = self.out_proj(context_vec)

        return context_vec
    
#测试
torch.manual_seed(123) # 设置随机种子以确保结果的可重复性

block_size = max_length # 定义块大小，这里与最大长度相同
d_in = output_dim # 输入和输出维度
d_out = d_in # 输出维度设置为与输入维度相同

# 初始化多头自注意力模块
mha = MultiHeadAttention(d_in, d_out, block_size, dropout=0.0, num_heads=2)

# 假设 input_embeddings 是之前准备好的输入数据
batch = input_embeddings
# 使用多头自注意力模块处理输入数据
context_vecs = mha(batch)

# 打印上下文向量的维度
print("context_vecs.shape:", context_vecs.shape) #context_vecs.shape: torch.Size([8, 4, 256])