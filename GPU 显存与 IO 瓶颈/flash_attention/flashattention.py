from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
import torch

"""
flash_attn_func用于处理分离的QKV张量；
    alibi_slopes：
        如果传入一个张量, FlashAttention 会在计算注意力分数时，根据距离自动加上一个线性偏置值: scores_ij = Q_iK_j -m|i - j|
        m 就是传入的斜率。这会让模型天然地更关注附近的词，而“惩罚”距离远的词，从而具备处理比训练时更长文本的能力
        
flash_attn_qkvpacked_func用于处理拼接好的QKV张量
在训练中通常效率更高，因为避免了梯度拼接的开销。
"""
batch_size = 4
seq_len = 2048
n_heads = 16
head_dim = 64

device = torch.device('cuda')
dtype = torch.bfloat16

Q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
K = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
V = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)

output = flash_attn_func(
    Q, K, V,
    dropout_p=0.0, # 训练时可设置为0.1
    softmax_scale=None, # 默认为 1/sqrt(head_dim)
    causal=True,
    window_size=(-1, -1), # 非局部注意力机制
    alibi_slopes=None, 
    deterministic=False # 非确定性模式，训练时用
)

qkv = torch.stack((Q, K, V), dim=2) # [batch, seq_len, 3, n_heads, head_dim]
output = flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    causal=True
)


"""
GQA + ALIBI
"""

n_heads_q = 16
n_heads_kv = 4

Q_GQA = torch.randn(batch_size, seq_len, n_heads_q, head_dim, device=device, dtype=dtype)
K_GQA = torch.randn(batch_size, seq_len, n_heads_kv, head_dim, device=device, dtype=dtype)
V_GQA = torch.randn(batch_size, seq_len, n_heads_kv, head_dim, device=device, dtype=dtype)

alibi_slopes = torch.randn(n_heads_q, device=device, dtype=torch.float32) * 0.1

output = flash_attn_func(
    Q_GQA, K_GQA, V_GQA,
    causal=True,
    alibi_slopes=alibi_slopes
)
output.shape


"""
与现有训练框架集成
"""

# 在Hugging Face Transformers中使用FlashAttention（如果模型支持）
from transformers import AutoModelForCausalLM
import torch
 
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 关键参数
    device_map="auto"
)
 
# 在DeepSpeed中，通过配置文件启用
# ds_config.json中：
# {
#   "fp16": {"enabled": true},
#   "bf16": {"enabled": false},
#   "flashattention": {"enabled": true}
