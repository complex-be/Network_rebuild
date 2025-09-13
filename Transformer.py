from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============
# 掩码工具函数
# ===============

def make_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)  
    return mask  # bool


def make_subsequent_mask(size: int) -> torch.Tensor:
    """生成下三角 subsequent mask，保证解码器自回归地只看见过去。
    输出：BoolTensor，[1, 1, size, size]，True 表示遮挡。
    """
    # 上三角（对角线以上）为 1；我们把它当作需要『遮挡』的位置
    attn_shape = (1, 1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return subsequent  # True 表示遮挡


# =====================
# 位置编码（正余弦）
# =====================
class PositionalEncoding(nn.Module):
    pe:torch.Tensor
    """正余弦位置编码（论文公式），与 embedding 相加。
    为什么需要它：纯注意力没有位置信息，PE 显式注入顺序感知。
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不作为参数更新，但随模型保存/加载

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ================================
# 缩放点积注意力（核心公式）
# ================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,  # [B, H, S_q, d_k]
        K: torch.Tensor,  # [B, H, S_k, d_k]
        V: torch.Tensor,  # [B, H, S_k, d_v]
        attn_mask: Optional[torch.Tensor] = None,  # [B, 1 or H, S_q, S_k]，True=遮挡
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,S_q,S_k]
        if attn_mask is not None:
          
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # [B,H,S_q,d_v]
        return context, attn


# =====================
# 多头注意力（MHA）
# =====================
class MultiHeadAttention(nn.Module):
    """多头注意力：并行地学习不同『子空间』的注意力模式。
    形状约定：
      输入: x 或 (q, k, v) 形如 [B, S, D]
      内部：拆成 H 个头，每头维度 d_k = d_model // H
    关键技巧：
      - 用 view/transpose 重排张量到 [B, H, S, d_k]
      - 注意 mask 的广播维度
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性映射到 Q,K,V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 注意力核心
        self.attn = ScaledDotProductAttention(dropout)
        # 头拼接后的线性
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.size()
        x = x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,S,d_k]
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, S, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(B, S, H * d_k)  # [B,S,D]
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # [B,1(or H),S_q,S_k]
    ) -> torch.Tensor:
        
        residual = q
        Q = self._split_heads(self.w_q(q))
        K = self._split_heads(self.w_k(k))
        V = self._split_heads(self.w_v(v))

        if attn_mask is not None and attn_mask.size(1) == 1:
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)

        context, _ = self.attn(Q, K, V, attn_mask)
        out = self._combine_heads(context)
        out = self.w_o(out)
        
        out = self.layernorm(residual + self.dropout(out))
        return out


# =====================
# 前馈网络（逐位置）
# =====================
class FeedForward(nn.Module):
    """逐位置前馈：两个线性层，中间激活（GELU/ReLU），支持 dropout。
    每个位置独立计算，等价于 1x1 卷积。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("activation 仅支持 relu/gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.lin2(self.dropout(self.act(self.lin1(x))))
        x = self.layernorm(residual + self.dropout(x))
        return x


# =====================
# Transformer 编码层
# =====================
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation="relu")

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.ff(x)
        return x


# =====================
# Transformer 解码层
# =====================
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation="relu")

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = self.ff(x)
        return x

# =============
# 编码器/解码器
# =============
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, pad_idx: int, max_len: int = 5000):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_mask = make_padding_mask(src, self.pad_idx)
        x = self.pe(self.embed(src))  
        for layer in self.layers:
            x = layer(x, src_mask)
        return x, src_mask  

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, pad_idx: int, max_len: int = 5000):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_mask_pad = make_padding_mask(tgt, self.pad_idx)  
        tgt_mask_sub = make_subsequent_mask(tgt.size(1))     
        tgt_mask = tgt_mask_pad | tgt_mask_sub
        memory_mask = src_mask  

        x = self.pe(self.embed(tgt)) 
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        logits = self.generator(x) 
        return logits, tgt_mask


# =============
# 顶层模型
# =============
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 5000,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_layers, num_heads, d_ff, dropout, pad_idx, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, dropout, pad_idx, max_len)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        memory, src_mask = self.encoder(src)
        logits, tgt_mask = self.decoder(tgt_inp, memory, src_mask)
        return logits, (src_mask, tgt_mask)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_idx: int, eos_idx: int, max_len: int = 64) -> torch.Tensor:
        device = src.device
        memory, src_mask = self.encoder(src)
        B = src.size(0)
        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            logits, _ = self.decoder(ys, memory, src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B,1]
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_idx).all():
                break
        return ys


# =====================
# 使用示例（最小可运行）
# =====================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PAD, BOS, EOS = 0, 1, 2
    SRC_V, TGT_V = 50, 60
    model = Transformer(SRC_V, TGT_V, d_model=128, num_layers=2, num_heads=4, d_ff=256, dropout=0.1, pad_idx=PAD).to(device)

    B, S, T = 4, 7, 6
    src = torch.randint(3, SRC_V, (B, S), device=device)
    tgt = torch.randint(3, TGT_V, (B, T), device=device)
    src[:, -2:] = PAD
    tgt[:, -1:] = PAD

    tgt_inp = torch.cat([torch.full((B, 1), BOS, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)
    tgt_out = tgt  

    logits, (src_mask, tgt_mask) = model(src, tgt_inp)
    print("logits:", logits.shape)  # [B,T,V]
    print("src_mask:", src_mask.shape, "tgt_mask:", tgt_mask.shape)

    loss = F.cross_entropy(logits.view(-1, TGT_V), tgt_out.view(-1), ignore_index=PAD)
    print("loss:", float(loss))

    # 极简训练循环（仅演示若干步）
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    model.train()
    for step in range(5):
        optim.zero_grad()
        logits, _ = model(src, tgt_inp)
        loss = F.cross_entropy(logits.view(-1, TGT_V), tgt_out.view(-1), ignore_index=PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 稍作梯度裁剪更稳定
        optim.step()
        print(f"step {step}: loss={loss.item():.4f}")

    # 贪心解码演示
    model.eval()
    with torch.no_grad():
        gen = model.greedy_decode(src, BOS, EOS, max_len=10)
        print("generated:", gen)
