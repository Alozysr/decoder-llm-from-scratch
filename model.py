"""
model.py - Decoder-Only Transformer (GPT) Mimarisi
Bu dosya modelin tüm yapı taşlarını (Self-Attention, Transformer Blokları ve GPT sınıfı) içerir.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Çok Başlıklı Nedensel Öz-Dikkat Mekanizması (Multi-Head Causal Self-Attention)
    Gelecekteki token'ları maskelemek için alt üçgensel matris (tril) kullanır.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd, n_head'e tam bölünmeli!"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V projeksiyonlarını tek bir lineer katmanda birleştirerek verimlilik sağlıyoruz
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Çıktı projeksiyonu
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Düzenlileştirme (Dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Nedensel maske (Causal mask): Gelecekteki token'ları görmeyi engeller
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension

        # Q, K, V hesapla ve ayır
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Boyutlandırma: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention skorları: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Değerlerle çarpım: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        # Başlıkları birleştir: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Çıkış projeksiyonu ve residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """
    Tek bir Transformer Decoder Bloğu
    Pre-LayerNorm mimarisi: Norm -> Attention -> Residual -> Norm -> MLP -> Residual
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Decoder-Only Transformer Dil Modeli (GPT Mimarisi)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blokları dizisi
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying (Ağırlık Paylaşımı): Parametre tasarrufu sağlar ve dil modelleme performansını artırır
        self.token_emb.weight = self.lm_head.weight

        # Ağırlıkları ilklendir
        self.apply(self._init_weights)

        # Özel residual projeksiyon ölçeklendirmesi (GPT-2 standardı)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Toplam eğitilebilir parametre sayısını döndürür."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Dizi uzunluğu ({T}), izin verilen block_size ({self.config.block_size}) değerini aşamaz!"
        )

        # Token ve pozisyon embedding'lerini topla
        tok_emb = self.token_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_emb = self.pos_emb(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Transformer bloklarından geçir
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Oto-regresif metin üretimi fonksiyonu.
        Temperature, Repetition Penalty, Top-k ve Top-p (Nucleus) filtrelemelerini destekler.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Bağlam boyutu block_size'ı aşarsa son kısmı al
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Yalnızca son token çıktısını al: (B, vocab_size)

            # Repetition penalty uygula
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    for token in set(idx[b].tolist()):
                        if logits[b, token] > 0:
                            logits[b, token] /= repetition_penalty
                        else:
                            logits[b, token] *= repetition_penalty

            # Sıcaklık ölçeklendirmesi
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtreleme
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (Nucleus) filtreleme
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Eşiği aşan token'ları maskele
                sorted_indices_to_remove = cumulative_probs > top_p
                # En az bir token'ı elde tutmak için kaydır
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Maskeyi orijinal sıralamaya geri yerleştir
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Olasılık dağılımını hesapla ve bir sonraki token'ı seç
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Üretilen token'ı mevcut diziye ekle
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
