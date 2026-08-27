import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # causal mask to ensure that attention is only performed to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; self-attend: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ Transformer block """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence uzunluğu block_size'ı aşamaz"

        tok_emb = self.token_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (T, n_embd)

        x = self.drop(tok_emb + pos_emb)  # token + pozisyon bilgisi birleşiyor

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # context'i block_size ile sınırla
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # sadece son token'ın logit'i

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    class MLP(nn.Module):
      def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)  # genişlet
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * n_embd, n_embd)  # tekrar daralt
        self.dropout = nn.Dropout(dropout)

      def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    class Block(nn.Module):
      def __init__(self, n_embd, n_head, block_size, dropout=0.1):
          super().__init__()
          self.ln1 = nn.LayerNorm(n_embd)
          self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
          self.ln2 = nn.LayerNorm(n_embd)
          self.mlp = MLP(n_embd, dropout)

      def forward(self, x):
          x = x + self.attn(self.ln1(x))  # residual connection + attention
          x = x + self.mlp(self.ln2(x))   # residual connection + MLP
          return x
    class GPT(nn.Module):
      def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
          super().__init__()
          self.block_size = block_size

          self.token_emb = nn.Embedding(vocab_size, n_embd)
          self.pos_emb = nn.Embedding(block_size, n_embd)
          self.drop = nn.Dropout(dropout)

          self.blocks = nn.ModuleList([
              Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
          ])

          self.ln_f = nn.LayerNorm(n_embd)
          self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

      def forward(self, idx, targets=None):
          B, T = idx.shape
          assert T <= self.block_size, "Sequence uzunluğu block_size'ı aşamaz"

          tok_emb = self.token_emb(idx)  # (B, T, n_embd)
          pos = torch.arange(T, device=idx.device)
          pos_emb = self.pos_emb(pos)  # (T, n_embd)

          x = self.drop(tok_emb + pos_emb)  # token + pozisyon bilgisi birleşiyor

          for block in self.blocks:
              x = block(x)

          x = self.ln_f(x)
          logits = self.lm_head(x)  # (B, T, vocab_size)

          loss = None
          if targets is not None:
              loss = F.cross_entropy(
                  logits.view(-1, logits.size(-1)),
                  targets.view(-1)
              )

          return logits, loss

      @torch.no_grad()
      def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
          for _ in range(max_new_tokens):
              idx_cond = idx[:, -self.block_size:]  # context'i block_size ile sınırla
              logits, _ = self(idx_cond)
              logits = logits[:, -1, :] / temperature  # sadece son token'ın logit'i

              if top_k is not None:
                  v, _ = torch.topk(logits, top_k)
                  logits[logits < v[:, [-1]]] = float('-inf')

              probs = F.softmax(logits, dim=-1)
              idx_next = torch.multinomial(probs, num_samples=1)
              idx = torch.cat((idx, idx_next), dim=1)

          return idx
