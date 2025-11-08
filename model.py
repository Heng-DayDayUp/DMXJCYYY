# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1, use_relative=False, max_rel_pos=128):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_relative = use_relative
        self.max_rel_pos = max_rel_pos
        # relative bias table will be set by caller (layers) if use_relative==True

    def forward(self, q, k, v, mask=None, rel_bias=None):
        # q,k,v: (batch, heads, seq_len, dim)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if self.use_relative and rel_bias is not None:
            # rel_bias shape should be (heads, seq_len, seq_len) or broadcastable
            scores = scores + rel_bias.unsqueeze(0)  # broadcast over batch
        if mask is not None:
            # mask: (batch, 1, 1, seq_k) or (batch, 1, seq_q, seq_k)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_relative=False, max_rel_pos=128):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout, use_relative, max_rel_pos)
        self.dropout = nn.Dropout(dropout)

        # relative positional bias table (Shaw-style) per head if requested
        self.use_relative = use_relative
        if use_relative:
            # We'll create the relative distance buckets in the layer
            # bias table shape: (2*max_rel_pos-1, num_heads)
            self.max_rel_pos = max_rel_pos
            self.relative_bias_table = nn.Parameter(torch.zeros((2 * max_rel_pos - 1, num_heads)))
            nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def _split_heads(self, x):
        # x: (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k)
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        # x: (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous()
        b, seq, _, _ = x.size()
        return x.view(b, seq, self.d_model)

    def _compute_rel_bias(self, qlen, klen, device):
        """
        Compute relative position bias matrix of shape (num_heads, qlen, klen)
        Following a simple Shaw-like implementation using a learned lookup table.
        """
        max_rel = self.max_rel_pos
        # distances: (qlen, klen)
        q_pos = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        k_pos = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        relative_pos = k_pos - q_pos  # range -(qlen-1) .. (klen-1)
        relative_pos_clipped = relative_pos.clamp(-max_rel + 1, max_rel - 1) + (max_rel - 1)
        # lookup table: shape (2*max_rel-1, num_heads)
        table = self.relative_bias_table  # (2*max_rel-1, num_heads)
        bias = table[relative_pos_clipped]  # (qlen, klen, num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # (num_heads, qlen, klen)
        return bias

    def forward(self, q, k, v, mask=None):
        bsz = q.size(0)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        rel_bias = None
        if self.use_relative:
            qlen = q.size(2)
            klen = k.size(2)
            device = q.device
            rel_bias = self._compute_rel_bias(qlen, klen, device)  # (heads, qlen, klen)
        if mask is not None:
            # adapt mask shape for broadcast if needed (batch, 1, qlen, klen)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        out, attn = self.attention(q, k, v, mask, rel_bias)
        out = self._combine_heads(out)
        out = self.w_o(out)
        out = self.dropout(out)
        return out, attn


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # GELU or ReLU: choose ReLU for simplicity (can change)
        self.act = F.relu

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_relative=False, max_rel_pos=128, use_residual=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative, max_rel_pos)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x, mask=None):
        attn_out, attn = self.self_attn(x, x, x, mask)
        if self.use_residual:
            x = x + self.dropout(attn_out)
        else:
            x = self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        if self.use_residual:
            x = x + self.dropout(ffn_out)
        else:
            x = self.dropout(ffn_out)
        x = self.norm2(x)
        return x, attn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_relative=False, max_rel_pos=128, use_residual=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative, max_rel_pos)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative, max_rel_pos)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # masked self-attention
        self_attn_out, self_attn = self.self_attn(x, x, x, tgt_mask)
        if self.use_residual:
            x = x + self.dropout(self_attn_out)
        else:
            x = self.dropout(self_attn_out)
        x = self.norm1(x)

        # encoder-decoder attention
        enc_attn_out, enc_attn = self.enc_attn(x, memory, memory, memory_mask)
        if self.use_residual:
            x = x + self.dropout(enc_attn_out)
        else:
            x = self.dropout(enc_attn_out)
        x = self.norm2(x)

        ffn_out = self.ffn(x)
        if self.use_residual:
            x = x + self.dropout(ffn_out)
        else:
            x = self.dropout(ffn_out)
        x = self.norm3(x)
        return x, self_attn, enc_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1,
                 max_len=5000, use_pos=True, use_relative=False, max_rel_pos=128, use_residual=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len) if use_pos else None
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, use_relative, max_rel_pos, use_residual) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.use_pos = use_pos

    def forward(self, src, src_mask=None):
        # src: (batch, seq)
        x = self.embedding(src) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        attns = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attns.append(attn)
        x = self.norm(x)
        logits = self.output(x)
        return logits, x, attns  # return logits and final hidden for encoder-decoder attention


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1,
                 max_len=5000, use_pos=True, use_relative=False, max_rel_pos=128, use_residual=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len) if use_pos else None
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, use_relative, max_rel_pos, use_residual) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.use_pos = use_pos

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: (batch, seq)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        self_attns = []
        enc_attns = []
        for layer in self.layers:
            x, self_attn, enc_attn = layer(x, memory, tgt_mask, memory_mask)
            self_attns.append(self_attn)
            enc_attns.append(enc_attn)
        x = self.norm(x)
        logits = self.output(x)
        return logits, x, self_attns, enc_attns


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1,
                 max_len=5000, use_pos=True, use_relative=False, max_rel_pos=128, use_residual=True):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len, use_pos, use_relative, max_rel_pos, use_residual)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len, use_pos, use_relative, max_rel_pos, use_residual)
        self.tgt_vocab = tgt_vocab

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        logits_enc, memory, enc_attns = self.encoder(src, src_mask)
        logits_dec, dec_hidden, self_attns, enc_attns = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return logits_dec, self_attns, enc_attns

    @staticmethod
    def make_src_mask(src):
        # src padding mask (batch, 1, 1, src_len) - here we assume no padding for char datasets
        # If padding token exists, create mask accordingly.
        return None

    @staticmethod
    def make_tgt_mask(tgt_len, device):
        # causal mask for decoder self-attention: shape (1, 1, tgt_len, tgt_len) or (batch, 1, tgt_len, tgt_len)
        mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,tgt_len,tgt_len)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
