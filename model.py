import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, self.num_heads, 3 * self.d_k).permute(2, 0, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = x.permute(1, 2, 0, 3).contiguous().view(B, T, C)
        return self.out_proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attn(x))
        x = self.norm1(x)
        x = x + self.dropout2(self.ff(x))
        x = self.norm2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SpatialEmbedding(nn.Module):
    def __init__(self, num_nodes, d_model):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, d_model)

    def forward(self, node_ids):
        return self.emb(node_ids)

class OTBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        cost = torch.cdist(x, x, p=2)
        align = torch.softmax(-cost, dim=-1)
        x = self.w2(torch.matmul(align, self.w1(x)))
        return x

class MultiScaleTemporal(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, scales=[1,2,4]):
        super().__init__()
        self.scales = scales
        self.encoders = nn.ModuleList([TemporalEncoder(d_model, num_heads, d_ff, num_layers) for _ in scales])

    def forward(self, x):
        outputs = []
        for i, scale in enumerate(self.scales):
            x_scaled = x[:, ::scale, :]
            out = self.encoders[i](x_scaled)
            outputs.append(out)
        min_len = min([o.size(1) for o in outputs])
        outputs = [o[:, :min_len, :] for o in outputs]
        return torch.mean(torch.stack(outputs, dim=0), dim=0)

class OTSTModel(nn.Module):
    def __init__(self, num_nodes, d_model=128, d_ff=256, num_heads=8, num_layers=4, horizon=12):
        super().__init__()
        self.num_nodes = num_nodes
        self.spatial_emb = SpatialEmbedding(num_nodes, d_model)
        self.temporal_enc = MultiScaleTemporal(d_model, num_heads, d_ff, num_layers)
        self.ot_block = OTBlock(d_model)
        self.fc_out = nn.Linear(d_model, horizon)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, node_ids):
        B, T, N = x.size()
        se = self.spatial_emb(node_ids).unsqueeze(1).repeat(1, T, 1)
        x = x + se
        x = x.permute(0, 2, 1)
        x = x.reshape(B*N, T, -1)
        x = self.temporal_enc(x)
        x = self.ot_block(x)
        x = self.layer_norm(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x.view(B, N, -1)

class GSA(nn.Module):
    def __init__(self, d_model, num_nodes):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.num_nodes = num_nodes

    def forward(self, x, adj):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(Q.size(-1)) * adj, dim=-1)
        out = torch.matmul(attn, V)
        return out

class GTA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.temporal_enc = TemporalEncoder(d_model, num_heads, d_model*2, 2)

    def forward(self, x):
        return self.temporal_enc(x)

class OTAlignment(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ot_block = OTBlock(d_model)

    def forward(self, gsa_feat, gta_feat):
        fused = gsa_feat + gta_feat
        aligned = self.ot_block(fused)
        return aligned

class OTSTNet(nn.Module):
    def __init__(self, num_nodes, d_model=128, d_ff=256, num_heads=8, num_layers=4, horizon=12):
        super().__init__()
        self.gsa = GSA(d_model, num_nodes)
        self.gta = GTA(d_model, num_heads)
        self.ot_align = OTAlignment(d_model)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x, adj, node_ids):
        gsa_feat = self.gsa(x, adj)
        gta_feat = self.gta(x)
        ot_feat = self.ot_align(gsa_feat, gta_feat)
        out = self.fc(ot_feat[:, -1, :])
        return out

class MultiHeadAggregator(nn.Module):
    def __init__(self, d_model, num_heads, horizon):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d_model, d_model//num_heads) for _ in range(num_heads)])
        self.out_proj = nn.Linear(d_model, horizon)

    def forward(self, x):
        head_outs = [h(x) for h in self.heads]
        concat = torch.cat(head_outs, dim=-1)
        return self.out_proj(concat)

class HierarchicalOTST(nn.Module):
    def __init__(self, num_nodes, d_model=128, d_ff=256, num_heads=8, num_layers=4, horizon=12):
        super().__init__()
        self.spatial_enc = GSA(d_model, num_nodes)
        self.temporal_enc = GTA(d_model, num_heads)
        self.ot_align = OTAlignment(d_model)
        self.aggregator = MultiHeadAggregator(d_model, num_heads, horizon)

    def forward(self, x, adj, node_ids):
        spatial_feat = self.spatial_enc(x, adj)
        temporal_feat =
