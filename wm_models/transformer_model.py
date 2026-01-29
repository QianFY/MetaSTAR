import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from wm_models.attention_blocks import get_vector_mask
from wm_models.attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache
import torchkit.pytorch_utils as ptu


class StochasticTransformer(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout, task_embedding_size):
        super().__init__()

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary
        self.task_token = nn.Parameter(torch.randn(1, 1, feat_dim))

        self.task_embedding = nn.Linear(feat_dim, task_embedding_size)

    def forward(self, sample, action, mask):
        B, L, _ = sample.shape
        feats = self.stem(torch.cat([sample, action], dim=-1))
        feats = self.position_encoding(feats)
        cls_tokens = repeat(self.task_token, '1 1 D -> B 1 D', B = B)
        feats = torch.cat([feats, cls_tokens], dim=1)  # (B, L+1, D)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)
        task_encoding = self.task_embedding(feats[:, -1, :])  # (B, D)
        return feats[:, :-1, :], task_encoding  # (B, L, D), (B, D)
    
    def predict_next(self, sample, action, mask):
        # [T, L, D]
        T, L, _ = sample.shape
        feats = self.stem(torch.cat([sample, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)
        feats = feats.view(T * L, -1)
        feats = feats.unsqueeze(1)  # (T*L, 1, D)
        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)
            
        return feats


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout, task_embedding_size):
        super().__init__()
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6) 

        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))

        self.task_embedding = nn.Linear(feat_dim, task_embedding_size)

    def forward(self, sample, action, mask):
        B, L, _ = sample.shape
        feats = self.stem(torch.cat([sample, action], dim=-1))
        feats = self.position_encoding(feats)
        cls_tokens = repeat(self.cls_token, '1 1 D -> B 1 D', B = B)
        feats = torch.cat([feats, cls_tokens], dim=1)  # (B, L+1, D)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)
        task_encoding = self.task_embedding(feats[:, -1, :])  # (B, D)
        # task_encoding = F.normalize(task_encoding, dim=-1)
        # task_encoding = feats[:, -1, :]  # (B, D)
        return feats[:, :-1, :], task_encoding  # (B, L, D), (B, D)
    

    def predict_next(self, sample, action, mask):
        # [B, 1, D]
        feats = self.stem(torch.cat([sample, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)
        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)
            
        return feats
    
    def predict_next_with_context(self, sample, action, mask):
        T, L, _ = sample.shape
        feats = self.stem(torch.cat([sample, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)
        
        return feats

    def reset_kv_cache_list(self, batch_size):
        '''
        Reset self.kv_cache_list
        '''
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), device=ptu.device))

    def forward_with_kv_cache(self, sample, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert sample.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1]+1, ptu.device)
        
        feats = self.stem(torch.cat([sample, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        return feats
