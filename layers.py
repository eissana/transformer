import torch
import torch.nn.functional as F
import torch.nn as nn

class Head(nn.Module):
    '''
    Self-attention head layer.
    '''
    def __init__(self, head_size, config):
        super().__init__()

        nembd = config['nembd']
        block_size = config['block_size']

        self.query = nn.Linear(nembd, head_size, bias=False)
        self.key = nn.Linear(nembd, head_size, bias=False)
        self.value = nn.Linear(nembd, head_size, bias=False)
        self.dropout = nn.Dropout(config['dropout'])
        # tril is not a model parameter so we register it as a buffer.
        # block_size is the maximum size. The actual size can be smaller.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        _, T, C = x.shape
        query = self.query(x)
        key = self.key(x)
        weights = query @ key.transpose(-2, -1) * C**-0.5
        
        # The time dimension can be smaller than the block-size.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        value = self.value(x)
        out = weights @ value

        return out

  
class MultiHead(nn.Module):
    def __init__(self, nhead, head_size, config):
        super().__init__()

        nembd = config['nembd']

        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(nhead)])
        self.proj = nn.Linear(nembd, nembd)
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, fain_in, fan_out, config):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(fain_in, 4 * fan_out),
            nn.ReLU(),
            nn.Linear(4 * fan_out, fan_out),  # projection
            nn.Dropout(config['dropout'])
        )
        
    def forward(self, x):
        out = self.feed_forward(x)
        return out

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        nembd = config['nembd']
        nhead = config['nhead']

        self.self_attention_layer_norm = nn.LayerNorm(nembd)
        self.self_attention = MultiHead(nhead, nembd//nhead, config)
        self.feed_forward_layer_norm = nn.LayerNorm(nembd)
        self.feed_forward = FeedForward(nembd, nembd, config)
    
    def forward(self, x):
        out = self.self_attention_layer_norm(x)
        # adding residual connections too.
        out = out + self.self_attention(out)
        out = self.feed_forward_layer_norm(out)
        out = out + self.feed_forward(out)
        return out