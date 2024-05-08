import torch
import torch.nn.functional as F
import torch.nn as nn

class Head(nn.Module):
    '''
    Self-attention head layer.
    '''
    def __init__(self, head_size, config):
        super().__init__()
        self.query = nn.Linear(config['nembd'], head_size, bias=False)
        self.key = nn.Linear(config['nembd'], head_size, bias=False)
        self.value = nn.Linear(config['nembd'], head_size, bias=False)
        # tril is not a model parameter so we register it as a buffer.
        # block_size is the maximum size. The actual size can be smaller.
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        
    def forward(self, x):
        _, T, C = x.shape
        query = self.query(x)
        key = self.key(x)
        weights = query @ key.transpose(-2, -1) * C**-0.5
        
        # The time dimension can be smaller than the block-size.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        
        value = self.value(x)
        out = weights @ value

        return out

  
class MultiHead(nn.Module):
    def __init__(self, nhead, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(nhead)])
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return out


class FeedForward(nn.Module):
    def __init__(self, fain_in, fan_out):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(fain_in, fan_out),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.feed_forward(x)
        return out

class Block(nn.Module):
    def __init__(self, fan_in, fan_out, config):
        super().__init__()
        self.self_attention = MultiHead(fan_out, fan_in//fan_out, config)
        self.feed_forward = FeedForward(fan_in, fan_in)
    
    def forward(self, x):
        out = self.self_attention(x)
        out = self.feed_forward(out)
        return out