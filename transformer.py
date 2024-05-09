import torch
import torch.nn as nn
import layers

class Transformer(nn.Module):
    '''
    A simple transformer model for building a character-level
    language model using multiple self-attention heads.
    '''
    def __init__(self, config):
        super().__init__()
        nembd = config['nembd']
        nchars = config['nchars']
        block_size = config['block_size']
        nblock = config['nblock']
        self.token_emb = nn.Embedding(nchars, nembd)
        self.position_emb = nn.Embedding(block_size, nembd)
        self.blocks = nn.Sequential(*[layers.Block(config) for _ in range(nblock)])
        self.linear_layer = nn.LayerNorm(nembd)
        self.linear = nn.Linear(nembd, nchars)

    def forward(self, x):
        _, T = x.shape
        # x.shape == (nbatch, block_size, nchars)
        # token_emb.shape == (nbatch, block_size, nembd)
        token_emb = self.token_emb(x)
        # position_emb.shape == (nbatch, block_size, nembd)
        position_emb = self.position_emb(torch.arange(T))
        # out.shape == (nbatch, block_size, nembd)
        out = token_emb + position_emb
        # out.shape == (nbatch, block_size, nembd)
        out = self.blocks(out)
        out = self.linear_layer(out)
        # out.shape == (nbatch, block_size, nchars)
        out = self.linear(out)

        return out
