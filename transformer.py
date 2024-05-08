import torch
import torch.nn as nn
import layers

class Transformer(nn.Module):
    '''
    A simple transformer model for building a character-level
    language model using multiple self-attention heads.
    '''
    def __init__(self, nhead, config):
        super().__init__()
        self.token_emb = nn.Embedding(config['nchars'], config['nembd'])
        self.position_emb = nn.Embedding(config['block_size'], config['nembd'])
        self.blocks = nn.Sequential(
            layers.Block(config['nembd'], nhead, config),
            layers.Block(config['nembd'], nhead, config),
            layers.Block(config['nembd'], nhead, config),
        )
        self.linear = nn.Linear(config['nembd'], config['nchars'])

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
        # out.shape == (nbatch, block_size, nchars)
        out = self.linear(out)

        return out
