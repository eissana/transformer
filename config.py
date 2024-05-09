config = {
    'nepoch': 1000,  # number of epochs/iterations
    'nbatch': 30,  # batch size.
    'block_size': 8,  # text block-size to process
    'nembd': 32,  # size of embeddings
    'head_size': 32,  # self-attention head-size
    'nhead': 4,  # number of self-attention heads in each block
    'nblock': 3,  # self-attention blocks
    'lr': 1e-3,  # learning rate.
    'dropout': 0.1,  # dropout rate
    'model_filename': 'bin/model.pt',
}
