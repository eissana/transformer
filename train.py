import torch
import numpy as np
import matplotlib.pyplot as plt

from transformer import Transformer
import util
import metrics
from data_handler import DataHandler, split
from config import config


if __name__ == '__main__':
    dh = DataHandler('data/hafez.txt', encoding='utf-16')
    config['nchars'] = dh.nchars()

    data = split(dh.data(), train_ratio=0.8, valid_ratio=0.1)

    model = Transformer(nhead=4, config=config)
    print(f'Model parameters: {util.nparameters(model)}')
    
    losses = metrics.optimize(model, data, config)
    torch.save(model, config['model_filename'])

    eval_size = 10
    print(f'final training loss: {np.mean(losses['train'][-eval_size:])}')
    print(f'final validation loss: {np.mean(losses['valid'][-eval_size:])}')
 
    plt.plot(torch.tensor(losses['train']).view(-1, eval_size).mean(axis=1));
    plt.plot(torch.tensor(losses['valid']).view(-1, eval_size).mean(axis=1));
    plt.legend(['training loss', 'validation loss'])
    plt.show()
 