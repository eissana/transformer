import torch
import util
from config import config
from data_handler import DataHandler 

if __name__ == '__main__':
    dh = DataHandler('data/hafez.txt', encoding='utf-16')    
    model = torch.load(config['model_filename'])
    util.gen_text(model, config, dh.decode)
