import torch
import torch.nn.functional as F
import time
from bidi.algorithm import get_display

def nparameters(model):
    '''
    Returns the total number of model parameters.
    '''
    return sum([p.nelement() for p in model.parameters()])

def gen_text(model, config, decoder):
    '''
    Generates text using the model starting from nothing.
    '''
    # starting from '\n' char we generate text.
    x = torch.zeros((1, 1), dtype=torch.long)
    while True:
        logits = model(x[:, -config['block_size']:])
        # only consider the last logit
        logits = logits[:, -1, :]
        score = F.softmax(logits, dim=1)
        next_token = score.multinomial(1)
        x = torch.cat((x, next_token), dim=1)
        print(get_display(f'\r{decoder(x[0])}'), end='')
        time.sleep(.1)
