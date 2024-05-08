import torch


class DataHandler():
    def __init__(self, filename, encoding):
        with open(filename, 'r', encoding=encoding) as f:
            self.text = f.read()
        self.vocab = sorted(set(self.text))
        self.atoi = {c: i for i, c in enumerate(self.vocab)}
        self.itoa = {i: c for i, c in enumerate(self.vocab)}
   
    def data(self):
        return torch.tensor(self.encode(self.text), dtype=torch.long)
    
    def encode(self, text=None):
        if text is None:
            text = self.text
        return [self.atoi[c] for c in text]

    def decode(self, encoded_text):
        return ''.join([self.itoa[i] for i in encoded_text.tolist()])

    def nchars(self):
        return len(self.vocab)

def split(data, train_ratio=0.8, valid_ratio=0.1):
    data_size = len(data)
    ntrain, nvalid = int(train_ratio*data_size), int(valid_ratio*data_size)

    data = {'train': data[:ntrain],
            'valid': data[ntrain:ntrain+nvalid],
            'test': data[ntrain+nvalid:]}
    return data

def get_batch(data, config):
    '''
    Generates a batch of examples.
    (x[i], y[i]) is a pair of consecutive characters in the text.
    '''
    indices = torch.randint(len(data)-config['block_size'], (config['nbatch'],))
    x = torch.stack([data[i:i+config['block_size']] for i in indices])
    y = torch.stack([data[i+1:i+config['block_size']+1] for i in indices])
    return x, y