import torch
import torch.nn.functional as F
import data_handler

def get_loss(logits, y):
    '''
    Computes cross-entropy loss, given logits and labels.
    '''
    B, T, C = logits.shape
    # F.cross_entropy expects size C, (B, C), or (B, C, ...)
    # logits shape is (B, T, C), so we flatten the first two dimensions.
    return F.cross_entropy(logits.view(B*T, C), y.view(B*T))

def optimize(model, data, config, losses=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    if losses is None:
        losses = {
            'train': [],
            'valid': [],
        }
    for _ in range(config['nepoch']):
        x, y = data_handler.get_batch(data['train'], config)

        logits = model(x)
        loss = get_loss(logits, y)
        losses['train'].append(loss.item())

        with torch.no_grad():
            x, y = data_handler.get_batch(data['valid'], config)
            logits = model(x)
            vloss = get_loss(logits, y)
            losses['valid'].append(vloss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return losses