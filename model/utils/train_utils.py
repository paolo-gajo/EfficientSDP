from tqdm import tqdm
import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def print_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return total_params, trainable_params

def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    model.set_mode('train')

    with tqdm(train_loader, position=0, leave = False) as pbar:
        for inp_data in pbar:
            optimizer.zero_grad()
            loss = model(inp_data)
            
            ## if loss is nan, we keep going forward
            if torch.isnan(loss).item():
                continue
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}", refresh = True)
            loss.backward()
            optimizer.step()

    return model

def validate_epoch(model, data_loader):
    """
        runs validation
    """
    model.eval()
    model.set_mode('validation')

    ## let's get total loss over the validation set
    losses = [model(inp_data).item() for inp_data in tqdm(data_loader, position=0, leave = False)]

    return round(np.mean(losses), 3)

def get_scheduler(optimizer, warmup_steps: int, train_steps: int, scheduler_type: str = None, use_warmup: bool = False):
    if use_warmup:
        if scheduler_type == 'linear':
            return get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_train_steps=train_steps,)
        elif scheduler_type == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_train_steps=train_steps,)
    else:
        return None