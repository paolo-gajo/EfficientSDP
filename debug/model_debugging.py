import torch
from collections import defaultdict
import re
import numpy as np

def collect_parser_layer_grads(model, grad_norm_parser):
    grad_norm_layers = defaultdict(list)
    for i, (name, param) in enumerate(model.parser.encoder_h.named_parameters()):
        d = int(re.search(r'\d', name).group(0))
        grad_norm_layers[d].append(param.grad.detach().cpu().norm())
    grad_norm_layers = [np.mean(el) for i, el in grad_norm_layers.items()]
    grad_norm_parser.append(grad_norm_layers)

def nan_checker(model_output, model, model_label = ''):
    has_nan = torch.isnan(model_output).any()
    if has_nan:
        raise ValueError(f'{model_label} output has NaNs!')
    for name, param in model.named_parameters():
        if param is not None:
            if torch.isnan(param).any():
                print(f"NaNs found in weights of parameter: {name}")
                raise ValueError(f'{model_label} weights have NaNs!')
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaNs found in gradients of parameter: {name}")
                raise ValueError(f'{model_label} gradients have NaNs!')
            
def check_param_norm(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        norms = torch.stack([p.detach().norm() for p in model.parameters()])
    return norms.mean()

def check_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        norms = [p.grad.detach().norm() for p in model.parameters()
                 if p.grad is not None]
        if not norms:
            return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(norms).mean()
