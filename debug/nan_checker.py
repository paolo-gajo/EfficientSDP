import torch

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
        