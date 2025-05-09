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
            
def check_param_norm(model):
    param_norms = []
    for name, param in model.named_parameters():
        param_norms.append(torch.norm(param))
    return torch.mean(torch.Tensor(param_norms))

def check_param_norm(model):
    param_norms = []
    for name, param in model.named_parameters():
        param_norms.append(torch.norm(param))
    return torch.mean(torch.Tensor(param_norms))

def check_grad_norm(model):
    """
    Computes the average norm of gradients for all parameters in a model that have gradients.
    
    Args:
        model: PyTorch model
        
    Returns:
        torch.Tensor: Average gradient norm (scalar)
    """
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(torch.norm(param.grad))
    
    if len(grad_norms) > 0:
        return torch.mean(torch.stack(grad_norms))
    else:
        return torch.tensor(0.0)