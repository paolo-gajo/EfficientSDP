"""
    Build optimizer from the configuration file!
"""

from allennlp.training.optimizers import Optimizer
from allennlp.common.params import Params

def build_optimizer(config, model):
    """
        We use allennlp's optimizers, which are supported to get optmizers using 
        from_params function. Different params take different arguments, so 
        for now, we are just supporting learning_rate in the arguments, rest 
        of the arguments are defaults.
    """
    params = Params({'type':config['optimizer'], 'lr': config['learning_rate'], 'betas':config['betas']})
    optimizer = Optimizer.from_params(model_parameters = model.named_parameters(), params = params)
    return optimizer