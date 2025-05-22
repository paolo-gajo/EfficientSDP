from model.stepparser import StepParser
from model.utils import is_file
import torch
from peft import LoraConfig, get_peft_model, TaskType
from model.utils.train_utils import print_params

def build_model(config, model_start_path = None, verbose = False):
    """
        build and return full architecture from the config
    """

    ## get model from config
    model = StepParser(config)

    ## move to the correct device
    model.to(torch.device(config['device']))

    ## if model_path is provided, we load that model checkpoint, it's used for finetuning
    if model_start_path:
        model_path = model_start_path
    elif config['model_path']:
        model_path = config['model_path']

    ## let's load the model if path exists! 
    if is_file(model_path):
        model.load_state_dict(torch.load(model_path))
        
    ## freeze encoder if asked for
    if config['freeze_encoder']:
        model.encoder.freeze_encoder()

    ## freeze tagger if asked for
    if config['freeze_tagger']:
        model.freeze_tagger()

    ## freeze parser if asked for
    if config['freeze_parser']:
        model.freeze_parser()

    if config['use_gnn_steps'] > 0:
        model.freeze_gnn()
    
    if len(config['unfreeze_layers']):
        for l in config['unfreeze_layers']:
            for param in model.encoder.encoder.encoder.layer[l].parameters():
                param.requires_grad = True
    
    if config['use_lora']:
        print('Applying LoRA...')
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
            # layers_to_transform=[11],
        )
        model.encoder.encoder = get_peft_model(model.encoder.encoder, lora_config)
    
    if verbose: print_params(model)
    
    return model