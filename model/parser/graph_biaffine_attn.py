from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.nn import *
from model.utils.nn import adjust_for_sentinel
from model.utils.nn import get_encoder
from debug import save_heatmap
import numpy as np

class GraphBiaffineAttention(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.config = config

        self.head_arc_feedforward = nn.Linear(config['feat_dim'], config['arc_representation_dim'])
        self.dept_arc_feedforward = nn.Linear(config['feat_dim'], config['arc_representation_dim'])

        self.arc_bilinear = BilinearMatrixAttention(config['arc_representation_dim'],
                                    config['arc_representation_dim'],
                                    activation = nn.ReLU() if self.config['biaffine_activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type=self.config['bias_type'],
                                    arc_norm=self.config['arc_norm'],
                                    )

        self.head_tag_feedforward = nn.Linear(config['feat_dim'], config['tag_representation_dim'])
        self.dep_tag_feedforward = nn.Linear(config['feat_dim'], config['tag_representation_dim'])

        self._dropout = nn.Dropout(config['mlp_dropout'])
        self._head_sentinel = torch.nn.Parameter(torch.randn(config['feat_dim']))
        self._run_inits(config['feat_dim'], config['arc_representation_dim'])
        self.tag_representation_dim = config['tag_representation_dim']

    def forward(
        self,
        input: torch.FloatTensor,
        mask: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:

        batch_size, _, encoding_dim = input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        
        input = torch.cat([head_sentinel, input], dim=1)
        input = self._dropout(input)
            
        # shape (batch_size, sequence_length, config['arc_representation_dim'])
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(input)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(input)))
        dep_tag = self._dropout(F.elu(self.dep_tag_feedforward(input)))

        arc_logits = self.arc_bilinear(head_arc, dept_arc)

        output = {
            'head_tag': head_tag,
            'dep_tag': dep_tag,
            'arc_logits': arc_logits,
            'mask': mask,
        }

        return output

    def _run_inits(self, encoder_dim, config):
        if self.config['parser_init'] == 'xu':
            self.apply(self._init_weights_xavier_uniform)
        elif self.config['parser_init'] == 'norm':
            self.apply(self._init_norm)
        elif self.config['parser_init'] == 'xu+norm':
            self.apply(self._init_weights_xavier_uniform)
            torch.nn.init.normal_(self.head_arc_feedforward.weight, std=np.sqrt(2 / (encoder_dim + config['arc_representation_dim'])))
            torch.nn.init.normal_(self.dept_arc_feedforward.weight, std=np.sqrt(2 / (encoder_dim + config['arc_representation_dim'])))
        if self.config['bma_init'] == 'norm':
            torch.nn.init.normal_(self.arc_bilinear._weight_matrix, std=np.sqrt(2 / (encoder_dim + config['arc_representation_dim'])))
            torch.nn.init.normal_(self.arc_bilinear._bias, std=np.sqrt(2 / (encoder_dim + config['arc_representation_dim'])))

    def _init_norm(self, module):
        """
        Initialize module parameters using a normal distribution.
        Applies nn.init.normal_ to weight and bias tensors with mean=0.0 and std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size).
        For 1D tensors (e.g., biases or projection vectors), temporarily unsqueeze to 2D.
        """
        # Weights
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight
            if w.dim() < 2:
                # Temporarily make it 2D
                w_unsq = w.unsqueeze(0)
                nn.init.normal_(w_unsq, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))
                module.weight.data = w_unsq.squeeze(0)
            else:
                nn.init.normal_(w, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))

        # Biases
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            b = module.bias
            if b.dim() < 2:
                b_unsq = b.unsqueeze(0)
                nn.init.normal_(b_unsq, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))
                module.bias.data = b_unsq.squeeze(0)
            else:
                nn.init.normal_(b, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))

    def _init_weights_xavier_uniform(self, module):
        """
        Initialize module parameters using Xavier Uniform initialization.
        Applies nn.init.xavier_uniform_ to weight and bias tensors.
        For 1D tensors (e.g., biases), temporarily unsqueeze to make them 2D.
        """
        # Initialize weights if they exist and are tensors.
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            if module.weight.dim() < 2:
                # For 1D tensors, unsqueeze to apply Xavier uniform.
                weight_unsqueezed = module.weight.unsqueeze(0)
                nn.init.xavier_uniform_(weight_unsqueezed)
                module.weight.data = weight_unsqueezed.squeeze(0)
            else:
                nn.init.xavier_uniform_(module.weight)

        # Initialize biases if they exist and are tensors.
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            if module.bias.dim() < 2:
                bias_unsqueezed = module.bias.unsqueeze(0)
                nn.init.xavier_uniform_(bias_unsqueezed)
                module.bias.data = bias_unsqueezed.squeeze(0)
            else:
                nn.init.xavier_uniform_(module.bias)

    @classmethod
    def get_model(cls, config):
        model_obj = cls(config=config)
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj

