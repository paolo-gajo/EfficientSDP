import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings

class Tagger(nn.Module):
    """
    This is a tagger head. Input is a representation coming from the encoder 
    and the output is class scores for each token.
    """
    def __init__(self, config): 
        """
        Args:
            config: A dictionary with keys:
                - encoder_output_dim: Dimension of encoder representations.
                - n_tags: Number of class labels.
                - use_tagger_rnn: Boolean to use the LSTM tagger.
                - gumbel_softmax: Boolean for using Gumbel softmax.
        """
        super().__init__()
        self.config = config
        hidden_dropout_prob = 0.2
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        hidden_size_tagger = 128
        encoder_output_size = self.config['encoder_output_dim']
        self.num_tags = self.config['n_tags']
        parser_rnn_type = self.config['parser_rnn_type']
        if config['use_tagger_rnn']:
            if parser_rnn_type == 'lstm':
                self.seq_encoder = nn.LSTM(
                    input_size=encoder_output_size,
                    hidden_size=hidden_size_tagger,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif parser_rnn_type == 'gru':
                self.seq_encoder = nn.GRU(
                    input_size=encoder_output_size,
                    hidden_size=hidden_size_tagger,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            else:
                warnings.warn(f"Parser type `{parser_rnn_type}` is neither `gru` nor `lstm`. Setting it to None.")
                self.seq_encoder = None
            classifier_input_size = 2 * hidden_size_tagger
        else:
            classifier_input_size = encoder_output_size
        
        self.classifier = nn.Linear(classifier_input_size, self.num_tags)
        self.tagger_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.gumbel_softmax = self.config['gumbel_softmax']
        self.apply(self._init_weights)
        self.mode = 'train'

    def set_mode(self, mode='train'):
        """
        Set mode for training, validation, or testing.
        """
        self.mode = mode

    def _init_weights(self, module):
        """
        Initialize module parameters using Xavier Uniform initialization.
        Applies nn.init.xavier_uniform_ to weight and bias tensors.
        For 1D tensors (e.g., biases), temporarily unsqueeze to make them 2D.
        """
        # Initialize weights if they exist and are tensors.
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            if module.weight.dim() < 2:
                # For 1D tensors, unsqueeze to apply Xavier uniform.
                weight_unsqueezed = module.weight.unsqueeze(0)
                nn.init.xavier_uniform_(weight_unsqueezed)
                module.weight.data = weight_unsqueezed.squeeze(0)
            else:
                nn.init.xavier_uniform_(module.weight)
                
        # Initialize biases if they exist and are tensors.
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            if module.bias.dim() < 2:
                bias_unsqueezed = module.bias.unsqueeze(0)
                nn.init.xavier_uniform_(bias_unsqueezed)
                module.bias.data = bias_unsqueezed.squeeze(0)
            else:
                nn.init.xavier_uniform_(module.bias)

    def forward(self, encoder_reps: torch.Tensor, mask: torch.Tensor, labels=None):
        """
        Uses encoder representations to predict a tag for each token.
        
        Args:
            encoder_reps: Tensor of shape (batch_size, seq_len, hidden_size) from the encoder.
            mask: Binary mask of shape (batch_size, seq_len) where 1 indicates valid tokens.
            labels: (Optional) Tensor of shape (batch_size, seq_len) containing true tag labels.
        """
        encoder_reps = self.dropout(encoder_reps)
        
        if self.config['use_tagger_rnn']:
            # Compute lengths from the binary mask.
            lengths = mask.sum(dim=1).cpu()
            # Pack the padded sequence using the lengths.
            packed_input = pack_padded_sequence(encoder_reps, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.seq_encoder(packed_input)
            # Unpack the sequence, ensuring the output has the original sequence length.
            encoder_reps, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=encoder_reps.size(1))
        
        logits = self.classifier(encoder_reps)
        
        # Compute loss if in training or validation mode.
        if self.mode in ['train', 'validation']:
            loss = self.tagger_loss(logits.reshape(-1, self.num_tags), labels.reshape(-1))
        else:  # For testing mode, no loss is computed.
            loss = None

        return TokenClassifierOutput(loss=loss, logits=logits)

    def softargmax(self, input, beta=100):
        """
        Differentiable soft argmax to retain gradients.
        """
        *_, n = input.shape
        input = nn.functional.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, n - 1, n).to(input.device)
        result = torch.sum(input * indices, dim=-1)
        pred_tags = torch.round(torch.clamp(result, min=0, max=n - 1)).long()
        return pred_tags

    def get_predicted_classes_as_one_hot(self, tagger_output, temperature=1e-3):
        """
        Returns one-hot vectors of predicted classes using a low-temperature softmax
        for differentiability.
        """
        tagger_output = tagger_output - (torch.max(tagger_output, dim=-1))[0].unsqueeze(-1)
        if self.gumbel_softmax:
            tagger_output = nn.functional.gumbel_softmax(tagger_output, tau=temperature, hard=False, dim=-1)
        else:
            tagger_output = nn.functional.softmax(tagger_output / temperature, dim=-1)
        return tagger_output

    def get_predicted_classes(self, tagger_output):
        """
        Get predicted classes from the tagger's output logits.
        """
        pred_classes = self.softargmax(tagger_output)
        return pred_classes

    def make_output_human_readable(self, tagger_output, attention_mask):
        """
        Converts tagger outputs to a human-readable format by applying the attention mask.
        
        Args:
            tagger_output: A TokenClassifierOutput containing logits.
            attention_mask: A binary mask to extract logits corresponding to valid tokens.
        """
        batchsize = tagger_output.logits.shape[0]
        tagger_out_classes = []
        
        for i in range(batchsize):
            logits = tagger_output.logits[i]
            mask = attention_mask[i]
            logits_masked = logits[torch.where(mask == 1)[0]]
            class_labels = torch.argmax(logits_masked, dim=1)
            tagger_out_classes.append(class_labels.cpu().detach().numpy().tolist())

        return tagger_out_classes
