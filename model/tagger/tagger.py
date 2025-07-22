import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.modeling_outputs import TokenClassifierOutput
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
        tagger_rnn_type = self.config['tagger_rnn_type']
        if config['use_tagger_rnn']:
            if tagger_rnn_type == 'lstm':
                self.seq_encoder = nn.LSTM(
                    input_size=encoder_output_size,
                    hidden_size=hidden_size_tagger,
                    num_layers=config['tagger_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=config['tagger_dropout'] if config['tagger_rnn_layers'] > 1 else 0,
                )
            elif tagger_rnn_type == 'gru':
                self.seq_encoder = nn.GRU(
                    input_size=encoder_output_size,
                    hidden_size=hidden_size_tagger,
                    num_layers=config['tagger_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=config['tagger_dropout'] if config['tagger_rnn_layers'] > 1 else 0,
                )
            else:
                warnings.warn(f"Tagger RNN type `{tagger_rnn_type}` is neither `gru` nor `lstm`. Setting it to None.")
                self.seq_encoder = None
            classifier_input_size = 2 * hidden_size_tagger
        else:
            classifier_input_size = encoder_output_size
        
        self.classifier = nn.Linear(classifier_input_size, self.num_tags)
        
        self.tag_dropout = nn.Dropout(config['tag_dropout'])

        if config['tag_embedding_type'] == 'linear':
            self.tag_embedder = nn.Linear(config["n_tags"], config["tag_representation_dim"])
            print('Using nn.Linear for tag embeddings!')
        elif config['tag_embedding_type'] == 'embedding':
            self.tag_embedder = nn.Embedding(config["n_tags"], config["tag_representation_dim"])
            print('Using nn.Embedding for tag embeddings!')
        elif config['tag_embedding_type'] == 'none':
            self.tag_embedder = None
            print('NOT using tag embeddings!')
        else:
            raise ValueError('Parameter `tag_embedding_type` can only be == `linear` or `embedding` or `none`!')

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

    def forward(self, input: torch.Tensor, mask: torch.Tensor, labels=None):
        """
        Uses encoder representations to predict a tag for each token.
        
        Args:
            input: Tensor of shape (batch_size, seq_len, hidden_size) from the encoder.
            mask: Binary mask of shape (batch_size, seq_len) where 1 indicates valid tokens.
            labels: (Optional) Tensor of shape (batch_size, seq_len) containing true tag labels.
        """
        self.labels = labels
        input = self.dropout(input)
        
        if self.config['use_tagger_rnn']:
            # Compute lengths from the binary mask.
            lengths = mask.sum(dim=1).cpu()
            # Pack the padded sequence using the lengths.
            packed_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.seq_encoder(packed_input)
            # Unpack the sequence, ensuring the output has the original sequence length.
            input, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input.size(1))

        logits = self.classifier(input)
        
        # Compute loss if in training or validation mode.
        if self.mode in ['train', 'validation']:
            loss = self.tagger_loss(logits.reshape(-1, self.num_tags), labels.reshape(-1))
        else:  # For testing mode, no loss is computed.
            loss = None
        output = TokenClassifierOutput(loss=loss, logits=logits)
        
        if self.config["tag_embedding_type"] != 'none':
            output.tag_embeddings = self.make_embeddings(output, labels)
        else:
            output.tag_embeddings = None
        
        return output

    def make_embeddings(self, output, labels):
        pos_tags_pred_one_hot = self.get_predicted_classes_as_one_hot(output.logits)

        # Ground-truth tags
        try:
            pos_tags_gt = torch.nn.functional.one_hot(labels, num_classes=self.config["n_tags"])
        except:
            warnings.warn("Ground truth tags are unavailable, using predicted tags for all purposes.")
            pos_tags_gt = pos_tags_pred_one_hot

        # Use predicted or ground truth tags based on config
        pos_tags_parser_one_hot = pos_tags_pred_one_hot if self.config["use_pred_tags"] else pos_tags_gt
        pos_tags_parser_cls_idx = torch.argmax(output.logits, dim=-1) if self.config["use_pred_tags"] else labels

        pos_tags_dict = {
            'pos_tags_one_hot': pos_tags_parser_one_hot.float(),
            'pos_tags_labels': pos_tags_parser_cls_idx,
        }

        if self.config['tag_embedding_type'] == 'linear':
            pos_tags = pos_tags_dict['pos_tags_one_hot']
        elif self.config['tag_embedding_type'] == 'embedding':
            pos_tags = pos_tags_dict['pos_tags_labels']
        elif self.config['tag_embedding_type'] == 'none':
            pos_tags = None
        else:
            raise ValueError('parameter `tag_embedding_type` can only be == `linear` or `embedding`')
        embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
        return embeddings

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
            class_labels = torch.argmax(logits_masked, dim=1) if self.config['use_pred_tags'] else self.labels[i]
            tagger_out_classes.append(class_labels.cpu().detach().numpy().tolist())

        return tagger_out_classes
