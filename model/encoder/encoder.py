import torch 
import torch.nn as nn
from transformers import AutoModel, BertConfig#, BertModel
from model.encoder.bert_custom import BertModelNoPos, BertModelSteps, BertModelLaplacian, BertModel
from transformers import BatchEncoding
from typing import List, Dict, Optional
import inspect

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        if self.config['use_bert_positional_embeddings']:
            if self.config['use_abs_step_embeddings']:
                bert_config = BertConfig()
                bert_config.max_steps = self.config['max_steps']
                # bert_config.device = self.config['device']
                self.encoder = BertModelSteps.from_pretrained(self.config['model_name'], config=bert_config)
            elif self.config['laplacian_pe'] == 'encoder':
                bert_config = BertConfig()
                bert_config.max_steps = self.config['max_steps']
                # bert_config.device = self.config['device']
                self.encoder = BertModelLaplacian.from_pretrained(self.config['model_name'], config=bert_config)
            else:
                self.encoder = BertModel.from_pretrained(self.config['model_name'])
        else:
            self.encoder = BertModelNoPos.from_pretrained(self.config['model_name'])
        self.encoder_input_keys = [key for key in inspect.signature(self.encoder.forward).parameters.keys()]
        if self.config['use_multihead_attention']:
            self.mha = nn.MultiheadAttention(self.config['encoder_output_dim'], self.config['self_attention_heads'], batch_first = True, dropout = 0.2)


    def forward(
        self, 
        encoded_input: Dict,
        ) -> torch.Tensor:

        input_to_encoder = {key : encoded_input[key] if key in encoded_input else None for key in self.encoder_input_keys}
        if self.config['laplacian_pe'] == 'encoder':
            assert 'graph_laplacian' in input_to_encoder.keys(), 'graph_laplacian not in bert keys'
        input_to_encoder = {key : value for key, value in input_to_encoder.items() if not key.endswith('_custom')}
        outputs = self.encoder(**input_to_encoder)
        
        if self.config['rep_mode'] == 'words':
            encoded_output = self.merge_subword_representation(outputs, encoded_input)
            attention_mask = encoded_input['words_mask_custom']
        elif self.config['rep_mode'] == 'tokens':
            encoded_output = outputs.last_hidden_state
            attention_mask = encoded_input['attention_mask']

        if self.config['use_multihead_attention']:
            print('Using new MHA after encoder output.')
            # `attention_mask` here was just the original encoded_input['attention_mask'] although the representation was merged
            encoded_output, _ = self.mha(encoded_output, encoded_output, encoded_output, key_padding_mask = attention_mask < 0.5 )

        return encoded_output

    def freeze_encoder(self):
        """
            freeze model parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
            unfreeze model parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

    def merge_subword_representation(self, outputs, encoded_input: BatchEncoding):
        """
            Merges subword representations
            Parameters:
            ----------
            outputs: Of type torch.Tensor
            encoded_input: Of type BatchEncoding
        """

        ## keep it same dimensional as original output to keep 
        ## it batch friendly!
        
        outputs_new = outputs.last_hidden_state.clone()

        for i, batch in enumerate(outputs.last_hidden_state):
            word_idxs = encoded_input['word_ids_custom'][i]

            tot_words = torch.max(word_idxs).item() + 1
            
            for word_idx in range(tot_words):
                """
                    merging subword representations
                """
                word_representation = batch[torch.where(word_idxs == word_idx)[0]].mean(dim = 0)
                outputs_new[i][word_idx] = word_representation
        
        return outputs_new

class BERTWordEmbeddings(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bert_embeddings = AutoModel.from_pretrained(model_name).embeddings
        self.word_embeddings = bert_embeddings.word_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings
        pass
    
    def forward(self, input: Dict[str, torch.Tensor]):
        E = self.word_embeddings(input['input_ids'])
        T = self.token_type_embeddings(input['token_type_ids'])
        output = E + T
        # output = self.merge_subword_representation(output, input)
        return output
    
    def merge_subword_representation(self, outputs, encoded_input):
        """
        (taken from MTRFG)
        Merges subword representations
        Parameters:
        ----------
        outputs: Of type torch.Tensor
        encoded_input: Of type BatchEncoding
        """

        ## keep it same dimensional as original output to keep 
        ## it batch friendly!

        outputs_new = outputs.clone()
        # sizes = []
        for i, t in enumerate(outputs):
            word_indices = encoded_input['word_ids_custom'][i]

            tot_words = torch.max(word_indices).item() + 1
            # sizes.append(tot_words)
            
            for word_idx in range(tot_words):
                """
                    merging subword representations
                """

                # here they just overwrite and leave the remaining tokens as they are
                # because in the tagger they are going to mask the tensors
                # that have index > tot_words by applying `words_mask_custom`
                outputs_new[i][word_idx] = t[torch.where(word_indices == word_idx)[0]].mean(dim = 0)
                # outputs_new[i][word_idx] = t[torch.where(word_indices == word_idx)[0]][0]
        # print(sizes)
        return outputs_new

    def freeze_encoder(self):
        """
            freeze model parameters
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
            unfreeze model parameters
        """
        for param in self.parameters():
            param.requires_grad = True