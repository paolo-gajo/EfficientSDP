import unittest
import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.initializers import InitializerApplicator, Initializer
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary

from model.parser.biaffine_dependency_parser_simple import MTRFGParserSimple

class TestDependencyParserSimple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = MTRFGParserSimple.from_donatelli_config()
        cls.parser.eval()
        cls.embedding_dim = cls.parser.encoder.get_input_dim()

    def test_input_change_changes_output(self):
        embedded_text = torch.randn(2, 3, self.embedding_dim)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        out1 = self.parser.forward(embedded_text, mask)
        embedded_text[0][1] = torch.randn(self.embedding_dim)
        out2 = self.parser.forward(embedded_text, mask)

        self.assertTrue(out1['loss'] != out2['loss'])

    def test_masked_input_change_doesnt_changes_output(self):
        embedding_dim = self.parser.encoder.get_input_dim()
        embedded_text = torch.randn(2, 3, self.embedding_dim)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        out1 = self.parser.forward(embedded_text, mask)
        embedded_text[0][2] = torch.randn(embedding_dim)
        out2 = self.parser.forward(embedded_text, mask)
        
        self.assertTrue(out1['loss'] == out2['loss'])

    def test_same_output_as_old_parser(self):
        # TODO: forward the parser and then assert something. Maybe check that output is the same with old parses loaded from the config file or sth like that.
        # (put the config file under the test folder)
        # NOTE: I've stopped working on this currently as it seemed to take too much time

        # vocab = Vocabulary()
        # embedding_dim = 768
        # encoder = Seq2SeqEncoder.by_name('stacked_bidirectional_lstm')(input_size=embedding_dim, hidden_size=400,
        #     num_layers=3, recurrent_dropout_probability=0.3, use_highway=True)
        # old_parser = MTRFGParser(vocab, None, encoder, 10, 10)
        # parser = MTRFGParserSimple(encoder)
        pass


