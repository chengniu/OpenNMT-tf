import opennmt as onmt


class Transformer3p3(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self, dtype=tf.float32):
    super(Transformer3p3, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=256,
            dtype=dtype),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=256,
            dtype=dtype),
        num_layers=3,
        num_units=256,
        num_heads=4,
        ffn_inner_dim=512,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1)
