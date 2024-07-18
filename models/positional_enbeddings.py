import keras
import keras_nlp
import tensorflow as tf
from keras import layers


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = keras_nlp.layers.PositionEmbedding(sequence_length)

        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        embed_tokens = self.token_embeddings(inputs)
        embed_tokens = embed_tokens * self.embed_scale
        embed_postion = self.position_embeddings(embed_tokens)
        return embed_tokens + embed_postion

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
