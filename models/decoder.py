import keras
import tensorflow as tf
from keras import layers

from models.positional_enbeddings import PositionalEmbedding
from utils.config import EMBED_DIM, SEQ_LENGTH, VOCAB_SIZE


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super(TransformerDecoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.mhsa1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.2
        )
        self.mhsa2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.2
        )

        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(0.5)

        self.embeddings = PositionalEmbedding(
            sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM
        )

        self.point_wise_fc1 = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

        self.dense = layers.Dense(VOCAB_SIZE, activation="softmax")

    def call(self, inputs, training, encoder_outputs, mask=None):
        embed_inputs = self.embeddings(inputs)  # (5, 24, 512)
        causal_mask = self.get_causal_attention_mask(
            embed_inputs
        )  # casual_mask.shape => (batch, target_seq_len, source_seq_len)

        if mask is not None:  # mask.shape => (batch, target_seq_len)
            padding_mask = tf.cast(
                mask[:, :, tf.newaxis], dtype=tf.int32
            )  # padding_mask.shape => (batch, target_seq_len, 1)
            combined_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype=tf.int32
            )  # combined_mask.shape => (batch, 1, target_seq_len)
            combined_mask = tf.minimum(
                combined_mask, causal_mask
            )  # combined_mask.shape => (batch, target_seq_len, source_seq_len)
        else:
            padding_mask = None
            combined_mask = None

        inputs1 = self.layernorm1(embed_inputs)
        attention_output1 = self.mhsa1(
            key=inputs1,
            query=inputs1,
            value=inputs1,
            attention_mask=combined_mask,  # shape -> (B, T, S) => (batch_size, target_sqe_len, source_seq_len)
            training=training,
            # use_causal_mask=True
        )

        inputs2 = self.layernorm2(attention_output1 + embed_inputs)
        attention_output2 = self.mhsa2(
            key=encoder_outputs,
            query=inputs2,
            value=encoder_outputs,
            attention_mask=padding_mask,  # shape -> (batch_size, target_sqeuence, 1) => source_seq_len is 1 cause want to apply it to all uniformly
            # as there is no zero or pad token present in encoder input cause the input is the image feature map and always have same lenght
            training=training,
        )

        inputs3 = self.layernorm3(attention_output2 + inputs2)
        fc_output1 = self.point_wise_fc1(inputs3)
        output3 = self.dropout1(fc_output1, training=training)

        output4 = self.dense(output3)

        return output4

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.tile(mask, mult)
