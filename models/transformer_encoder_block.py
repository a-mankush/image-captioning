import keras
from keras import layers


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.mhsa1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )

        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(0.2)

        self.dense1 = layers.Dense(embed_dim, activation="relu")

        self.point_wise_fc = keras.Sequential(
            [layers.Dense(embed_dim, activation="relu"), layers.Dense(embed_dim)]
        )

    def call(self, inputs, training, mask=None):
        inputs = self.layernorm1(inputs)
        inputs = self.dense1(inputs)

        inputs = self.layernorm2(inputs)
        mhsa1_output = self.mhsa1(
            key=inputs,
            query=inputs,
            value=inputs,
            training=training,
            attention_mask=None,
        )

        skip_output = mhsa1_output + inputs
        layer_norm_output = self.layernorm3(skip_output)
        encoder_output = self.point_wise_fc(layer_norm_output)
        encoder_output = self.dropout1(encoder_output, training=training)

        return encoder_output
