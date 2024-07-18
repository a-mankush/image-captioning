import keras

cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
)
