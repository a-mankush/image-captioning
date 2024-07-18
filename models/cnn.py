import keras
from keras import layers
from keras.applications import InceptionResNetV2, InceptionV3, efficientnet

from utils.config import IMAGE_SIZE


def get_EfficientNetB0():
    return efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )


def get_InceptionResNetV2():
    return InceptionResNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )


def get_InceptionV3():
    return InceptionV3(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )


def get_cnn_model(model_name="EfficientNetB0"):
    if model_name == "EfficientNetB0":
        base_model = get_EfficientNetB0()
    elif model_name == "InceptionResNetV2":
        base_model = get_InceptionResNetV2()
    elif model_name == "InceptionV3":
        base_model = get_InceptionV3()

    # Frease our feature extractor
    base_model.trainable = False
    base_model_out = base_model.output
    # (None, 8, 8, 1280) ->(None, 8*8, 1280)(None,         64,       1280)
    # Intrepretation of the shape  ->      (Batch_size, seq_len, emded_dim) <- for understanding only not actually sequence length or embed_dim
    # This is the input require by Encoder
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)

    return cnn_model
