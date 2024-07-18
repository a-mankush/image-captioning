import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from data.data_pipeline import decode_and_resize
from data.preprocessing import valid_data, vectorization
from models import image_captioning_model
from utils.config import SEQ_LENGTH

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_len = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())


def generate_caption():
    sample_img = np.random.choice(valid_images)

    sample_img = decode_and_resize(sample_img)

    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    img = tf.expand_dims(sample_img, 0)
    img = image_captioning_model.cnn_model(img)

    encoded_image = image_captioning_model.encoder(img, training=False)

    decoded_caption = "<start> "

    for i in range(max_decoded_sentence_len):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.not_equal(tokenized_caption, 0)
        predictions = image_captioning_model.decoder(
            tokenized_caption, training=False, encoder_outputs=encoded_image, mask=mask
        )

        sampled_token_index = np.argmax(predictions[0, i, :])

        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "")
    print("Predicted Caption:", decoded_caption)


generate_caption()
generate_caption()
generate_caption()
