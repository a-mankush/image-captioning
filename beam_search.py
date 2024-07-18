import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.data_pipeline import decode_and_resize
from data.preprocessing import valid_data, vectorization
from models import image_captioning_model
from utils.config import SEQ_LENGTH

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_len = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())


def generate_caption_beam_search(beam_width=3):
    sample_img = np.random.choice(valid_images)
    sample_img = decode_and_resize(sample_img)

    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    img = tf.expand_dims(sample_img, 0)
    img = image_captioning_model.cnn_model(img)
    encoded_image = image_captioning_model.encoder(img, training=False)

    start_token = vectorization.get_vocabulary().index("<start>")
    end_token = vectorization.get_vocabulary().index("<end>")

    initial_state = tf.constant([[start_token]] * beam_width)
    initial_probs = tf.zeros((beam_width,))

    beams = [(initial_state[i], initial_probs[i]) for i in range(beam_width)]

    for _ in range(max_decoded_sentence_len):
        candidates = []
        for sequence, sequence_prob in beams:
            if sequence[-1] == end_token:
                candidates.append((sequence, sequence_prob))
                continue

            tokenized_caption = tf.expand_dims(sequence, 0)
            mask = tf.not_equal(tokenized_caption, 0)
            predictions = image_captioning_model.decoder(
                tokenized_caption,
                training=False,
                encoder_outputs=encoded_image,
                mask=mask,
            )

            top_k_values, top_k_indices = tf.nn.top_k(
                predictions[0, -1, :], k=beam_width
            )

            for i in range(beam_width):
                token = top_k_indices[i].numpy()
                prob = top_k_values[i].numpy()
                new_sequence = tf.concat([sequence, [token]], axis=0)
                new_prob = sequence_prob - tf.math.log(prob)
                candidates.append((new_sequence, new_prob))

        ordered = sorted(candidates, key=lambda tup: tup[1])
        beams = ordered[:beam_width]

        if all(sequence[-1] == end_token for sequence, _ in beams):
            break

    best_sequence, _ = beams[0]
    decoded_caption = " ".join(
        [
            index_lookup[token.numpy()]
            for token in best_sequence[1:]
            if token != end_token
        ]
    )
    print("Predicted Caption:", decoded_caption)


# Generate captions using beam search
for _ in range(3):
    generate_caption_beam_search()
