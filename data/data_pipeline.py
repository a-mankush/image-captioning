import tensorflow as tf

from utils.config import AUTOTUNE, BATCH_SIZE, IMAGE_SIZE


def decode_and_resize(img_path):
    img = tf.io.read_file(
        img_path
    )  # result is a tensor of type tf.string containing the raw bytes of the image file.
    img = tf.image.decode_jpeg(
        img, channels=3
    )  # decode the raw bytes into a tensor representing the image
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset
