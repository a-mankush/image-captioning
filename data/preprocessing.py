import re

import keras
import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization

from data.data_pipeline import make_dataset
from data.load_data import load_captions_data, train_val_split


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")


# Load the Dataset
captions_mapping, text_data = load_captions_data(CAPTIONS_PATH)
train_data, valid_data = train_val_split(captions_mapping)
print("Number of Traning Samples: ", len(train_data))
print("Number of Validation Samples: ", len(valid_data))

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

vectorization.adapt(text_data)

# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)

# Pass the list of images path and the list of corrosponding captions
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))
