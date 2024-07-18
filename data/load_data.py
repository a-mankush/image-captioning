import os
from typing import DefaultDict

import numpy as np

from utils.config import IMAGES_PATH, SEQ_LENGTH


def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
      filename: Path to the text file containing caption data.

    Returns:
      caption_mapping: Dictionary with keys as image ids and values as a list of
        captions for the image.
      text_data: List containing all captions from the file.
    """

    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = DefaultDict(list)
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            image_name, caption = line.split("\t")

            image_name = image_name.split("#")[0]
            image_name = os.path.join(IMAGES_PATH, image_name.strip())

            # Remove caption that are either too short or too ling
            tokens = caption.strip().split()
            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(image_name)
                continue

            if image_name not in images_to_skip and image_name.endswith("jpg"):
                caption = "<start> " + caption.strip() + " <end>"
                caption_mapping[image_name].append(caption)
                text_data.append(caption)

    for image_name in images_to_skip:
        if image_name in caption_mapping:
            del caption_mapping[image_name]

    return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    all_images = list(caption_data.keys())

    if shuffle:
        np.random.shuffle(all_images)

    # Split into train and validation set
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }

    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    return training_data, validation_data
