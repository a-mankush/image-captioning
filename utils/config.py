# Configuration parameters
import tensorflow as tf

IMAGE_SIZE = (229, 229)
VOCAB_SIZE = 8672
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
IMAGES_PATH = "Flickr8k_Dataset/Flicker8k_Dataset/"
TEXT_PATH = "Flickr8k_text/Flickr8k.token.txt"
