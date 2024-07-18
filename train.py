import keras
import tensorflow as tf

from data.load_data import load_captions_data, train_val_split
from data.preprocessing import train_dataset, valid_dataset
from models.cnn import get_cnn_model
from models.decoder import TransformerDecoderBlock
from models.image_captioning_model import ImageCaptioningModel
from models.loss import cross_entropy
from models.lr import LRSchedule
from models.transformer_encoder_block import TransformerEncoderBlock
from utils.config import EMBED_DIM, EPOCHS, FF_DIM, TEXT_PATH


def main():
    # Load and prepare data
    captions_mapping, text_data = load_captions_data(TEXT_PATH)
    train_data, valid_data = train_val_split(captions_mapping)

    # Initialize model components
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(EMBED_DIM, num_heads=2)
    decoder = TransformerDecoderBlock(EMBED_DIM, FF_DIM, num_heads=4)

    # Create and compile the model
    image_captioning_model = ImageCaptioningModel(encoder, decoder, cnn_model)

    num_train_steps = len(train_dataset) * EPOCHS
    num_warmup_steps = num_train_steps // 15
    lr_schedule = LRSchedule(
        post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps
    )

    image_captioning_model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy
    )

    # Train the model
    history = image_captioning_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
    )

    # # Save the model
    # image_captioning_model.save_weights("image_captioning_weights.h5")


if __name__ == "__main__":
    main()
