import keras
import tensorflow as tf
from keras import layers


class ImageCaptioningModel(keras.Model):
    def __init__(
        self, encoder, decoder, cnn_model, num_caption_per_image=5, data_aug=None
    ):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.cnn_model = cnn_model
        self.num_caption_per_image = num_caption_per_image
        self.data_aug = data_aug
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.Mean(name="accuracy")

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(
            y_true, tf.argmax(y_pred, axis=2)
        )  # y_pred.shape => (batch_size, seq_len, vocab_size)
        accuracy = tf.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _calculate_loss_and_accuracy(self, img_embed, batch_seq, training):
        encoder_output = self.encoder(img_embed, training=training)
        decoder_input_seq = batch_seq[:, :-1]
        decoder_output_seq = batch_seq[:, 1:]
        mask = tf.not_equal(decoder_output_seq, 0)
        decoder_output = self.decoder(
            decoder_input_seq,
            training=training,
            encoder_outputs=encoder_output,
            mask=mask,
        )

        loss = self.calculate_loss(decoder_output_seq, decoder_output, mask)
        accuracy = self.calculate_accuracy(decoder_output_seq, decoder_output, mask)

        return loss, accuracy

    def train_step(self, batch_data):  # batch_data -> tf.data.Dataset
        batch_image, batch_seq = batch_data
        batch_loss = 0
        batch_accuracy = 0

        if self.data_aug:
            batch_image = self.data_aug(batch_image)

        img_embed = self.cnn_model(batch_image)

        for i in range(self.num_caption_per_image):
            with tf.GradientTape() as tape:
                loss, accuracy = self._calculate_loss_and_accuracy(
                    img_embed,
                    batch_seq[
                        :, i, :
                    ],  # Batch_seq-> (batch_size, no_of_caption, seq_len)
                    training=True,
                )

                batch_loss += loss
                batch_accuracy += accuracy

            train_var = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )
            gradients = tape.gradient(loss, train_var)
            self.optimizer.apply_gradients(zip(gradients, train_var))

        batch_accuracy /= float(self.num_caption_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.accuracy_tracker.update_state(batch_accuracy)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, batch_data):
        batch_image, batch_seq = batch_data
        img_embed = self.cnn_model(batch_image)

        batch_loss = 0
        batch_accuracy = 0
        for i in range(self.num_caption_per_image):
            loss, accuracy = self._calculate_loss_and_accuracy(
                img_embed,
                batch_seq[:, i, :],  # Batch_seq-> (batch_size, no_of_caption, seq_len)
                training=False,
            )

            batch_loss += loss
            batch_accuracy += accuracy

        batch_accuracy /= float(
            self.num_caption_per_image
        )  # Average across all captions
        self.loss_tracker.update_state(batch_loss)
        self.accuracy_tracker.update_state(batch_accuracy)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.accuracy_tracker]
