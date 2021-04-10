import tensorflow as tf


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_logs):
        self.training_logs = training_logs

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.training_logs.append(logs)