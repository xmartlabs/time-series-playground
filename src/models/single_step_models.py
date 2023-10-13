import tensorflow as tf
from models.time_series_model import TimeSeriesModel


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

    def get_eval_scores(self, window):
        train_score = self.evaluate(window.train)
        val_score = self.evaluate(window.val)
        test_score = self.evaluate(window.test, verbose=0)
        return train_score, val_score, test_score

    def save_model(self, filepath):
        self.save(filepath)


class LinearModel(TimeSeriesModel):
    def build_model(self, **kwargs):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1)
        ])


class DenseModel(TimeSeriesModel):
    def build_model(self, **kwargs):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])


class MultiStepDense(TimeSeriesModel):
    def build_model(self, **kwargs):
        self.model = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])


class ConvModel(TimeSeriesModel):
    def build_model(self, **kwargs):
        kernel_size = kwargs.get('conv_width', 3)
        out_steps = kwargs.get('out_steps', 1)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                   kernel_size=(kernel_size,),
                                   activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=out_steps),
            tf.keras.layers.Reshape([out_steps, 1])
        ])


class RNNModel(TimeSeriesModel):
    def build_model(self, **kwargs):
        lstm_units = kwargs.get('lstm_units')
        out_steps = kwargs.get('out_steps', 1)
        self.model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=out_steps),
            tf.keras.layers.Reshape([out_steps, 1])
        ])


class ConvRNNModel(TimeSeriesModel):
    def build_model(self, **kwargs):
        lstm_units = kwargs.get('lstm_units', 64)
        kernel_size = kwargs.get('conv_width', 3)
        out_steps = kwargs.get('out_steps', 1)
        dropout = kwargs.get('dropout', 0.2)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=kernel_size,
                                   strides=2,
                                   activation="relu",
                                   padding='causal',
                                   ),  # input_shape=[window_size, 1]
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dropout(rate=dropout),
            tf.keras.layers.Dense(units=out_steps),
            tf.keras.layers.Reshape([out_steps, 1])
        ])
