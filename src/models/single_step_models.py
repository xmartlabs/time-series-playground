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
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                   kernel_size=(kernel_size,),
                                   activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
        ])


class RNNModel(TimeSeriesModel):
    def build_model(self, **kwargs):
        lstm_units = kwargs.get('lstm_units')
        self.model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(lstm_units, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])
