import tensorflow as tf
from models.time_series_model import TimeSeriesModel
from tensorflow.keras.layers import (LSTM, Dense, Conv1D, Dropout, Reshape, Concatenate, IntegerLookup,
                                     BatchNormalization, Activation)
from tensorflow.keras import Input


def build_context_inputs():
    year_input = Input(shape=[1,], dtype=None, name="year")
    month_input = Input(shape=[1,], dtype=tf.uint8, name="month")
    day_input = Input(shape=[1,], dtype=tf.uint8, name="dayofyear")

    weekday_input = Input(shape=[1,], dtype=tf.uint8, name="weekday")
    weekday_one_hot = IntegerLookup(output_mode="one_hot", num_oov_indices=0, vocabulary=list(range(7)))(weekday_input)
    month_one_hot = IntegerLookup(output_mode="one_hot", num_oov_indices=0, vocabulary=list(range(1, 13)))(month_input)
    output = Concatenate(axis=-1)([year_input, day_input, month_one_hot, weekday_one_hot])
    return [year_input, day_input, month_input, weekday_input], output


class CXTConvRNNModel(TimeSeriesModel):
    def build_model(self, **kwargs):
        lstm_units = kwargs.get('lstm_units', 64)
        kernel_size = kwargs.get('conv_width', 3)
        out_steps = kwargs.get('out_steps', 1)
        dropout = kwargs.get('dropout', 0.1)
        input_width = kwargs.get('input_width', 672)
        time_series_input = Input(shape=[input_width, 1], dtype=None, name="time_series")
        contextual_inputs, contextual_output = build_context_inputs()
        ts_x = Conv1D(filters=64, kernel_size=kernel_size,
                      strides=2,
                      activation="relu",
                      padding='causal',
                      )(time_series_input)
        ts_x = LSTM(64, return_sequences=True)(ts_x)
        ts_x = LSTM(lstm_units)(ts_x)
        ctxt_x = Dense(units=8, activation=None)(contextual_output)
        ctxt_x = BatchNormalization()(ctxt_x)
        ctxt_x = Activation('relu')(ctxt_x)
        x = Concatenate(axis=-1)([ts_x, ctxt_x])
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(units=out_steps)(x)
        x = Reshape([out_steps, 1])(x)
        self.model = tf.keras.Model(inputs=[time_series_input, *contextual_inputs], outputs=x)
