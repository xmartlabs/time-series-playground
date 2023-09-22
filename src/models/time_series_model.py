import os
import tempfile

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


class TimeSeriesModel(tf.keras.Model):

    model = None

    def __init__(self):
        super().__init__(self)

    def build_model(self, **kwargs):
        raise NotImplementedError()

    def compile_and_fit(self, window, patience=2, epochs=2):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       mode='min')
        log_dir = os.path.join(tempfile.gettempdir(), 'tensorboard')
        tboard = TensorBoard(log_dir=log_dir, write_images=False, write_graph=False)

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = self.model.fit(window.train, epochs=epochs,
                                 validation_data=window.val,
                                 callbacks=[early_stopping, tboard])
        return history

    def predict(self, batch_generator):
        pass
