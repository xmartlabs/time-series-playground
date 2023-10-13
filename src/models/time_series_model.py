import os
import tempfile

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau


class TimeSeriesModel(tf.keras.Model):

    model = None

    def __init__(self):
        super().__init__(self)

    def build_model(self, **kwargs):
        raise NotImplementedError()

    def compile_and_fit(self, window, lr=0.001, patience=4, epochs=2):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=patience * 2,
                                       mode='min')
        log_dir = os.path.join(tempfile.gettempdir(), 'tensorboard')
        tboard = TensorBoard(log_dir=log_dir, write_images=False, write_graph=False)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, cooldown=3,
                                      patience=patience, min_lr=lr / 100)

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = self.model.fit(window.train, epochs=epochs,
                                 validation_data=window.val,
                                 callbacks=[early_stopping, tboard, reduce_lr])
        return history

    def get_eval_scores(self, window):
        train_score = self.model.evaluate(window.train)[1]
        val_score = self.model.evaluate(window.val)[1]
        test_score = self.model.evaluate(window.test, verbose=0)[1]
        return train_score, val_score, test_score

    def save_model(self, filepath):
        self.model.save(filepath)

    def __call__(self, inputs):
        return self.model(inputs)
