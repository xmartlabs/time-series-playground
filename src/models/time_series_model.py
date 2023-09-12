import tensorflow as tf


class TimeSeriesModel:

    model = None

    def __init__(self, tracker):
        self.tracker = tracker

    # hidden1_size=512, hidden2_size=128, l2_param=0.002, dropout_factor=0.2, bias_regularizer='l1'
    def build_model(self, **kwargs):
        raise NotImplementedError()

    def compile_and_fit(self, window, patience=2, epochs=20):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = self.model.fit(window.train, epochs=epochs,
                                 validation_data=window.val,
                                 callbacks=[early_stopping])
        return history

    def predict(self, batch_generator):
        pass
