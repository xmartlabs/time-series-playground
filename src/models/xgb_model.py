
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import plot_tree

from models.time_series_model import TimeSeriesModel


class XGBoostModel(TimeSeriesModel):
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    X_test = None
    y_test = None

    def build_model(self, **kwargs):
        estimators = kwargs.get('estimators', 20)
        max_depth = kwargs.get('max_depth', 4)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        subsample = kwargs.get('subsample', 1)
        regularization = kwargs.get('regularization', 1)
        self.model = xgb.XGBRegressor(n_estimators=estimators, max_depth=max_depth, eval_metric='mae',
                                      early_stopping_rounds=early_stopping_rounds,
                                      subsample=subsample, reg_lambda=regularization)

    def _convert_dataset(self, window):
        if self.X_train is not None:
            return
        train_u = list(window.train.unbatch().as_numpy_iterator())
        val_u = list(window.val.unbatch().as_numpy_iterator())
        test_u = list(window.test.unbatch().as_numpy_iterator())
        self.X_train = [x[0][:, 0] for x in train_u]
        self.y_train = [x[1][:, 0] for x in train_u]
        self.X_val = [x[0][:, 0] for x in val_u]
        self.y_val = [x[1][:, 0] for x in val_u]
        self.X_test = [x[0][:, 0] for x in test_u]
        self.y_test = [x[1][:, 0] for x in test_u]

    def compile_and_fit(self, window, lr=0.001, patience=4, epochs=2):
        self._convert_dataset(window)
        history = self.model.fit(self.X_train, self.y_train,
                                 eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                                 verbose=True)
        return history

    def get_eval_scores(self, window):
        self._convert_dataset(window)
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_val)
        test_pred = self.model.predict(self.X_test)
        train_score = mean_absolute_error(y_true=self.y_train, y_pred=train_pred)
        val_score = mean_absolute_error(y_true=self.y_val, y_pred=val_pred)
        test_score = mean_absolute_error(y_true=self.y_test, y_pred=test_pred)
        return train_score, val_score, test_score

    def plot(self, tracker, num_samples=3):
        assert self.X_train is not None, "You must call compile_and_fit first"
        inputs = self.X_train[0:num_samples]
        labels = self.y_train[0:num_samples]
        plt.figure(figsize=(12, 8))
        max_n = min(5, len(inputs))
        if tracker is not None:
            tracker.log_chart(title='Model Predictions', series='predictions', iteration=1, figure=plt)
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel('Total [normed]')
            label_index_start = inputs[n].shape[0]
            plt.plot(range(label_index_start), inputs[n],
                     label='Inputs', marker='.', zorder=-10)

            plt.scatter(range(label_index_start, label_index_start + 96), labels[n],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if self.model is not None:
                predictions = self.model.predict(inputs)
                plt.scatter(range(label_index_start, label_index_start + 96), predictions[n],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()
            plt.show()
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(self.model)
        plt.show()
        try:
            plot_tree(self.model)
            plt.show()
        except ImportError:
            print('Skipping tree plot: You must install graphviz to support plot tree')

    def save_model(self, filepath):
        self.model.save_model(filepath)
