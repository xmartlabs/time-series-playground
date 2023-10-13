import pandas as pd


def load_dataset():
    return pd.read_parquet('../dataset/LD2011_2014.parquet')


class PowerPreprocessor():

    def preprocess(self, points_per_day=96, include_context=False):
        df = load_dataset()
        # Slice [start:stop:step], starting from index 5 take every 6th record.
        # df = df[5::6]
        df = df.rename(columns={'Unnamed: 0': 'datetime'})
        df['datetime'] = pd.to_datetime(df.pop('datetime'), format='%Y-%m-%d %H:%M:%S')
        without_date = df.drop(columns=['datetime'])
        df['total'] = without_date.sum(axis=1)
        df = df[['datetime', 'total']]
        if include_context:
            df['weekday'] = df['datetime'].dt.weekday
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['dayofyear'] = df['datetime'].dt.dayofyear
            # Normalize year column
            df['year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
            df['dayofyear'] = df['dayofyear'] / 365.0
        df = df.set_index('datetime')

        # Split the data
        n = len(df)
        split1 = (int(n * 0.7) // points_per_day) * points_per_day
        split2 = (int(n * 0.9) // points_per_day) * points_per_day
        train_df = df[0:split1]
        val_df = df[split1:split2]
        test_df = df[split2:]

        # Normalize the data
        #
        # It is important to scale features before training a neural network. Normalization is a common way of doing this scaling:
        # subtract the mean and divide by the standard deviation of each feature.

        # TODO: Normalize by all power columns
        # train_mean = train_df[power_columns].mean(axis=None)
        train_mean = train_df['total'].mean()
        train_std = train_df['total'].std()

        train_df.loc[:, 'total'] = (train_df['total'] - train_mean) / train_std
        val_df.loc[:, 'total'] = (val_df['total'] - train_mean) / train_std
        test_df.loc[:, 'total'] = (test_df['total'] - train_mean) / train_std

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        self.num_features = df.shape[1]
