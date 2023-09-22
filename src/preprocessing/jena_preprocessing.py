import pandas as pd
import numpy as np
from dataset_loader import JenaDatasetLoader


def load_dataset():
    data_loader = JenaDatasetLoader()
    data_loader.load()
    return data_loader.get_data()


def preprocess_wind(df):
    # One thing that should stand out is the `min` value of the wind velocity (`wv (m/s)`) and the maximum value (`max. wv (m/s)`)
    # columns. This `-9999` is likely erroneous.
    # There's a separate wind direction column, so the velocity should be greater than zero (`>=0`). Replace it with zeros:
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # But this will be easier for the model to interpret if you convert the wind direction and velocity columns to a wind
    # **vector**:

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)


def process_time(df):
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # The `Date Time` column is very useful, but not in this string form. Start by converting it to seconds:
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    # Similar to the wind direction, the time in seconds is not a useful model input. Being weather data,
    # it has clear daily and yearly periodicity. There are many ways you could deal with periodicity.
    # You can get usable signals by using sine and cosine transforms to clear "Time of day" and "Time of year" signals:

    day = 24 * 60 * 60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day) - 0.25)
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day) - 0.25)
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


class JenaPreprocessor():

    def preprocess(self):
        df = load_dataset()
        # Slice [start:stop:step], starting from index 5 take every 6th record.
        df = df[5::6]
        preprocess_wind(df)
        process_time(df)
        # ### Split the data

        # You'll use a `(70%, 20%, 10%)` split for the training, validation, and test sets. Note the data is **not** being randomly shuffled before splitting. This is for two reasons:
        #
        # 1. It ensures that chopping the data into windows of consecutive samples is still possible.
        # 2. It ensures that the validation/test results are more realistic, being evaluated on the data collected after the model was trained.

        # + id="ia-MPAHxbInX"

        n = len(df)
        train_df = df[0:int(n * 0.7)]
        val_df = df[int(n * 0.7):int(n * 0.9)]
        test_df = df[int(n * 0.9):]


        # ### Normalize the data
        #
        # It is important to scale features before training a neural network. Normalization is a common way of doing this scaling:
        # subtract the mean and divide by the standard deviation of each feature.

        # The mean and standard deviation should only be computed using the training data so that the models have no access to the
        # values in the validation and test sets.
        #
        # It's also arguable that the model shouldn't have access to future values in the training set when training, and that this
        # normalization should be done using moving averages. That's not the focus of this tutorial, and the validation and test sets
        # ensure that you get (somewhat) honest metrics. So, in the interest of simplicity this tutorial uses a simple average.

        train_mean = train_df.mean()
        train_std = train_df.std()

        self.train_df = (train_df - train_mean) / train_std
        self.val_df = (val_df - train_mean) / train_std
        self.test_df = (test_df - train_mean) / train_std

        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        self.num_features = df.shape[1]
