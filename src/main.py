import pandas as pd
import tensorflow as tf

TIME_STEPS = 240
BATCH_SIZE = 128

def preprocess(data):
    data.set_index("datetime", inplace=True)

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.drop(columns=["Global_intensity"], axis=1)

    return data


def timeseries_dataset(data, timesteps=TIME_STEPS, batch_size=BATCH_SIZE):
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(timesteps, shift=60, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(timesteps))

    ds = ds.map(lambda x: (x, x))  # For autoencoder : y = X
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
