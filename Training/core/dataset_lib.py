""" This script is for building dataset """

import tensorflow as tf
from .model_hp import train_hparam


def build_dataset_from_numerical_data(data, batch_size=None):
    """
    serialize the data to accommodate the format of model input
    :param data, tuple or np.ndarray
    :param batch_size, scalar or none, the train paramemeter is default if none provided
    """
    batch_size = train_hparam.batch_size if batch_size is None else batch_size
    return tf.data.Dataset.from_tensor_slices(data). \
        cache(). \
        batch(batch_size). \
        prefetch(tf.data.experimental.AUTOTUNE)


def build_dataset_via_generator(generator, y=None, path='', batch_size=8):
    if y is not None:
        return tf.data.Dataset.from_generator(generator,
                                              output_types=(tf.int32, tf.int32),
                                              output_shapes=(tf.TensorShape([None]), tf.TensorShape([]))
                                              ). \
            padded_batch(batch_size, padded_shapes=([None], [])). \
            cache(path). \
            shuffle(buffer_size=100). \
            prefetch(tf.data.experimental.AUTOTUNE)
    else:
        return tf.data.Dataset.from_generator(generator,
                                              output_types=tf.int32,
                                              output_shapes=tf.TensorShape([None])
                                              ). \
            padded_batch(batch_size, padded_shapes=([None])). \
            cache(path). \
            prefetch(tf.data.experimental.AUTOTUNE)


