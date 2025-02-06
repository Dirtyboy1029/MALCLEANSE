""" This script is for building model graph"""

import tensorflow as tf

from .tools import utils
from .config import logging, ErrorHandler

logger = logging.getLogger('core.ensemble.model_lib')
logger.addHandler(ErrorHandler)


def model_builder(architecture_type='dnn'):
    assert architecture_type in model_name_type_dict, 'models are {}'.format(','.join(model_name_type_dict.keys()))
    return model_name_type_dict[architecture_type]


def _change_scaler_to_list(scaler):
    if not isinstance(scaler, (list, tuple)):
        return [scaler]
    else:
        return scaler


def _dnn_graph(input_dim=None, use_mc_dropout=False):
    """
    The deep neural network based malware detector.
    The implement is based on the paper, entitled ``Adversarial Examples for Malware Detection'',
    which can be found here:  http://patrickmcdaniel.org/pubs/esorics17.pdf

    We slightly change the model architecture by reducing the number of neurons at the last layer to one.
    """
    input_dim = _change_scaler_to_list(input_dim)
    from .model_hp import dnn_hparam
    logger.info(dict(dnn_hparam._asdict()))

    def wrapper(func):
        def graph():
            Dense, _1, _2, _3 = func()
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(input_dim[0],)))
            for units in dnn_hparam.hidden_units:
                model.add(Dense(units, activation=dnn_hparam.activation))
            # model.add(Dense(200, activation=dnn_hparam.activation, name="target_dense_1"))
            # model.add(Dense(200, activation=dnn_hparam.activation, name="target_dense_2"))
            if use_mc_dropout:
                model.add(tf.keras.layers.Dense(dnn_hparam.output_dim, activation=tf.nn.sigmoid))
            else:
                model.add(tf.keras.layers.Dropout(dnn_hparam.dropout_rate))
                model.add(Dense(dnn_hparam.output_dim, activation=tf.nn.sigmoid))
            return model

        return graph

    return wrapper




model_name_type_dict = {
    'dnn': _dnn_graph
}

def build_models(input_x, architecture_type, ensemble_type='vanilla', input_dim=None, use_mc_dropout=False):
    builder = model_builder(architecture_type)

    @builder(input_dim, use_mc_dropout)
    def graph():
        return utils.produce_layer(ensemble_type, dropout_rate=0.4)

    model = graph()
    return model(input_x)
