from os import path
import time

import numpy as np
import tensorflow as tf

from .vanilla import Vanilla
from ..model_hp import train_hparam
from ..dataset_lib import build_dataset_from_numerical_data
from ..tools import utils
from ..config import logging, ErrorHandler

logger = logging.getLogger('ensemble.deep_ensemble')
logger.addHandler(ErrorHandler)


class DeepEnsemble(Vanilla):
    def __init__(self,
                 architecture_type='dnn',
                 base_model=None,
                 n_members=10,
                 model_directory=None,
                 name='DEEPENSEMBLE'
                 ):
        super(DeepEnsemble, self).__init__(architecture_type,
                                           base_model,
                                           n_members,
                                           model_directory,
                                           name)
        self.hparam = train_hparam
        self.ensemble_type = 'deep_ensemble'


class WeightedDeepEnsemble(Vanilla):
    def __init__(self,
                 architecture_type='dnn',
                 base_model=None,
                 n_members=10,
                 model_directory=None,
                 name='WEIGTHEDDEEPENSEMBLE'
                 ):
        super(WeightedDeepEnsemble, self).__init__(architecture_type,
                                                   base_model,
                                                   n_members,
                                                   model_directory,
                                                   name)
        self.hparam = train_hparam
        self.ensemble_type = 'deep_ensemble'
        self.weight_modular = None

    def get_weight_modular(self):
        class Simplex(tf.keras.constraints.Constraint):
            def __call__(self, w):
                return tf.math.softmax(w - tf.math.reduce_max(w), axis=0)

        inputs = tf.keras.Input(shape=(self.n_members,))
        outs = tf.keras.layers.Dense(1, use_bias=False, activation=None, kernel_constraint=Simplex(), name='simplex')(
            inputs)
        return tf.keras.Model(inputs=inputs, outputs=outs)

    def predict(self, x, use_prob=False):
        """ conduct prediction """
        self.base_model = None
        self.weight_modular = None
        self.weights_list = []
        self._optimizers_dict = []
        self.load_ensemble_weights()
        output_list = []
        start_time = time.time()
        for base_model in self.model_generator():
            if isinstance(x, tf.data.Dataset):
                output_list.append(base_model.predict(x, verbose=1))
            elif isinstance(x, (np.ndarray, list)):
                output_list.append(base_model.predict(x, batch_size=self.hparam.batch_size, verbose=1))
            else:
                raise ValueError
        total_time = time.time() - start_time
        logger.info('Inference costs {} seconds.'.format(total_time))
        assert self.weight_modular is not None
        output = self.weight_modular(np.hstack(output_list)).numpy()
        if not use_prob:
            return np.stack(output_list, axis=1), self.weight_modular.get_layer('simplex').get_weights()
        else:
            return output

    def predict_in_training(self, x, use_prob=False):
        output_list = []
        start_time = time.time()
        for base_model in self.model_generator():
            if isinstance(x, tf.data.Dataset):
                output_list.append(base_model.predict(x, verbose=1))
            elif isinstance(x, (np.ndarray, list)):
                output_list.append(base_model.predict(x, batch_size=self.hparam.batch_size, verbose=1))
            else:
                raise ValueError
        total_time = time.time() - start_time
        logger.info('Inference costs {} seconds.'.format(total_time))
        assert self.weight_modular is not None
        output = self.weight_modular(np.hstack(output_list)).numpy()
        if not use_prob:
            return np.stack(output_list, axis=1), self.weight_modular.get_layer('simplex').get_weights()
        else:
            return output

    def fit(self, train_set, validation_set=None, input_dim=None, EPOCH=30, test_data=None, training_predict=True,
            **kwargs):
        """
        fit the ensemble by producing a lists of model weights
        :param train_set: tf.data.Dataset, the type shall accommodate to the input format of Tensorflow models
        :param validation_set: validation data, optional
        :param input_dim: integer or list, input dimension except for the batch size
        """
        # training preparation
        prob = []
        if self.base_model is None:
            self.build_model(input_dim=input_dim)
        if self.weight_modular is None:
            self.weight_modular = self.get_weight_modular()

        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate,
                                               clipvalue=self.hparam.clipvalue),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

        self.weight_modular.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate,
                                               clipvalue=self.hparam.clipvalue),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

        # training
        logger.info("hyper-parameters:")
        logger.info(dict(self.hparam._asdict()))
        logger.info("...training start!")

        best_val_accuracy = 0.
        total_time = 0.
        for epoch in range(EPOCH):
            for member_idx in range(self.n_members):
                if member_idx < len(self.weights_list):  # loading former weights
                    self.base_model.set_weights(self.weights_list[member_idx])
                    self.base_model.optimizer.set_weights(self._optimizers_dict[member_idx])
                elif member_idx == 0:
                    pass  # do nothing
                else:
                    self.reinitialize_base_model()

                msg = 'Epoch {}/{}, member {}/{}, and {} member(s) in list'.format(epoch + 1,
                                                                                   EPOCH, member_idx + 1,
                                                                                   self.n_members,
                                                                                   len(self.weights_list))
                print(msg)
                start_time = time.time()
                self.base_model.fit(train_set,
                                    epochs=epoch + 1,
                                    initial_epoch=epoch,
                                    validation_data=validation_set
                                    )
                self.update_weights(member_idx,
                                    self.base_model.get_weights(),
                                    self.base_model.optimizer.get_weights())

                end_time = time.time()
                total_time += end_time - start_time
            # training weight modular
            msg = "train the weight modular at epoch {}/{}"
            print(msg.format(epoch + 1, EPOCH))
            start_time = time.time()
            history = self.fit_weight_modular(train_set, validation_set, epoch)
            end_time = time.time()
            total_time += end_time - start_time
            # saving
            logger.info('Training ensemble costs {} in total (including validation).'.format(total_time))
            train_acc = history.history['binary_accuracy'][0]
            val_acc = history.history['val_binary_accuracy'][0]
            msg = 'Epoch {}/{}: training accuracy {:.5f}, validation accuracy {:.5f}.'.format(
                epoch + 1, self.hparam.n_epochs, train_acc, val_acc
            )
            logger.info(msg)
            if test_data is not None and training_predict is True: # and (epoch + 1) % 5 == 0
                prob.append(self.predict_in_training(test_data, use_prob=False))
        training_log = []
        return prob, training_log

    def fit_weight_modular(self, train_set, validation_set, epoch):
        """
        fit weight modular
        :param train_set: training set
        :param validation_set: validation set
        :param epoch: integer, training epoch
        :return: None
        """

        # obtain data
        def get_data(x_y_set):
            tsf_x = []
            tsf_y = []
            for _x, _y in x_y_set:
                _x_list = []
                for base_model in self.model_generator():
                    _x_pred = base_model(_x)
                    _x_list.append(_x_pred)
                tsf_x.append(np.hstack(_x_list))
                tsf_y.append(_y)
            return np.vstack(tsf_x), np.concatenate(tsf_y)

        transform_train_set = build_dataset_from_numerical_data(get_data(train_set))
        transform_val_set = build_dataset_from_numerical_data(get_data(validation_set))

        history = self.weight_modular.fit(transform_train_set,
                                          epochs=epoch + 1,
                                          initial_epoch=epoch,
                                          validation_data=transform_val_set
                                          )
        return history

    def save_ensemble_weights(self):
        if not path.exists(self.save_dir):
            utils.mkdir(self.save_dir)
        # save model configuration
        try:
            config = self.base_model.to_json()
            utils.dump_json(config, path.join(self.save_dir,
                                              self.architecture_type + '.json'))  # lightweight method for saving model configurature
        except Exception as e:
            pass
        finally:
            if not path.exists(path.join(self.save_dir, self.architecture_type)):
                utils.mkdir(path.join(self.save_dir, self.architecture_type))
            self.base_model.save(path.join(self.save_dir, self.architecture_type))
        print("Save the model configuration to directory {}".format(self.save_dir))

        # save model weights
        utils.dump_joblib(self.weights_list, path.join(self.save_dir, self.architecture_type + '.model'))
        utils.dump_joblib(self._optimizers_dict, path.join(self.save_dir, self.architecture_type + '.model.metadata'))
        print("Save the model weights to directory {}".format(self.save_dir))

        # save weight modular
        self.weight_modular.save(path.join(self.save_dir, self.architecture_type + '_weight_modular'))
        print("Save the weight modular weights to directory {}".format(self.save_dir))
        return

    def load_ensemble_weights(self):
        if path.exists(path.join(self.save_dir, self.architecture_type + '.json')):
            config = utils.load_json(path.join(self.save_dir, self.architecture_type + '.json'))
            self.base_model = tf.keras.models.model_from_json(config)
        elif path.exists(path.join(self.save_dir, self.architecture_type)):
            self.base_model = tf.keras.models.load_model(path.join(self.save_dir, self.architecture_type))
        else:
            logger.error("File not found: ".format(path.join(self.save_dir, self.architecture_type + '.json')))
            raise FileNotFoundError
        print("Load model config from {}.".format(self.save_dir))

        if path.exists(path.join(self.save_dir, self.architecture_type + '.model')):
            self.weights_list = utils.read_joblib(path.join(self.save_dir, self.architecture_type + '.model'))
        else:
            logger.error("File not found: ".format(path.join(self.save_dir, self.architecture_type + '.model')))
            raise FileNotFoundError
        print("Load model weights from {}.".format(self.save_dir))

        if path.exists(path.join(self.save_dir, self.architecture_type + '.model.metadata')):
            self._optimizers_dict = utils.read_joblib(
                path.join(self.save_dir, self.architecture_type + '.model.metadata'))
        else:
            self._optimizers_dict = [None] * len(self.weights_list)

        if path.exists(path.join(self.save_dir, self.architecture_type + '_weight_modular')):
            self.weight_modular = tf.keras.models.load_model(
                path.join(self.save_dir, self.architecture_type + '_weight_modular'))
        return
