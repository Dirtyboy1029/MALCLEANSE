import os.path as path
import time

import tensorflow as tf
import numpy as np

from .ensemble import Ensemble
from ..model_hp import train_hparam
from ..model_lib import model_builder
from ..tools import utils
from ..config import logging, ErrorHandler

logger = logging.getLogger('ensemble.vanilla')
logger.addHandler(ErrorHandler)


class Vanilla(Ensemble):
    """ vanilla model, i.e., the so-called ensemble just has a single model """

    def __init__(self, architecture_type='dnn', base_model=None, n_members=1, model_directory=None, name='VANILLA'):
        """
        initialization
        :param architecture_type: the type of base model
        :param base_model: an object of base model
        :param n_members: number of base models
        :param model_directory: a folder for saving ensemble weights
        """
        super(Vanilla, self).__init__(architecture_type, base_model, n_members, model_directory)
        self.hparam = train_hparam
        self.ensemble_type = 'vanilla'
        self.name = name.lower()
        self.save_dir = path.join(self.model_directory, self.name)

    def build_model(self, input_dim=None):
        """
        Build an ensemble model -- only the homogeneous structure is considered
        :param input_dim: integer or list, input dimension shall be set in some cases under eager mode
        """
        callable_graph = model_builder(self.architecture_type)

        @callable_graph(input_dim)
        def _builder():
            return utils.produce_layer(self.ensemble_type)

        self.base_model = _builder()
        return

    def predict(self, x, use_prob=False):
        """ conduct prediction """
        self.base_model = None
        self.weights_list = []
        self._optimizers_dict = []
        self.load_ensemble_weights()
        output_list = []
        start_time = time.time()
        for base_model in self.model_generator():
            if isinstance(x, tf.data.Dataset):
                output_list.append(base_model.predict(x, verbose=1))
            elif isinstance(x, (np.ndarray, list)):
                output_list.append(base_model.predict(x, verbose=1, batch_size=self.hparam.batch_size))
            else:
                raise ValueError
        total_time = time.time() - start_time
        logger.info('Inference costs {} seconds.'.format(total_time))
        if not use_prob:
            return np.stack(output_list, axis=1)
        else:
            return np.mean(np.stack(output_list, axis=1), axis=1)

    def predict_in_training(self, x, use_prob=False):
        output_list = []
        start_time = time.time()
        for base_model in self.model_generator():
            if isinstance(x, tf.data.Dataset):
                output_list.append(base_model.predict(x, verbose=1))
            elif isinstance(x, (np.ndarray, list)):
                output_list.append(base_model.predict(x, verbose=1, batch_size=self.hparam.batch_size))
            else:
                raise ValueError
        total_time = time.time() - start_time
        logger.info('Inference costs {} seconds.'.format(total_time))
        if not use_prob:
            return np.stack(output_list, axis=1)
        else:
            return np.mean(np.stack(output_list, axis=1), axis=1)

    def evaluate(self, x, gt_labels, threshold=0.5, name='test'):
        """
        get some statistical values
        :param x: tf.data.Dataset object
        :param gt_labels: ground truth labels
        :param threshold: float value between 0 and 1, to decide the predicted label
        :return: None
        """
        x_prob = self.predict(x, use_prob=True)
        x_pred = (x_prob >= threshold).astype(np.int32)

        # metrics
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        accuracy = accuracy_score(gt_labels, x_pred)
        b_accuracy = balanced_accuracy_score(gt_labels, x_pred)

        MSG = "The accuracy on the {} dataset is {:.5f}%"
        logger.info(MSG.format(name, accuracy * 100))
        MSG = "The balanced accuracy on the {} dataset is {:.5f}%"
        logger.info(MSG.format(name, b_accuracy * 100))
        is_single_class = False
        if np.all(gt_labels == 1.) or np.all(gt_labels == 0.):
            is_single_class = True
        if not is_single_class:
            tn, fp, fn, tp = confusion_matrix(gt_labels, x_pred).ravel()

            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            f1 = f1_score(gt_labels, x_pred, average='binary')

            print("Other evaluation metrics we may need:")
            MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
            logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))
        return x_prob

    def model_generator(self):
        try:
            if len(self.weights_list) <= 0:
                self.load_ensemble_weights()
        except Exception as e:
            raise Exception("Cannot load model weights:{}.".format(str(e)))

        for i, weights in enumerate(self.weights_list):
            self.base_model.set_weights(weights=weights)
            # if i in self._optimizers_dict and self.base_model.optimizer is not None:
            #     self.base_model.optimizer.set_weights(self._optimizers_dict[i])
            yield self.base_model

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

        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate,
                                               clipvalue=self.hparam.clipvalue),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        # training
        logger.info("hyper-parameters:")
        logger.info(dict(self.hparam._asdict()))
        logger.info("The number of trainable variables: {}".format(len(self.base_model.trainable_variables)))
        logger.info("...training start!")

        best_val_accuracy = 0.
        total_time = 0.

        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []

        for epoch in range(EPOCH):
            train_acc = 0.
            val_acc = 0.
            train_loss = 0.
            val_loss = 0.
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
                history = self.base_model.fit(train_set,
                                              epochs=epoch + 1,
                                              initial_epoch=epoch,
                                              validation_data=validation_set
                                              )
                train_acc += history.history['binary_accuracy'][0]
                val_acc += history.history['val_binary_accuracy'][0]
                train_loss += history.history['loss'][0]
                val_loss += history.history['val_loss'][0]
                self.update_weights(member_idx,
                                    self.base_model.get_weights(),
                                    self.base_model.optimizer.get_weights())
                end_time = time.time()
                total_time += end_time - start_time

            # saving
            logger.info('Training ensemble costs {} seconds in total (including validation).'.format(total_time))
            train_acc = train_acc / self.n_members
            val_acc = val_acc / self.n_members
            train_loss = train_loss / self.n_members
            val_loss = val_loss / self.n_members
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            msg = 'Epoch {}/{}: training accuracy {:.5f}, validation accuracy {:.5f}.'.format(
                epoch + 1, EPOCH, train_acc, val_acc
            )
            logger.info(msg)
            if test_data is not None and training_predict is True: # and (epoch + 1) % 5 == 0
                prob.append(self.predict_in_training(test_data, use_prob=False))

        self.save_ensemble_weights()
        training_log = [train_acc_list, train_loss_list, val_acc_list, val_loss_list]
        return prob, training_log

    def finetune(self, train_set, validation_set=None, input_dim=None, EPOCH=30, test_data=None, training_predict=True,
                 **kwargs):
        """
        fit the ensemble by producing a lists of model weights
        :param train_set: tf.data.Dataset, the type shall accommodate to the input format of Tensorflow models
        :param validation_set: validation data, optional
        :param input_dim: integer or list, input dimension except for the batch size
        """
        # training preparation
        prob = []

        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate,
                                               clipvalue=self.hparam.clipvalue),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        # training
        logger.info("hyper-parameters:")
        logger.info(dict(self.hparam._asdict()))
        logger.info("The number of trainable variables: {}".format(len(self.base_model.trainable_variables)))
        logger.info("...training start!")

        best_val_accuracy = 0.
        total_time = 0.

        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []

        for epoch in range(EPOCH):
            train_acc = 0.
            val_acc = 0.
            train_loss = 0.
            val_loss = 0.
            for member_idx in range(self.n_members):
                msg = 'Epoch {}/{}, member {}/{}, and {} member(s) in list'.format(epoch + 1,
                                                                                   EPOCH, member_idx + 1,
                                                                                   self.n_members,
                                                                                   len(self.weights_list))
                print(msg)
                start_time = time.time()
                history = self.base_model.fit(train_set,
                                              epochs=epoch + 1,
                                              initial_epoch=epoch,
                                              validation_data=validation_set
                                              )
                train_acc += history.history['binary_accuracy'][0]
                val_acc += history.history['val_binary_accuracy'][0]
                train_loss += history.history['loss'][0]
                val_loss += history.history['val_loss'][0]
                self.update_weights(member_idx,
                                    self.base_model.get_weights(),
                                    self.base_model.optimizer.get_weights())
                end_time = time.time()
                total_time += end_time - start_time

            # saving
            logger.info('Training ensemble costs {} seconds in total (including validation).'.format(total_time))
            train_acc = train_acc / self.n_members
            val_acc = val_acc / self.n_members
            train_loss = train_loss / self.n_members
            val_loss = val_loss / self.n_members
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            msg = 'Epoch {}/{}: training accuracy {:.5f}, validation accuracy {:.5f}.'.format(
                epoch + 1, EPOCH, train_acc, val_acc
            )
            logger.info(msg)
            if test_data is not None and training_predict is True:
                prob.append(self.predict_in_training(test_data, use_prob=True))

        training_log = [train_acc_list, train_loss_list, val_acc_list, val_loss_list]

        return prob, training_log

    def update_weights(self, member_idx, model_weights, optimizer_weights=None):
        if member_idx < len(self.weights_list):
            self.weights_list[member_idx] = model_weights
            self._optimizers_dict[member_idx] = optimizer_weights
        else:
            # append the weights at the rear of list
            assert len(self.weights_list) == len(self._optimizers_dict)
            self.weights_list.append(model_weights)
            set_idx = len(self.weights_list) - 1
            self._optimizers_dict[set_idx] = optimizer_weights
        return

    def save_ensemble_weights(self):
        # if not path.exists(self.save_dir):
        #     utils.mkdir(self.save_dir)
        # # save model configuration
        # try:
        #     config = self.base_model.to_json()
        #     utils.dump_json(config, path.join(self.save_dir,
        #                                       self.architecture_type + '.json'))  # lightweight method for saving model configurature
        # except Exception as e:
        #     pass
        # finally:
        if not path.exists(path.join(self.save_dir, self.architecture_type)):
            utils.mkdir(path.join(self.save_dir, self.architecture_type))
        self.base_model.save(path.join(self.save_dir, self.architecture_type))
        print("Save the model configuration to directory {}".format(self.save_dir))

        # save model weights
        utils.dump_joblib(self.weights_list, path.join(self.save_dir, self.architecture_type + '.model'))
        utils.dump_joblib(self._optimizers_dict, path.join(self.save_dir, self.architecture_type + '.model.metadata'))
        print("Save the model weights to directory {}".format(self.save_dir))
        return

    def load_ensemble_weights(self):
        # if path.exists(path.join(self.save_dir, self.architecture_type + '.json')):
        #     config = utils.load_json(path.join(self.save_dir, self.architecture_type + '.json'))
        #     self.base_model = tf.keras.models.model_from_json(config)
        if path.exists(path.join(self.save_dir, self.architecture_type)):
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
        return

    def get_n_members(self):
        return len(self.weights_list)

    def reinitialize_base_model(self):
        new_weights = []
        for w in self.base_model.weights:
            if w.trainable:
                if '/kernel' in w.name:  # default name
                    new_w = utils.glorot_uniform(w.numpy().shape)
                elif '/recurrent_kernel' in w.name:
                    initilizer = tf.keras.initializers.Orthogonal()
                    new_w = initilizer(w.numpy().shape).numpy()
                elif '/bias' in w.name:
                    new_w = utils.glorot_uniform(w.numpy().shape)
                else:
                    new_w = utils.glorot_uniform(w.numpy().shape)
            else:
                new_w = w.numpy()
            new_weights.append(new_w)
        self.base_model.set_weights(new_weights)
        return

    def gradient_loss_wrt_input(self, x, y=None):
        if self.base_model is None:
            raise ValueError("A learned model is expected. Please try load_ensemble_weights() first")

        # we set y[...]=1 by default
        y = np.ones(shape=x.shape[0], dtype=np.int64)
        binary_ce = tf.losses.binary_crossentropy
        grad = 0.
        for model_fn in self.model_generator():
            with tf.GradientTape() as g:
                g.watch(x)
                loss = binary_ce(y, model_fn(x))
            grad += g.gradient(loss, x)
        return grad
