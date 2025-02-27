import time

import tensorflow as tf

from .vanilla import model_builder
from .mc_dropout import MCDropout
from ..model_hp import train_hparam, bayesian_ensemble_hparam
from ..config import logging, ErrorHandler
from ..tools import utils

logger = logging.getLogger('ensemble.bayesian_ensemble')
logger.addHandler(ErrorHandler)


class BayesianEnsemble(MCDropout):
    def __init__(self,
                 architecture_type='dnn',
                 base_model=None,
                 n_members=1,
                 model_directory=None,
                 name='BAYESIAN_ENSEMBLE'
                 ):
        super(BayesianEnsemble, self).__init__(architecture_type,
                                               base_model,
                                               n_members,
                                               model_directory,
                                               name)
        self.hparam = utils.merge_namedtuples(train_hparam, bayesian_ensemble_hparam)
        self.ensemble_type = 'bayesian'

    def build_model(self, input_dim=None, scaler=1. / 10000):
        """
        Build an ensemble model -- only the homogeneous structure is considered
        :param input_dim: integer or list, input dimension shall be set in some cases under eager mode
        :param scaler: float value in the rage of [0, 1], weighted kl divergence
        """
        callable_graph = model_builder(self.architecture_type)

        @callable_graph(input_dim)
        def _builder():
            return utils.produce_layer(self.ensemble_type, kl_scaler=scaler)

        self.base_model = _builder()
        return

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
        prob_mean = []
        if self.base_model is None:
            # scaler = 1. / (len(list(train_set)) * self.hparam.batch_size)  # time-consuming
            scaler = 1. / 50000.
            self.build_model(input_dim=input_dim, scaler=scaler)

        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate,
                                               clipvalue=self.hparam.clipvalue),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            experimental_run_tf_function=False
        )

        # training
        logger.info("hyper-parameters:")
        logger.info(dict(self.hparam._asdict()))
        logger.info("...training start!")

        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []

        best_val_accuracy = 0.
        total_time = 0.
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
                                                                                   EPOCH,
                                                                                   member_idx + 1,
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
                val_acc += history.history['binary_accuracy'][0]
                self.update_weights(member_idx,
                                    self.base_model.get_weights(),
                                    self.base_model.optimizer.get_weights())
                end_time = time.time()
                total_time += end_time - start_time
            # saving
            logger.info('Training ensemble costs {} in total (including validation).'.format(total_time))
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
            # if (epoch + 1) % self.hparam.interval == 0:
            #     if val_acc >= best_val_accuracy:
            #         self.save_ensemble_weights()
            #         best_val_accuracy = val_acc
            #         msg = '\t The best validation accuracy is {:.5f}, obtained at epoch {}/{}'.format(
            #             best_val_accuracy, epoch + 1, EPOCH
            #         )
            #         logger.info(msg)
            if test_data is not None and training_predict is True: # and (epoch + 1) % 5 == 0
                prob.append(self.predict_in_training(test_data))
            # prob_mean.append(self.predict_in_training(test_data, use_prob=True))

        training_log = [train_acc_list, train_loss_list, val_acc_list, val_loss_list]
        return prob, training_log
