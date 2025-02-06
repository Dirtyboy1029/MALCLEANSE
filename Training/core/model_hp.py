from collections import namedtuple

_TRAIN_HP_TEMPLATE = namedtuple('training',
                                ['random_seed', 'n_epochs', 'batch_size', 'learning_rate', 'clipvalue', 'interval'])
train_hparam = _TRAIN_HP_TEMPLATE(random_seed=23456,
                                  n_epochs=30,
                                  batch_size=8,
                                  learning_rate=0.0001,
                                  clipvalue=100.,
                                  interval=2  # saving model weights
                                  )

_MC_DROPOUT_HP_TEMPLATE = namedtuple('mc_dropout', ['dropout_rate', 'n_sampling'])
mc_dropout_hparam = _MC_DROPOUT_HP_TEMPLATE(dropout_rate=0.4,
                                            n_sampling=10
                                            )
_BAYESIAN_HP_TEMPLATE = namedtuple('bayesian', ['n_sampling'])
bayesian_ensemble_hparam = _BAYESIAN_HP_TEMPLATE(n_sampling=10)

_DNN_HP_TEMPLATE = namedtuple('DNN',
                              ['hidden_units', 'dropout_rate', 'activation', 'output_dim'])

dnn_hparam = _DNN_HP_TEMPLATE(hidden_units=[200,200],
                              # DNN has two hidden layers with each having 200 neurons
                              dropout_rate=0.4,
                              activation='relu',
                              output_dim=1  # binary classification#
                              )

_DROIDETEC_HP_TEMPLATE = namedtuple('droidetec',
                                    ['vocab_size', 'n_embedding_dim', 'lstm_units', 'hidden_units',
                                     'dropout_rate', 'max_sequence_length', 'output_dim'])



