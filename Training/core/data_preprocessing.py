# conduct the group of 'out of distribution' experiments on drebin dataset
import os
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd
from .feature import feature_type_scope_dict
from .tools import utils
from .config import config, logging
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from .dataset_lib import build_dataset_from_numerical_data
from sklearn.feature_selection import SelectKBest, chi2


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def data_preprocessing(noise_type='thr_1_18', feature_type='drebin', model_type='thr_1_18'):
    intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
    save_path = os.path.join(intermediate_data_saving_dir,
                             'databases_' + str(noise_type) + '.conf')

    print('load filename and label from ' + save_path)
    data_filenames, gt_labels, noise_labels = utils.read_joblib(save_path)
    if feature_type == 'data':
        android_features_saving_dir = config.get('metadata', 'mwo_data_pool') + 'drebin'
        print('naive data pool:', android_features_saving_dir)
        oos_features = [os.path.join(android_features_saving_dir, filename) + '.data' for filename in data_filenames]
        FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                               binary=True, max_features=10000)

        x_train = FeatureVectorizer.fit_transform(oos_features)
        vocabulary = FeatureVectorizer.get_feature_names_out()
        save_to_txt(vocabulary, noise_type + '.vocab')
        dataX_np = x_train.toarray()

        input_dim = dataX_np.shape[1]
        dataset = build_dataset_from_numerical_data((dataX_np, noise_labels))
    elif feature_type == 'drebin':
        android_features_saving_dir = config.get('metadata', 'naive_data_pool')
        intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
        feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                                  intermediate_data_saving_dir,
                                                                  update=False,
                                                                  proc_number=8)
        oos_features = [os.path.join(android_features_saving_dir, filename) + '.drebin' for filename in data_filenames]
        dataset, input_dim, dataX_np = feature_extractor.feature2ipt(oos_features, noise_labels, noise_type=noise_type)
        input_dim = 10000

    elif feature_type == 'test_data':
        android_features_saving_dir = config.get('metadata', 'mwo_data_pool') + 'drebin'
        print('naive data pool:', android_features_saving_dir)
        oos_features = [os.path.join(android_features_saving_dir, filename) + '.data' for filename in
                        data_filenames]
        vocabulary = txt_to_list(model_type + '.vocab')
        FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                               binary=True, max_features=10000, vocabulary=vocabulary)

        x_train = FeatureVectorizer.fit_transform(oos_features)
        dataX_np = x_train.toarray()

        input_dim = dataX_np.shape[1]
        input_dim = dataX_np.shape[1]
        dataset = build_dataset_from_numerical_data((dataX_np, noise_labels))
    else:
        dataset = None
        dataX_np = None
        input_dim = None
    return dataset, gt_labels, noise_labels, input_dim, dataX_np, data_filenames
