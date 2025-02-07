# -*- coding: utf-8 -*- 
# @Time : 2024/9/13 18:28 
# @Author : DirtyBoy 
# @File : train_robust_model.py
import tensorflow as tf
from core.data_preprocessing import data_preprocessing
import argparse, os
import numpy as np
from sklearn.model_selection import KFold
from core.ensemble.vanilla import Vanilla
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.deep_ensemble import DeepEnsemble, WeightedDeepEnsemble


def build_dataset_from_numerical_data(data, batch_size=8):
    return tf.data.Dataset.from_tensor_slices(data). \
        cache(). \
        batch(batch_size). \
        shuffle(True). \
        prefetch(tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str, default='thr_1_18')
    args = parser.parse_args()
    noise_type = args.noise_type

    _, index, ratio = noise_type.split('_')

    noise_type = f'mwo_thr_{index}_{ratio}'
    dataset, gt_labels, noise_labels, input_dim, x_train, data_filenames = data_preprocessing(
        noise_type=noise_type, feature_type='data')

    model = Vanilla(architecture_type='dnn',
                    model_directory='../Model/' + noise_type)

    model.fit(train_set=dataset, validation_set=dataset,
              input_dim=10000,
              EPOCH=30,
              training_predict=False)

    noise_type = f'robust_thr_{index}_{ratio}'
    dataset, _, _, _, _, _ = data_preprocessing(
        noise_type=noise_type, feature_type='data')

    model = Vanilla(architecture_type='dnn',
                    model_directory='../Model/' + noise_type)

    model.fit(train_set=dataset, validation_set=dataset,
              input_dim=10000,
              EPOCH=30,
              training_predict=False)
