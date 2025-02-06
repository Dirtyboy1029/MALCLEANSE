# -*- coding: utf-8 -*- 
# @Time : 2024/1/28 13:21 
# @Author : DirtyBoy 
# @File : train_noise_model.py
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


def save_logger(base_path, vanilla_prob, type, fold=0):
    np.save(os.path.join(base_path, type + '/prob/fold' + str(fold + 1)), vanilla_prob)


def main_deepdrebin(model_type, noise_type, EPOCH=30, feature_type='data'):
    if model_type == 'vanilla':
        model_architecture = Vanilla
    elif model_type == 'bayesian':
        model_architecture = BayesianEnsemble
    elif model_type == 'mcdropout':
        model_architecture = MCDropout
    elif model_type == 'deepensemble':
        model_architecture = DeepEnsemble
    else:
        model_architecture = None
    output_path = 'output/' + noise_type

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if not os.path.isdir(os.path.join(output_path, model_type + '/prob')):
        os.makedirs(os.path.join(output_path, model_type + '/prob'))

    if os.path.isfile(os.path.join(output_path, model_type + '/prob/fold5.npy')):
        print('********************************************************************************')
        print('cross val training detail: ' + model_type + ' ' + noise_type)
        print('training finish! ... file exist!!')
        print('********************************************************************************')
    else:
        dataset, gt_labels, noise_labels, input_dim, x_train, data_filenames = data_preprocessing(
            noise_type=noise_type, feature_type=feature_type)
        if os.path.isfile(os.path.join(output_path, 'index/fold5.npy')):
            for fold in range(5):
                if os.path.isfile(os.path.join(output_path, model_type + '/prob/fold' + str(fold + 1) + '.npy')):
                    print('********************************************************************************')
                    print('cross val training detail: ' + model_type + ' ' + noise_type)
                    print('*********No.' + str(fold + 1) + ' fold is exist ********')
                    print('********************************************************************************')
                else:
                    print('********************************************************************************')
                    print('cross val training detail: ' + model_type + ' ' + noise_type)
                    print('*********processing  No.' + str(fold + 1) + ' fold training********')
                    print('********************************************************************************')
                    print('load index from ', os.path.join(output_path, 'index/fold' + str(fold + 1) + '.npy'))
                    train_index, test_index = np.load(os.path.join(output_path, 'index/fold' + str(fold + 1) + '.npy'),
                                                      allow_pickle=True)
                    test_data = x_train[test_index]
                    train_set = build_dataset_from_numerical_data((x_train[train_index], noise_labels[train_index]))
                    validation_set = build_dataset_from_numerical_data((x_train[test_index], gt_labels[test_index]))

                    model = model_architecture(architecture_type='dnn',
                                               model_directory='../Model/' + noise_type + '/fold' + str(fold + 1))
                    model_prob, _ = model.fit(train_set=train_set, validation_set=validation_set,
                                              input_dim=input_dim,
                                              EPOCH=EPOCH,
                                              test_data=test_data,
                                              training_predict=True)
                    save_logger(output_path, model_prob, type=model_type, fold=fold)
                    del model_prob
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=123)

            for fold, (train_index, test_index) in enumerate(kf.split(x_train)):
                print('********************************************************************************')
                print('cross val training detail: ' + model_type + ' ' + noise_type)
                print('*********processing  No.' + str(fold + 1) + ' fold training********')
                print('********************************************************************************')
                if not os.path.isdir(os.path.join(output_path, 'index')):
                    os.makedirs(os.path.join(output_path, 'index'))
                np.save(os.path.join(output_path, 'index/fold' + str(fold + 1)), [train_index, test_index])

                test_data = x_train[test_index]
                train_set = build_dataset_from_numerical_data((x_train[train_index], noise_labels[train_index]))
                validation_set = build_dataset_from_numerical_data((x_train[test_index], gt_labels[test_index]))

                model = model_architecture(architecture_type='dnn',
                                           model_directory='../Model/' + noise_type + '/fold' + str(fold + 1))
                model_prob, _ = model.fit(train_set=train_set, validation_set=validation_set,
                                          input_dim=input_dim,
                                          EPOCH=EPOCH,
                                          test_data=test_data,
                                          training_predict=True)
                save_logger(output_path, model_prob, type=model_type, fold=fold)
                del model_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', '-mt', type=str, default="vanilla")
    parser.add_argument('-noise_type', '-nt', type=str, default='thr_1_18')
    args = parser.parse_args()
    model_type = args.model_type
    noise_type = args.noise_type
    main_deepdrebin(model_type, noise_type)

    ##CUDA_VISIBLE_DEVICES=0 python3 train_noise_model.py -mt vanilla -nt thr_1_10
