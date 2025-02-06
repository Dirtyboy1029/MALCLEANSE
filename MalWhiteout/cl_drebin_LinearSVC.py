# /usr/vin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.model_selection import GridSearchCV
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import logging
import CommonModules as CM
# from joblib import dump, load
# from pprint import pprint
import json, os
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
import sys
# from GetApkData import GetApkData
import psutil, argparse, logging
from scipy import sparse

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def data_preprocessing(noise_type='random',FeatureOption=True):
    android_features_saving_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/malwhiteout_naive_pool/drebin'
    intermediate_data_saving_dir = '/home/lhd/MalCleanse/Training/config'
    data_filenames, gt_labels, noise_labels = read_joblib(os.path.join(
        intermediate_data_saving_dir, 'databases_' + str(noise_type) + '.conf'))
    data_filenames = [item + '.data' for item in data_filenames]
    oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]

    FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption, max_features=10000)
    x_train = FeatureVectorizer.fit_transform(oos_features)
    y_train = noise_labels
    print('load data finish!!')
    print(x_train.shape, y_train.shape)
    return x_train, y_train


def predict_noise(x_train, y_train, noise_type):
    print('cross val....')
    Clf = LearningWithNoisyLabels(SVC(kernel='linear', probability=True))
    Clf.fit(x_train, y_train)
    psx = estimate_cv_predicted_probabilities(
        X=x_train,
        labels=y_train,
        clf=Clf.clf,
        cv_n_folds=Clf.cv_n_folds,
        seed=Clf.seed,
    )

    np.save('output/psx_drebin_' + str(noise_type) , psx)

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    np.save('output/drebinsvm_labelerrorsmask_'  + str(noise_type) ,
            label_errors_mask)


def main(noise_type):
    x_train, y_train = data_preprocessing(noise_type, FeatureOption=True)
    predict_noise(x_train, y_train, noise_type)


if __name__ == "__main__":
    for i in range(1, 4):

        for noise_type in [str(i) + '_10', str(i) + '_15', str(i) + '_18', str(i) + '_20', str(i) + '_22',
                           str(i) + '_25', str(i) + '_26', str(i) + '_28']:
            Noise_type = 'thr_' + noise_type
            print(str(Noise_type))
            start = time.time()
            main(noise_type=Noise_type)
            end = time.time()
            print(str(Noise_type) + ' use time:', end - start)

        for noise_type in [str(i) + '_10', str(i) + '_15', str(i) + '_18', str(i) + '_20']:
            Noise_type = 'thr_variation_' + noise_type
            print(str(Noise_type))
            start = time.time()
            main(noise_type=Noise_type)
            end = time.time()
            print(str(Noise_type) + ' use time:', end - start)


