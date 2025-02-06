# /usr/vin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import psutil
import time
import numpy as np
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities

# logging level
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def data_preprocessing(noise_type, num_features_to_be_selected=5000, feature_option=False):
    android_features_saving_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/malwhiteout_naive_pool/csbd'
    intermediate_data_saving_dir = '/home/lhd/MalCleanse/Training/config'
    data_filenames, gt_labels, noise_labels = read_joblib(os.path.join(
        intermediate_data_saving_dir, 'databases_' + str(noise_type) + '.conf'))
    data_filenames = [item + '.txt' for item in data_filenames]
    oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]
    Logger.info("All Samples loaded")
    # Creating feature vectors
    feature_vectorizer = TF(input='filename', lowercase=False, token_pattern=None,
                            tokenizer=lambda s: s.split(), binary=feature_option)
    x_train = feature_vectorizer.fit_transform(oos_features)
    y_train = noise_labels
    Logger.info("Training Label array - generated")

    # Doing feature selection
    features = feature_vectorizer.get_feature_names_out()
    Logger.info("Total number of features: {} ".format(len(features)))

    if len(features) > num_features_to_be_selected:
        # with feature selection
        Logger.info("Gonna select %s features", num_features_to_be_selected)
        fs_algo = SelectKBest(chi2, k=num_features_to_be_selected)

        x_train = fs_algo.fit_transform(x_train, y_train)
    print(x_train.shape, y_train.shape)
    return x_train, y_train


def predict_noise(x_train, y_train, noise_type):
    clf = LearningWithNoisyLabels(RandomForestClassifier(n_estimators=200))
    clf.fit(x_train, y_train)

    psx = estimate_cv_predicted_probabilities(
        X=x_train,
        labels=y_train,
        clf=clf.clf,
        cv_n_folds=clf.cv_n_folds,
        seed=clf.seed, )

    np.save('output/psx_csbdRF_' + str(noise_type), psx)

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    np.save('output/csbdRF_labelerrorsmask_' + str(noise_type),
            label_errors_mask)


def main(noise_type):
    x_train, y_train = data_preprocessing(noise_type)
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
