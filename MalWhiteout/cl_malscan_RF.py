from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
import csv, os
from itertools import islice
import argparse
import time


def feature_extraction(file):
    vectors = []
    labels = []
    with open(file, 'r') as f:
        csv_data = csv.reader(f)
        for line in islice(csv_data, 1, None):
            vector = [float(i) for i in line[1:-1]]
            label = int(float(line[-1]))
            vectors.append(vector)
            labels.append(label)
    return vectors, labels


def degree_centrality_feature(feature_dir, noise_type):
    feature_csv = os.path.join(feature_dir,
                               data_type + '_' + noise_type + '_' + str(noise_hyper) + '_degree_malscan_features.csv')
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels


def katz_centrality_feature(feature_dir, noise_type):
    feature_csv = os.path.join(feature_dir,
                               noise_type + '_katz_malscan_features.csv')
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels


def closeness_centrality_feature(feature_dir, data_type, noise_type, noise_hyper):
    feature_csv = os.path.join(feature_dir, data_type + '_' + noise_type + '_' + str(
        noise_hyper) + '_closeness_malscan_features.csv')
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels


def harmonic_centrality_feature(feature_dir, data_type, noise_type, noise_hyper):
    feature_csv = os.path.join(feature_dir,
                               data_type + '_' + noise_type + '_' + str(noise_hyper) + '_harmonic_malscan_features.csv')
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels


def random_features(vectors, labels):
    Vec_Lab = []
    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]


def predict_noise(vectors, labels, noise_type, feature_type):
    x_train = np.array(vectors)
    y_train = np.array(labels)
    print(x_train.shape, y_train.shape)

    clf = LearningWithNoisyLabels(RandomForestClassifier(n_estimators=200))

    clf.fit(x_train, y_train)

    psx = estimate_cv_predicted_probabilities(
        X=x_train,
        labels=y_train,
        clf=clf.clf,
        cv_n_folds=clf.cv_n_folds,
        seed=clf.seed,
    )
    np.save('output/psx_malscan_' + feature_type + '_' + noise_type, psx)

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    np.save(
        'output/malscan_labelerrorsmask_' + feature_type + '_' + noise_type,
        label_errors_mask)


def main(feature_type, feature_dir, noise_type):
    if feature_dir[-1] == '/':
        feature_dir = feature_dir
    else:
        feature_dir += '/'

    if feature_type == 'degree':
        degree_vectors, degree_labels = degree_centrality_feature(feature_dir, noise_type)
        predict_noise(degree_vectors, degree_labels, noise_type, feature_type)

    # elif feature_type == 'harmonic':
    #     harmonic_vectors, harmonic_labels = harmonic_centrality_feature(feature_dir, data_type, noise_type, noise_hyper)
    #     # classification(harmonic_vectors, harmonic_labels, 35)
    #     predict_noise(harmonic_vectors, harmonic_labels, data_type, noise_type, noise_hyper, feature_type)
    elif feature_type == 'katz':
        katz_vectors, katz_labels = katz_centrality_feature(feature_dir, noise_type)
        predict_noise(katz_vectors, katz_labels, noise_type, feature_type)

    # elif feature_type == 'closeness':
    #     closeness_vectors, closeness_labels = closeness_centrality_feature(feature_dir, data_type, noise_type,
    #                                                                        noise_hyper)
    #     predict_noise(closeness_vectors, closeness_labels, data_type, noise_type, noise_hyper, feature_type)

    else:
        print('Error Centrality Type!')


if __name__ == '__main__':
    feature_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/malwhiteout_naive_pool/malscan_csv'

    for feature_type in ['katz', ]:

        for i in range(3, 4):

            for noise_type in [str(i) + '_18', str(i) + '_20', str(i) + '_22',
                               str(i) + '_25', str(i) + '_26', str(i) + '_28']:
                Noise_type = 'thr_' + noise_type
                start = time.time()
                main(feature_type, feature_dir, Noise_type)
                end = time.time()
                print(str(Noise_type) + ' use time:', end - start)

            for noise_type in [str(i) + '_10', str(i) + '_15', str(i) + '_18', str(i) + '_20']:
                Noise_type = 'thr_variation_' + noise_type
                start = time.time()
                main(feature_type, feature_dir, Noise_type)
                end = time.time()
                print(str(Noise_type) + ' use time:', end - start)
