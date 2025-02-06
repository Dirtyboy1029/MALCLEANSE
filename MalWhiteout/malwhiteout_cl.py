# -*- coding: utf-8 -*- 
# @Time : 2024/6/2 11:05 
# @Author : DirtyBoy 
# @File : malwhiteout_cl.py
import os
import numpy as np
import cleanlab as clb
from collections import Counter
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def main_(noise_type_source):
    ENDING = []
    for i in range(1, 4):
        a, b = noise_type_source.rsplit('_', 1)
        noise_type = a + '_' + str(i) + '_' + b
        ending = []
        data_filenames, gt_labels, noise_labels = read_joblib(
            '../Training/config/' + 'databases_' + str(noise_type) + '.conf')
        noise_benign_index = np.where(noise_labels == 0)[0]
        gt_labels = np.array(gt_labels)
        noise_labels = np.array(noise_labels)
        is_noise = []
        for i in range(len(gt_labels)):
            if gt_labels[i] == noise_labels[i]:
                is_noise.append(0)
            else:
                is_noise.append(1)
        is_noise = np.array(is_noise)
        MSG = "Before Noise accurce is {:.2f}%."
        if i == 1:
            print(np.sum(is_noise))
            print(MSG.format(accuracy_score(gt_labels, noise_labels) * 100))
        drebin_psx = np.load('output/psx_drebin_' + str(noise_type) + '.npy')
        csbd_psx = np.load('output/psx_csbdRF_' + str(noise_type) + '.npy')
        malscan_psx = np.load(
            'output/psx_malscan_katz_' + noise_type + '.npy')
        combined_psx = np.concatenate((drebin_psx, malscan_psx), axis=1)
        combined_psx = np.concatenate((combined_psx, csbd_psx), axis=1)
        lnl = LearningWithNoisyLabels(clf=LogisticRegression())
        lnl.fit(X=combined_psx, s=noise_labels)
        psx = estimate_cv_predicted_probabilities(
            X=combined_psx,
            labels=noise_labels,
            clf=lnl.clf,
            cv_n_folds=lnl.cv_n_folds,
            seed=lnl.seed,
        )
        np.save('mwo_output/' + 'mwo_psx_' + noise_type, psx)
        ordered_label_errors = clb.pruning.get_noise_indices(s=noise_labels,
                                                             psx=psx,
                                                             prune_method='prune_by_noise_rate')

        np.save('mwo_output/' + 'mwo_label_mask_' + noise_type, ordered_label_errors[noise_benign_index].astype(int))
        modify_label = list(noise_labels)
        for i, is_error in enumerate(ordered_label_errors):
            if is_error and noise_labels[i] == 0:
                modify_label[i] = 1
        modify_accuracy = accuracy_score(gt_labels, modify_label)
        recall = recall_score(is_noise[noise_benign_index], ordered_label_errors[noise_benign_index].astype(int),
                              average='binary')
        prescion = precision_score(is_noise[noise_benign_index], ordered_label_errors[noise_benign_index].astype(int),
                                   average='binary')
        f1 = f1_score(is_noise[noise_benign_index], ordered_label_errors[noise_benign_index].astype(int),
                      average='binary')

        ending.append(modify_accuracy)
        ending.append(recall)
        ending.append(prescion)
        ending.append(f1)
        ENDING.append(ending)
        print(ending)
    return ENDING


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str)
    args = parser.parse_args()
    noise_type_source = args.noise_type

    end = np.mean(main_(noise_type_source), axis=0)
    print(end)
    MSG = "Noise Benign accurce is {:.2f}%."
    print(MSG.format(end[0] * 100))
    # print("The dataset denoise on the dataset is ", end[0])

    MSG = "Recall is {:.2f}%"
    print(MSG.format(end[1] * 100))
    MSG = "Prescion is {:.2f}%"
    print(MSG.format(end[2] * 100))
    MSG = "F1 score is {:.2f}%"
    print(MSG.format(end[3] * 100))
