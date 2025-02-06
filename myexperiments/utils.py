# -*- coding: utf-8 -*- 
# @Time : 2024/5/6 13:53 
# @Author : DirtyBoy 
# @File : utils.py
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import os


def get_confident_joint_index(noise_labels, prob):
    pred = (prob > 0.5).astype(int)
    noise_0_pred_0 = np.where((noise_labels == 0) & (pred == 0))[0]
    noise_0_pred_1 = np.where((noise_labels == 0) & (pred == 1))[0]
    noise_1_pred_0 = np.where((noise_labels == 1) & (pred == 0))[0]
    noise_1_pred_1 = np.where((noise_labels == 1) & (pred == 1))[0]
    return [noise_0_pred_0, noise_0_pred_1, noise_1_pred_0, noise_1_pred_1]


def data_process(noise_type, model_type='vanilla'):
    save_path = '../Training/config/' + 'databases_' + str(noise_type) + '.conf'
    data_filenames, gt_labels, noise_labels = read_joblib(save_path)
    gt_labels = np.array(gt_labels)
    noise_labels = np.array(noise_labels)
    samples_num = len(data_filenames)
    data = np.load(
        '../Training/output/' + noise_type + '/' + model_type + '/prob/fold1.npy')
    epoch_num = data.shape[0]
    if model_type == 'vanilla':
        vanilla_prob = np.zeros((epoch_num, samples_num))
    else:
        vanilla_prob = np.zeros((epoch_num, samples_num, 10))
    for i in range(epoch_num):
        if model_type == 'vanilla':
            vanilla_ = np.zeros(samples_num)
        else:
            vanilla_ = np.zeros((samples_num, 10))
        for fold in range(5):
            index = np.load(
                '../Training/output/' + noise_type + '/index/fold' + str(fold + 1) + '.npy', allow_pickle=True)[1]
            vanilla_data = np.load(
                '../Training/output/' + noise_type + '/' + model_type + '/prob/fold' + str(fold + 1) + '.npy')[i]
            vanilla_[index] = np.squeeze(vanilla_data)

        vanilla_prob[i] = vanilla_
    return data_filenames, gt_labels, noise_labels, vanilla_prob


def nll(p, label, eps=1e-10, base=2):
    """
    negative log likelihood (NLL)
    :param p: predictive labels
    :param eps: a small value prevents the overflow
    :param base: the base of log function
    :return: the mean of NLL
    """
    p = np.array(p)
    q = np.full(len(p), label)
    nll = -(q * np.log(p + eps) + (1. - q) * np.log(1. - p + eps))
    if base is not None:
        nll = np.clip(nll / np.log(base), a_min=0., a_max=1000)
    return np.mean(nll)


def prob_label_kld(p, label, number=10, w=None, base=2, eps=1e-10):
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))

    def _check_probablities(p, q=None):
        assert 0. <= np.all(p) <= 1.
        if q is not None:
            assert len(p) == len(q), \
                'Probabilies and ground truth must have the same number of elements.'

    _check_probablities(p)
    q_arr = np.full(number, label)
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)

    return (kld / number)[0][0]


def predictive_entropy(p, base=2, eps=1e-10, number=10):
    """
    calculate entropy in element-wise
    :param p: probabilities
    :param base: default exp
    :return: average entropy value
    """
    p_arr = np.asarray(p)

    def _check_probablities(p, q=None):
        assert 0. <= np.all(p) <= 1.
        if q is not None:
            assert len(p) == len(q), \
                'Probabilies and ground truth must have the same number of elements.'

    _check_probablities(p)
    enc = -(p_arr * np.log(p_arr + eps) + (1. - p_arr) * np.log(1. - p_arr + eps))
    if base is not None:
        enc = np.clip(enc / np.log(base), a_min=0., a_max=1000)
    enc_ = []
    for item in enc:
        enc_.append([np.sum(item) / number])
    return np.mean(enc_)


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def evaluate_dataset_noise(noise_type, data_filenames, gt_labels, noise_labels):
    from sklearn.metrics import accuracy_score
    print('***************************************************************************')
    print('***********************Noise type ' + noise_type + '**************************')
    print('***************************************************************************')
    print('Contain samples ' + str(len(data_filenames)))
    print('Noise benign', len(np.where(noise_labels == 0)[0]))
    print('Noise malware', len(np.where(noise_labels == 1)[0]))
    print('Ground benign', len(np.where(gt_labels == 0)[0]))
    print('Benign noise ratio is', round(
        (1 - accuracy_score(gt_labels[np.where(gt_labels == 0)[0]], noise_labels[np.where(gt_labels == 0)[0]])) * 100,
        2), '%')
    print('Ground malware', len(np.where(gt_labels == 1)[0]))
    print('Malware noise ratio is', round(
        (1 - accuracy_score(gt_labels[np.where(gt_labels == 1)[0]], noise_labels[np.where(gt_labels == 1)[0]])) * 100,
        2), '%')
    print('Noise ratio is', round((1 - accuracy_score(gt_labels, noise_labels)) * 100, 2), '%')
    print('***************************************************************************')


def prob2psx(cv_prob):
    cv_prob = list(cv_prob)
    prob_ = []
    for i in range(len(cv_prob)):
        prob_.append(1 - cv_prob[i])
    psx = np.array([prob_, cv_prob]).T
    return psx


def evaluate_cleanlab(gt_labels, noise_labels, ordered_label_errors, clean_malware=True):
    denoise_label = []
    if clean_malware:
        for i, item in enumerate(ordered_label_errors):
            if item:
                denoise_label.append((noise_labels[i] + 1) % 2)
            else:
                denoise_label.append(noise_labels[i])
    else:
        for i, item in enumerate(ordered_label_errors):
            if item and noise_labels[i] == 1:
                denoise_label.append((noise_labels[i] + 1) % 2)
            else:
                denoise_label.append(noise_labels[i])
    return round(accuracy_score(gt_labels, denoise_label), 6)
