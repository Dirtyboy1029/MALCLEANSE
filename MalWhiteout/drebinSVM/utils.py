# -*- coding: utf-8 -*- 
# @Time : 2023/3/14 16:39 
# @Author : DirtyBoy 
# @File : utils.py

import os
import glob
import logging
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.feature_extraction.text import CountVectorizer as CT
import os
import CommonModules as CM
import psutil
import time
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import scipy.sparse as sp
import pandas as pd
import warnings


def bool_noise_to_list(noise_rate_label_errors_mask):
    noise_rate_label_error = []
    for item in noise_rate_label_errors_mask:
        if item == True:
            noise_rate_label_error.append(1)
        else:
            noise_rate_label_error.append(0)
    return noise_rate_label_error


def save_to_vocab(goal, file_path):
    f = open(file_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def vocab_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def model_evaluate(model, x, gt_label, threshold=0.5, name='test'):
    x_prob = model.predict(x)
    x_pred = (x_prob >= threshold).astype(np.int32)

    # metrics
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
    accuracy = accuracy_score(gt_label, x_pred)
    b_accuracy = balanced_accuracy_score(gt_label, x_pred)

    MSG = "The accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, accuracy * 100))
    MSG = "The balanced accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, b_accuracy * 100))
    is_single_class = False
    if np.all(gt_label == 1.) or np.all(gt_label == 0.):
        is_single_class = True
    if not is_single_class:
        tn, fp, fn, tp = confusion_matrix(gt_label, x_pred).ravel()

        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(gt_label, x_pred, average='binary')

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        print(MSG.format(fnr * 100, fpr * 100, f1 * 100))
        return MSG.format(fnr * 100, fpr * 100, f1 * 100) + "The balanced accuracy on the {} dataset is {:.5f}%".format(
            name, accuracy * 100)


def read_drebin_feature_vector(data_type, FeatureOption, dim=10000):
    # creating feature vector
    if data_type == 'amd':
        TrainMalSet = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_ml_model/amd/malware_2000'
        TrainGoodSet = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_ml_model/amd/benign_2000'
    elif data_type == 'malradar':
        TrainMalSet = '/home/public/rmt/heren/experiment/cl-exp/maldir'
        TrainGoodSet = '/home/public/rmt/heren/experiment/cl-exp/gooddir'
    elif data_type == 'drebin':
        TrainMalSet = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_ml_model/drebin/malware_5452'
        TrainGoodSet = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_ml_model/drebin/benign_5448'
    else:
        raise ValueError
    logging.basicConfig(level=logging.INFO)
    Logger = logging.getLogger('main.stdout')
    TrainMalSamples = CM.IO.ListFiles(TrainMalSet, ".data")
    TrainGoodSamples = CM.IO.ListFiles(TrainGoodSet, ".data")
    sample_list = []
    for sample in TrainMalSamples:
        sample_list.append(sample.split('/')[-1])
    for sample in TrainGoodSamples:
        sample_list.append(sample.split('/')[-1])

    # save file name
    with open(data_type + 'samples_drebin.txt', 'w') as sf:
        sf.write(str(sample_list))

    Logger.info("Loaded Samples")

    # label training sets malware as 1 and goodware as 0
    Train_Mal_labels = np.ones(len(TrainMalSamples), dtype=int)
    Train_Good_labels = np.zeros(len(TrainGoodSamples), dtype=int)
    y_train = np.concatenate((Train_Mal_labels, Train_Good_labels), axis=0)

    if os.path.exists('/home/lhd/Android_malware_detector_set/ML/drebinSVM/config/domin_drebin.vocab'):
        vocab = vocab_to_list('/home/lhd/Android_malware_detector_set/ML/drebinSVM/config/domin_drebin.vocab')
        feature = vocab_to_list('/home/lhd/Android_malware_detector_set/ML/drebinSVM/config/all_drebin.vocab')
        FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                               binary=FeatureOption, vocabulary=feature)
        x_train = FeatureVectorizer.fit_transform(TrainMalSamples + TrainGoodSamples)
        N = len(y_train)
        M = len(vocab)
        represention = np.zeros((N, M), dtype=np.float64)

        for i, item in enumerate(vocab):
            index = feature.index(item)
            represention[:, i] = x_train.todense()[:, index].reshape(N)[0]
        # warnings.warn('特征不存在，生成' + str(num) + '个全0特征!!!')
        x_train = represention
        features = vocab
    else:
        FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                               binary=FeatureOption)
        x_train = FeatureVectorizer.fit_transform(TrainMalSamples + TrainGoodSamples)
        features = FeatureVectorizer.get_feature_names()
        if len(features) > dim:
            # with feature selection
            Logger.info("Gonna select %s features", dim)
            fs_algo = SelectKBest(chi2, k=dim)
            x_train = fs_algo.fit_transform(x_train, y_train)
            vocab = fs_algo.get_feature_names_out(features)
            vocab_domin_path = '/home/lhd/Android_malware_detector_set/ML/drebinSVM/config/domin_drebin.vocab'
            vocab_all_path = '/home/lhd/Android_malware_detector_set/ML/drebinSVM/config/all_drebin.vocab'
            save_to_vocab(vocab, file_path=vocab_domin_path)
            save_to_vocab(features, file_path=vocab_all_path)
            x_train = x_train.todense()
            print('feature selected completed! feature vocab file saved')
    print('Data reading completed!!')
    print('Format of data:', 'feature type: ', x_train.shape, 'label type: ', y_train.shape)
    return sample_list, x_train, y_train, features
