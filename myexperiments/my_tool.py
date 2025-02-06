# -*- coding: utf-8 -*- 
# @Time : 2024/5/11 21:11 
# @Author : DirtyBoy 
# @File : my_tool_difference.py
import numpy as np
from utils import data_process, evaluate_dataset_noise, get_confident_joint_index
from sklearn.cluster import KMeans
from metrics_utils import *
import argparse, os
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str, default='thr_3_10')
    parser.add_argument('-detection_model', '-dm', type=str, default='if')
    parser.add_argument('-svm_kernel', '-sk', type=str, default='rbf')  # ['linear'、'poly'、'rbf'、'sigmoid']
    args = parser.parse_args()
    detection_model = args.detection_model
    svm_kernel = args.svm_kernel
    noise_type = args.noise_type

    if not os.path.isdir('my_tool_difference/dbscan/'):
        os.makedirs('my_tool_difference/dbscan/')
    if not os.path.isdir('my_tool_difference/isolation_forest/'):
        os.makedirs('my_tool_difference/isolation_forest/')

    print('-----------------------------------------------------------------------------------')
    print('-------------' + noise_type + '--------------------------')
    print('-----------------------------------------------------------------------------------')
    data_filenames, gt_labels, noise_labels, bayesian_prob = data_process(
        noise_type, model_type='bayesian')
    print('Source benign noise number is ',
          len(np.where(noise_labels == 0)[0]) - len(np.where(gt_labels == 0)[0]))
    print('Source benign noise ratio is {:.5f}%'.format(
        ((len(np.where(noise_labels == 0)[0]) - len(np.where(gt_labels == 0)[0])) / len(
            np.where(noise_labels == 0)[0])) * 100))

    noise_benign_index = np.where(noise_labels == 0)[0]
    noise_malware_index = np.where(noise_labels == 1)[0]

    train_data = []
    test_data = []
    for epoch in range(30):
        bay_prob = bayesian_prob[epoch]
        uc_entropy = np.array([predictive_entropy(prob_) for prob_ in bay_prob])
        uc_kld = np.array([predictive_kld(prob_) for prob_ in bay_prob])
        uc_std = np.array([predictive_std(prob_) for prob_ in bay_prob])
        uc_mean = np.array([np.mean(prob_) for prob_ in bay_prob])

        train_data.append(uc_entropy[noise_benign_index])
        train_data.append(uc_kld[noise_benign_index])
        train_data.append(uc_std[noise_benign_index])
        train_data.append(uc_mean[noise_benign_index])


    train_data = np.array(train_data).T
    bay_prob = np.mean(bayesian_prob[29], axis=1)
    confident_joint = get_confident_joint_index(noise_labels, bay_prob)
    print(len(confident_joint[0]), len(confident_joint[2]))
    eva_noise_ratio = len(confident_joint[2]) / (len(confident_joint[2]) + len(confident_joint[0]))
    eva_noise_num = int(len(np.where(noise_labels == 0)[0]) * eva_noise_ratio)
    print('evalaute noise ratio is {:.5f}%'.format(eva_noise_ratio * 100))
    print('evalaute noise number is ', eva_noise_num)

    if detection_model == 'if':
        isolation_forest = IsolationForest(n_estimators=100,
                                           contamination=eva_noise_ratio)
        isolation_forest.fit(train_data)
        outliers_isolation_forest = isolation_forest.predict(train_data)
        np.save('my_tool_difference/isolation_forest/' + noise_type,
                outliers_isolation_forest)
    elif detection_model == 'dbscan':
        dbscan = DBSCAN(eps=0.1,
                        min_samples=eva_noise_num)  
        dbscan.fit(train_data)
        outliers_dbscan = dbscan.labels_
        np.save('my_tool_difference/dbscan/' + noise_type, outliers_dbscan)
