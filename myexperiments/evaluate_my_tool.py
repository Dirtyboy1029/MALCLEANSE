# -*- coding: utf-8 -*- 
# @Time : 2024/5/13 15:27 
# @Author : DirtyBoy 
# @File : evaluate_my_tool.py
from utils import read_joblib
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from utils import evaluate_dataset_noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str,
                        default='thr_3_10')
    parser.add_argument('-anomaly_detection_algorithm', '-ada', type=str,
                        default='isolation_forest')  #isolation_forest
    args = parser.parse_args()
    noise_type = args.noise_type
    anomaly_detection_algorithm = args.anomaly_detection_algorithm  # 0.97 9.61  10.16  3.47  0.99

    print('-----------------------------------------------------------------------------------')
    print(
        '-------------' + noise_type + '     ' + anomaly_detection_algorithm + '--------------------------')
    print('-----------------------------------------------------------------------------------')

    save_path = '../Training/config/' + 'databases_' + str(noise_type) + '.conf'
    data_filenames, gt_labels, noise_labels = read_joblib(save_path)
    gt_labels = np.array(gt_labels)
    noise_labels = np.array(noise_labels)
    evaluate_dataset_noise(noise_type, data_filenames, gt_labels, noise_labels)
    noise_benign_index = np.where(noise_labels == 0)[0]
    noise_malware_index = np.where(noise_labels == 1)[0]
    benign_noise = 0
    for i in noise_benign_index:
        if gt_labels[i] != noise_labels[i]:
            benign_noise = benign_noise + 1

    malware_noise = 0
    for i in noise_malware_index:
        if gt_labels[i] != noise_labels[i]:
            malware_noise = malware_noise + 1

    print("Total samples", len(gt_labels))
    accuracy = accuracy_score(gt_labels, noise_labels)
    accuracy1 = accuracy_score(gt_labels[noise_benign_index], noise_labels[noise_benign_index])
    MSG = "The Noise benign on the dataset is "
    print(MSG, benign_noise)
    print("The Noise malware on the dataset is ", malware_noise)
    MSG = "The source accuracy on the dataset is {:.2f}%, Noise Benign accurce is {:.2f}%."
    print(MSG.format(accuracy * 100, accuracy1 * 100))

    samples_num = len(data_filenames)

    label_noise_mask = np.load(
        'my_tool_difference/' + anomaly_detection_algorithm + '/' + noise_type + '.npy')
    tmp_label = []
    for i in label_noise_mask:
        if i == -1:
            tmp_label.append(1)
        else:
            tmp_label.append(0)
    noise_labels[noise_benign_index] = tmp_label

    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    benign_noise = 0
    for i in noise_benign_index:
        if gt_labels[i] != noise_labels[i]:
            benign_noise = benign_noise + 1
    accuracy = accuracy_score(gt_labels, noise_labels)
    accuracy1 = accuracy_score(gt_labels[noise_benign_index], tmp_label)
    recall = recall_score(gt_labels[noise_benign_index], tmp_label)
    prescion = precision_score(gt_labels[noise_benign_index], tmp_label)
    f1 = f1_score(gt_labels[noise_benign_index], tmp_label)
    print("The Noise benign denoise on the dataset is ", benign_noise)
    MSG = "The clean accuracy on the dataset is {:.2f}%, Noise Benign accurce is {:.2f}%."
    print(MSG.format(accuracy * 100, accuracy1 * 100))
    print((1 - accuracy) * 100, '%')
    MSG = "Recall is {:.5f}%"
    print(MSG.format(recall * 100))

    MSG = "Prescion is {:.5f}%"
    print(MSG.format(prescion * 100))

    MSG = "F1 score is {:.5f}%"
    print(MSG.format(f1 * 100))

    if 'variation' in noise_type:
        def txt_to_list(txt_path):
            f = open(txt_path, "r")
            return f.read().splitlines()


        obfs_list = txt_to_list('../Training/config/Malradar_variation.txt')
        malware_common = []
        malware_obfs = []
        for item in noise_benign_index:
            if data_filenames[item] in obfs_list:
                malware_obfs.append(item)
            else:
                if gt_labels[item] == 1:
                    malware_common.append(item)
        print('obfs is', len(malware_obfs))
        print('Noise is ', len(malware_obfs) - np.sum(noise_labels[malware_obfs]))
        print('LAcc is ', accuracy_score(gt_labels[malware_obfs], noise_labels[malware_obfs]))
        print('common is ', len(malware_common))
        print('Noise is ', len(malware_common) - np.sum(noise_labels[malware_common]))
        print('LAcc is ', accuracy_score(gt_labels[malware_common], noise_labels[malware_common]))
