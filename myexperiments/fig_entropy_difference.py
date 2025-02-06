# -*- coding: utf-8 -*- 
# @Time : 2024/5/22 11:24 
# @Author : DirtyBoy 
# @File : fig_entropy_difference.py
import numpy as np
from utils import data_process_for_bayesian, evaluate_dataset_noise, prob2psx, evaluate_cleanlab, predictive_entropy
from sklearn.metrics import accuracy_score
import cleanlab as clb
import argparse
import matplotlib.pyplot as plt


def aaaa(data):
    q1, q3 = np.percentile(data, [25, 75])  # 使用30%和70%作为四分位数

    median = np.median(data)
    iqr = q3 - q1

    whisker_low = q1 - 1.5 * iqr
    whisker_high = q3 + 1.5 * iqr
    return q1, q3, median, whisker_low, whisker_high


def tmp_data(noise_type):
    data_filenames, gt_labels, noise_labels, bayesian_prob = data_process_for_bayesian(data_type, noise_type,
                                                                                       noise_hyper)
    noise_benign_index = np.where(noise_labels == 0)[0]

    noise_benign_gt_malware_index = [i for i in noise_benign_index if gt_labels[i] != noise_labels[i]]
    noise_benign_gt_benign_index = [i for i in noise_benign_index if gt_labels[i] == noise_labels[i]]

    entropy = []
    for i in range(len(bayesian_prob)):
        if i % 2 == 0:
            entropy.append(np.array([predictive_entropy(item) for item in bayesian_prob[i]]))

    benign_entropy = []
    malware_entropy = []

    for i in range(len(entropy)):
        benign_entropy.append(entropy[i][noise_benign_gt_benign_index])
        malware_entropy.append(entropy[i][noise_benign_gt_malware_index])
    return benign_entropy, malware_entropy, len(entropy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_type', '-dt', type=str, default="malradar")
    parser.add_argument('-noise_type', '-nt', type=str,
                        default='Alibaba')  ##['AVG', 'F-Secure', 'Ikarus', 'Sophos', 'Kaspersky','Alibaba','ZoneAlarm']
    parser.add_argument('-noise_hyper', '-nh', type=int, default=0)
    args = parser.parse_args()
    data_type = args.train_data_type
    noise_type = args.noise_type
    noise_hyper = args.noise_hyper
    noise_type = 'F-Secure'
    benign_entropy, malware_entropy, num = tmp_data(noise_type)
    q1, q3, median, whisker_low, whisker_high = aaaa(malware_entropy)
    print(q1, q3, median, whisker_low, whisker_high)
    # fig, axs = plt.subplots(5, 1, figsize=(6, 8))
    #
    # for ii, noise_type in enumerate(['F-Secure', ]):  # 'Ikarus', 'ZoneAlarm', 'Alibaba', 'Microsoft'
    #     benign_entropy, malware_entropy, num = tmp_data(noise_type)
    #     for i in range(num):
    #         bp1 = axs[ii].boxplot(benign_entropy[i], showfliers=False, positions=[i + 1 - 0.2], widths=0.3,
    #                               patch_artist=True,
    #                               boxprops=dict(facecolor='green'), whis=1.5)
    #
    #
    #         # for whisker in bp1['whiskers']:
    #         #     whisker.set_visible(False)
    #         # for cap in bp1['caps']:
    #         #     cap.set_visible(False)
    #         bp2 = axs[ii].boxplot(malware_entropy[i], showfliers=False, positions=[i + 1 + 0.2], widths=0.3,
    #                               patch_artist=True,
    #                               boxprops=dict(facecolor='red'), whis=1.5)
    #         # for whisker in bp2['whiskers']:
    #         #     whisker.set_visible(False)
    #         # for cap in bp2['caps']:
    #         #     cap.set_visible(False)
    #
    #     axs[ii].set_title(noise_type, fontsize=10, fontweight='bold')
    #
    # axs[0].set_xticks([])
    # axs[1].set_xticks([])
    # axs[2].set_xticks([])
    # axs[3].set_xticks([])
    # axs[4].set_xticks([])
    #
    # axs[0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # axs[0].set_xticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # axs[1].set_xticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # axs[2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # axs[2].set_xticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # axs[3].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # axs[3].set_xticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # axs[4].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # axs[4].set_xticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    #
    # axs[0].set_yticks([])
    # axs[1].set_yticks([])
    # axs[2].set_yticks([])
    # axs[3].set_yticks([])
    # axs[4].set_yticks([])
    #
    # # axs[0].set_ylim(0, 0.03)
    # # axs[2].set_ylim(0, 0.08)
    # # axs[3].set_ylim(0, 0.003)
    # # axs[4].set_ylim(0, 0.05)
    # axs[4].set_xlabel('Epoch', fontsize=18, fontweight='bold')
    # axs[2].set_ylabel('Entropy', fontsize=18, fontweight='bold')
    # axs[0].plot([], [], color='red', label='MisLabeled')
    # axs[0].plot([], [], color='green', label='Correctly Labeled')
    # legend_properties = {'weight': 'bold', 'size': 8}
    # axs[0].legend(loc='upper left', prop=legend_properties)
    # plt.tight_layout()
    # # plt.savefig('samples_diff_box.pdf')
    # plt.show()
