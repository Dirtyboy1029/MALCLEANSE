# -*- coding: utf-8 -*- 
# @Time : 2024/5/7 15:30 
# @Author : DirtyBoy 
# @File : evaluate_cleanlab.py
import numpy as np
from utils import data_process, evaluate_dataset_noise, prob2psx, evaluate_cleanlab
from sklearn.metrics import accuracy_score
import cleanlab as clb
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str, default='thr_18')
    parser.add_argument('-model_type', '-mt', type=str, nargs='+',
                        default=['bayesian'])
    parser.add_argument('-i', type=int,
                        default=4)
    args = parser.parse_args()
    iii = args.i
    Noise_type = args.noise_type
    model_type_set = args.model_type
    all = []

    for j in range(1, iii):
        Tmp = []
        if 'random' in Noise_type:
            a, b = Noise_type.split('_', 1)
        else:
            a, b = Noise_type.rsplit('_', 1)
        noise_type = a + '_' + str(j) + '_' + b
        prob_set = []
        for model_type in model_type_set:
            data_filenames, gt_labels, noise_labels, model_prob = data_process(noise_type, model_type=model_type)
            prob_set.append(model_prob)
        evaluate_dataset_noise(noise_type, data_filenames, gt_labels, noise_labels)
        print('******' + 'Cross-validation for ' + str(
            (prob_set[0].shape[0]) * 5) + ' epochs, evaluating every 10 epochs.' + '*******')
        print('***************************************************************************')
        for i in range(prob_set[0].shape[0]):
            if (i + 1) % 5 == 0:
            #if i < int(prob_set[0].shape[0] / 2):
                tmp = []
                print('******************************epoch ' + str(
                    (i + 1)) + '***************************************')
                noise_labels = [int(item) for item in noise_labels]
                ordered_label_errors_set = []
                for iii, item in enumerate(prob_set):
                    if model_type_set[iii] == 'vanilla':
                        ordered_label_errors_set.append(
                            clb.pruning.get_noise_indices(s=noise_labels, psx=prob2psx(item[i]),
                                                          prune_method='prune_by_noise_rate'))
                    else:
                        ordered_label_errors_set.append(
                            clb.pruning.get_noise_indices(s=noise_labels, psx=prob2psx(np.mean(item[i], axis=1)),
                                                          prune_method='prune_by_noise_rate'))
                if 'random' in noise_type:
                    acc_set = [evaluate_cleanlab(gt_labels, noise_labels, demo, clean_malware=False).astype(np.float64)
                               for demo in
                               ordered_label_errors_set]
                    ending = ''
                    for ii, acc in enumerate(acc_set):
                        ending = ending + model_type_set[ii] + ' acc ' + str(round(acc * 100, 2)) + '%' + '  '

                else:
                    acc_set = [evaluate_cleanlab(gt_labels, noise_labels, demo, clean_malware=True).astype(np.float64)
                               for demo in
                               ordered_label_errors_set]
                    ending = ''
                    for ii, acc in enumerate(acc_set):
                        ending = ending + model_type_set[ii] + ' acc ' + str(round(acc * 100, 2)) + '%' + '  '
                print(ending)
                tmp.append(acc_set)
                Tmp.append(tmp)
            all.append(Tmp)
        average_result = np.mean(all, axis=0)
        print(all)
    # np.save(data_type + '_' + Noise_type + '_' + feature_type + '_' + str(noise_hyper), np.array(all))
    # print(average_result)

    for jj, demo2 in enumerate(average_result):
        Ending = ''
        for iiii, demo3 in enumerate(demo2[0]):
            Ending = Ending + model_type_set[iiii] + ' acc ' + str(round(demo3 * 100, 2)) + '%' + '  '
        print(Ending)
