# -*- coding: utf-8 -*- 
# @Time : 2024/9/13 18:46 
# @Author : DirtyBoy 
# @File : evaluate_model.py
import tensorflow as tf
from core.data_preprocessing import data_preprocessing
import argparse
from core.ensemble.vanilla import Vanilla


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str, default='thr_1_18')
    args = parser.parse_args()
    noise_type = args.noise_type
    model = Vanilla(architecture_type='dnn',
                    model_directory='../Model/' + noise_type)

    mwo_model = Vanilla(architecture_type='dnn',
                        model_directory='../Model/mwo_' + noise_type)

    robust_model = Vanilla(architecture_type='dnn',
                           model_directory='../Model/robust_' + noise_type)

    for item in ['backdoorware', 'adware', 'smsware', 'ransomware']:
        _, base_gt_labels, _, _, base_x_train, _ = data_preprocessing(
            noise_type=item, feature_type='test_data', model_type=noise_type)

        _, mwo_gt_labels, _, _, mwo_x_train, _ = data_preprocessing(
            noise_type=item, feature_type='test_data', model_type='mwo_' + noise_type)

        _, robust_gt_labels, _, _, robust_x_train, _ = data_preprocessing(
            noise_type=item, feature_type='test_data', model_type='robust_' + noise_type)

        print('-----------------------------' + item + '-----------------------------------')
        print('----------------------------- base model -----------------------------------')
        model.evaluate(base_x_train, base_gt_labels)
        print('----------------------------- mwo model -----------------------------------')
        mwo_model.evaluate(mwo_x_train, mwo_gt_labels)
        print('----------------------------- robust model -----------------------------------')
        robust_model.evaluate(robust_x_train, robust_gt_labels)
        print('----------------------------------------------------------------')
