# -*- coding: utf-8 -*- 
# @Time : 2023/3/29 21:42 
# @Author : DirtyBoy 
# @File : evaluation.py
import pickle
import argparse
from ML.drebinSVM.utils import model_evaluate, read_drebin_feature_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', '-mt', type=str, default="mldrebin")
    parser.add_argument('-train_data', '-train_d', type=str, default="malradar")
    parser.add_argument('-test_data', '-test_d', type=str, default="amd")
    parser.add_argument('-vocab_type', '-vt', type=str, default="malradar")
    parser.add_argument('-input_dim', '-i', type=int, default=10000)
    args = parser.parse_args()
    model_type = args.model_type
    train_data = args.train_data
    test_data = args.test_data
    vocab_type = args.vocab_type
    input_dim = args.input_dim
    sample_list, x_test, y_test, _ = read_drebin_feature_vector(test_data, False, input_dim)

    model_save_path = '/home/lhd/Android_malware_detector_set/ML/drebinSVM/model/model_type_' + model_type + '_data_type_' + train_data + '_vocab_' + vocab_type + str(
        input_dim) + '.pkl'

    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)
    model_evaluate(model, x_test, y_test, name=test_data)
