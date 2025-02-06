# -*- coding: utf-8 -*- 
# @Time : 2023/3/29 10:15 
# @Author : DirtyBoy 
# @File : train.py
import argparse
import pickle
from ML.drebinSVM.utils import read_drebin_feature_vector
from sklearn.svm import SVC
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', '-dt', type=str, default="malradar")
    parser.add_argument('-model_type', '-mt', type=str, default="svmdrebin")
    parser.add_argument('-vocab_type', '-vt', type=str, default="drebin")
    parser.add_argument('-input_dim', '-i', type=int, default=10000)
    args = parser.parse_args()
    model_type = args.model_type
    data_type = args.data_type
    vocab_type = args.vocab_type
    input_dim = args.input_dim
    sample_list, x_train, y_train,_ = read_drebin_feature_vector(data_type, False, input_dim)

    Clf = SVC(kernel='linear', probability=True,verbose=1)
    Clf.fit(x_train, y_train)
    model_save_path = '/home/lhd/Android_malware_detector_set/ML/drebinSVM/model/model_type_' + model_type + '_data_type_' + data_type + '_vocab_' + vocab_type + str(
        input_dim) + '.pkl'
    with open(model_save_path, 'wb') as f:
        pickle.dump(Clf, f)
