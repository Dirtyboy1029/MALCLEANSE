import os
import multiprocessing
import collections
import warnings
import tempfile
import numpy as np
import tensorflow as tf
from ..tools import progressbar_wrapper, utils
from ..dataset_lib import build_dataset_from_numerical_data
from ..config import logging, ErrorHandler

# from ...config import logging, ErrorHandler

logger = logging.getLogger('core.feature.feature_extraction')
logger.addHandler(ErrorHandler)


def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class FeatureExtraction(object):
    """Produce features for ML algorithms"""

    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 file_ext=None,
                 update=False,
                 proc_number=2):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param intermediate_save_dir: a directory for saving meta information
        :param file_ext: file extent
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        self.naive_data_save_dir = naive_data_save_dir
        utils.mkdir(self.naive_data_save_dir)
        self.meta_data_save_dir = intermediate_save_dir
        utils.mkdir(self.meta_data_save_dir)
        self.file_ext = file_ext
        self.update = update
        self.proc_number = int(proc_number)

    def feature_extraction(self, sample_dir, use_order_features=False):
        """
        extract the android features from Android packages and save the extractions into designed directory
        :param sample_dir: malicious / benign samples for the subsequent process of feature extraction
        :param use_order_features: following the order of the provided sample paths
        """
        raise NotImplementedError

    def feature_preprocess(self, feature_path_list, gt_labels, noise_type):
        """
        pre-processing the naive data to accommodate the input format of ML algorithms
        :param feature_path_list: feature paths produced by the method of feature_extraction
        :param gt_labels: corresponding ground truth labels
        """
        raise NotImplementedError

    def feature2ipt(self, feature_path_list, labels=None, is_training_set=False, noise_type='random'):
        """
        Mapping features to the input space

        :param feature_path_list, a list of paths point to the features
        :param labels, ground truth labels
        :param is_training_set, boolean type
        """
        raise NotImplementedError

    @staticmethod
    def _check(sample_dir):
        """
        check a valid directory and produce a list of file paths
        """
        if isinstance(sample_dir, str):
            if not os.path.exists(sample_dir):
                print(sample_dir)
                MSG = "No such directory or file {} exists!".format(sample_dir)
                raise ValueError(MSG)
            elif os.path.isfile(sample_dir):
                sample_path_list = [sample_dir]
            elif os.path.isdir(sample_dir):
                sample_path_list = list(utils.retrive_files_set(sample_dir, "", ".apk|"))
                assert len(sample_path_list) > 0, 'No files'
            else:
                raise ValueError(" No such path {}".format(sample_dir))
        elif isinstance(sample_dir, list):
            sample_path_list = [path for path in sample_dir if os.path.isfile(path)]
        else:
            MSG = "A directory or a list of paths are allowed!"
            raise ValueError(MSG)

        return sample_path_list


class DrebinFeature(FeatureExtraction):
    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 file_ext='.drebin',
                 update=False,
                 proc_number=2):
        super(DrebinFeature, self).__init__(naive_data_save_dir,
                                            intermediate_save_dir,
                                            file_ext,
                                            update,
                                            proc_number)

    def feature_extraction(self, sample_dir, use_order_features=False):
        """
        drebin features
        :return: 2D list, [[a list of features from an apk],...,[a list of features from an apk]]
        """
        from ..feature.drebin.drebin import AxplorerMapping, get_drebin_feature

        sample_path_list = self._check(sample_dir)
        pool = multiprocessing.Pool(self.proc_number)
        pbar = progressbar_wrapper.ProgressBar()
        process_results = []
        tasks = []
        pmap = AxplorerMapping()

        for i, apk_path in enumerate(sample_path_list):
            sha256 = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256 + self.file_ext)
            if os.path.exists(save_path) and (not self.update):
                continue
            tasks.append(apk_path)
            process_results = pool.apply_async(get_drebin_feature,
                                               args=(apk_path, pmap, save_path),
                                               callback=pbar.CallbackForProgressBar)

        pool.close()
        if process_results:
            pbar.DisplayProgressBar(process_results, len(tasks), type='hour')
        pool.join()

        feature_path_list = []
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path):
                feature_path_list.append(save_path)
            else:
                warnings.warn("Fail to perform feature extraction for '{}'".format(apk_path))

        return feature_path_list

    def load_features(self, feature_path_list):
        """
        load features
        :param feature_path_list: feature paths produced by the method of feature_extraction
        :return: a list of features
        """
        from .drebin.drebin import wrapper_load_features
        feature_list = []
        n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(n_proc)
        for res in pool.imap(wrapper_load_features, feature_path_list):
            if not isinstance(res, Exception):
                feature_list.append(res)
            else:
                print(str(res))
        return feature_list

    def feature_preprocess(self, feature_path_list, gt_labels, noise_type):
        """
        pre-processing the naive data to accommodate the input format of ML algorithms
        :param feature_path_list: feature paths produced by the method of feature_extraction
        :param gt_labels: corresponding ground truth labels

        """
        vocab_path = os.path.join(self.meta_data_save_dir, noise_type + '.vocab')
        if self.update or (not os.path.exists(vocab_path)):
            assert len(feature_path_list) == len(gt_labels)
            features = self.load_features(feature_path_list)
            tmp_vocab = self.get_vocabulary(features)
            # we select 10,000 features
            selected_vocab = self.feature_selection(features, gt_labels, tmp_vocab, dim=10000)
            utils.dump_pickle(selected_vocab, vocab_path)
            print('save vocab to ' + vocab_path)
        return

    def feature2ipt(self, feature_path_list, labels=None,  is_training_set=False,
                    noise_type='random'):
        """
        Mapping features to the input space
        :param feature_path_list: the feature paths produced by the method of feature_extraction
        :param labels: the ground truth labels correspond to features
        :param is_training_set, not used here
        :return tf.data; input dimension of an item of data
        :rtype tf.data.Dataset object; integer
        """
        # load
        vocab_path = os.path.join(self.meta_data_save_dir, noise_type + '.vocab')

        if not os.path.exists(vocab_path):
            if labels is not None:
                self.feature_preprocess(feature_path_list, labels, noise_type=noise_type)
            else:
                raise ValueError('Need ground truth label!')
        vocab = utils.read_pickle(vocab_path)
        print('read vocab from ' + vocab_path)
        dim = len(vocab)
        print(dim)
        features = self.load_features(feature_path_list)
        dataX_np = self.get_feature_representation(features, vocab)
        if labels is not None:
            return build_dataset_from_numerical_data((dataX_np, labels)), dim, dataX_np
        else:
            return build_dataset_from_numerical_data(dataX_np), dim, dataX_np

    def feature_selection(self, train_features, train_y, vocab, dim):
        """
        feature selection
        :param train_features: 2D feature
        :type train_features: numpy object
        :param train_y: ground truth labels
        :param vocab: a list of words (i.e., features)
        :param dim: the number of remained words
        :return: chose vocab
        """
        is_malware = (train_y == 1)
        mal_features = np.array(train_features, dtype=object)[is_malware]
        ben_features = np.array(train_features, dtype=object)[~is_malware]

        if (len(mal_features) <= 0) or (len(ben_features) <= 0):
            return vocab

        mal_representations = self.get_feature_representation(mal_features, vocab)
        mal_frequency = np.sum(mal_representations, axis=0) / float(len(mal_features))
        ben_representations = self.get_feature_representation(ben_features, vocab)
        ben_frequency = np.sum(ben_representations, axis=0) / float(len(ben_features))

        # eliminate the words showing zero occurrence in apk files
        is_null_feature = np.all(mal_representations == 0, axis=0) & np.all(ben_representations, axis=0)
        mal_representations, ben_representations = None, None
        vocab_filtered = list(np.array(vocab)[~is_null_feature])

        if len(vocab_filtered) <= dim:
            return vocab_filtered
        else:
            feature_frq_diff = np.abs(mal_frequency[~is_null_feature] - ben_frequency[~is_null_feature])
            position_flag = np.argsort(feature_frq_diff)[::-1][:dim]

            vocab_selected = []
            for p in position_flag:
                vocab_selected.append(vocab_filtered[p])
            return vocab_selected

    def load_vocabulary(self, data_type):
        vocab_path = os.path.join(self.meta_data_save_dir, data_type + '_drebin.vocab')
        if not os.path.exists(vocab_path):
            raise ValueError("A vocabulary is needed.")
        vocab = utils.read_pickle(vocab_path)
        return vocab

    @staticmethod
    def get_vocabulary(feature_list, n=300000):
        """
        obtain the vocabulary based on the feature
        :param feature_list: 2D list of naive feature
        :param n: the number of top frequency items
        :return: feature vocabulary
        """
        c = collections.Counter()

        for features in feature_list:
            for feature in features:
                c[feature] = c[feature] + 1

        vocab, count = zip(*c.most_common(n))
        return list(vocab)

    @staticmethod
    def get_feature_representation(feature_list, vocab):
        """
        mapping feature to numerical representation
        :param feature_list: 2D feature list with shape [number of files, number of feature]
        :param vocab: a list of words
        :return: 2D representation
        :rtype numpy.ndarray
        """
        N = len(feature_list)
        M = len(vocab)

        assert N > 0 and M > 0

        representations = np.zeros((N, M), dtype=np.float32)
        dictionary = dict(zip(vocab, range(len(vocab))))
        for i, features in enumerate(feature_list):
            if len(features) > 0:
                filled_positions = [idx for idx in list(map(dictionary.get, features)) if idx is not None]
                if len(filled_positions) != 0:
                    representations[i, filled_positions] = 1.
                else:
                    warnings.warn("Produce zero feature vector.")

        return representations
