# # -*- coding: utf-8 -*-

import os
import gzip
import wget
import cPickle as pickle

import numpy as np
import torch
import torch.utils.data as data


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DomainDatasetBase(data.ConcatDataset):
    """ Base class for multi domain dataset class
    subclasses must have these class variables:
        all_domain_key: domain keys for the dataset
        SingleDataset: dataset class for one domain dataset
    """
    SingleDataset = None

    def __init__(self, domain_keys, require_domain=True, datasets=None):
        """ Base class for multi domain dataset class
        Args:
            domain_keys: list or str
            require_domain: Boolean
        """
        assert isinstance(domain_keys, list) or isinstance(domain_keys, str)
        if isinstance(domain_keys, list):
            self.domain_keys = domain_keys
        elif isinstance(domain_keys, str):
            self.domain_keys = [x for x in domain_keys.split(',')]
        self.require_domain = require_domain
        self.domain_dict = dict(zip(self.domain_keys, range(len(self.domain_keys))))

        if datasets is None:
            datasets = []
            for domain_key in self.domain_keys:
                extra_args = {k: v for dic in [self.domain_specific_params(), self.domain_default_params()] for k, v in dic.items()}
                datasets += [self.get_single_dataset(domain_key, **extra_args)]
        super(DomainDatasetBase, self).__init__(datasets)

    def get_single_dataset(self, domain_key, **kwargs):
        return self.SingleDataset(domain_key, **kwargs)

    def domain_specific_params(self):
        return {}

    def domain_default_params(self):
        return {}

    def __getitem__(self, idx):
        X, y, d = super(DomainDatasetBase, self).__getitem__(idx)
        if self.require_domain:
            D = self.domain_dict[d]
            return X, y, D
        return X, y

    @classmethod
    def get_disjoint_domains(cls, domain_keys):
        if isinstance(domain_keys, str):
            domain_keys = [x for x in domain_keys.split(',')]

        all_domain_keys = cls.get('all_domain_key')[:]
        for domain_key in domain_keys:
            all_domain_keys.remove(domain_key)
        return all_domain_keys

    @classmethod
    def get(cls, name):
        if hasattr(cls, name):
            return getattr(cls, name)
        elif hasattr(cls.SingleDataset, name):
            return getattr(cls.SingleDataset, name)


CONFIG = {}
CONFIG['url'] = 'https://github.com/ghif/mtae/raw/master/MNIST_6rot.pkl.gz'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets')


class _SingleMNISTR(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/MNIST_6rot.pkl.gz')
    all_domain_key = ['M0', 'M15', 'M30', 'M45', 'M60', 'M75']
    input_shape = (1, 16, 16)
    num_classes = 10

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        domain_id = self.all_domain_key.index(domain_key)
        img_rows, img_cols = self.input_shape[1:]

        all_domains = pickle.load(gzip.open(self.path, 'rb'))
        X, y = all_domains[domain_id]
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        self.X = X
        self.y = y

    def download(self):
        output_dir = os.path.dirname(self.path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wget.download(CONFIG['url'], out=self.path)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)


class MNISTR(DomainDatasetBase):
    """ Rotation version MNIST

    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleMNISTR


class _SingleBiasedMNISTR(data.Dataset):
    """
        | domain_key | what percentage of 1~5 are used |
        | 0          | 100% |
        | 15         | 85%  |
        | 30         | 70%  |
        | 45         | 55%  |
        | 60         | 40%  |
        | 75         | 25%  |
    """
    path = os.path.expanduser('~/.torch/datasets/MNIST_6rot.pkl.gz')
    all_domain_key = ['0', '15', '30', '45', '60', '75']
    input_shape = (1, 16, 16)
    num_classes = 10
    bias = {'0' : 1,
            '15': 0.85,
            '30': 0.7,
            '45': 0.55,
            '60': 0.4,
            '75': 0.25}

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        domain_id = self.all_domain_key.index(domain_key)
        img_rows, img_cols = self.input_shape[1:]

        all_domains = pickle.load(gzip.open(self.path, 'rb'))
        X, y = all_domains[domain_id]
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        indices = self.get_biased_indices(y)
        self.X = X[indices]
        self.y = y[indices]

    def download(self):
        output_dir = os.path.dirname(self.path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wget.download(CONFIG['url'], out=self.path)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)

    def get_biased_indices(self, y):
        def trim(indices):
            biased_len = int(len(indices) * self.bias[self.domain_key])
            indices = indices[:biased_len]
            return indices

        zero_to_four = []
        for i in range(5):
            zero_to_four += trim(np.where(y == i)[0]).tolist()
        five_to_nine = np.where(y >= 5)[0]
        if len(zero_to_four) == 0:
            return five_to_nine
        indices = np.append(zero_to_four, five_to_nine)
        return indices


class BiasedMNISTR(DomainDatasetBase):
    """ Rotation version MNIST with d -> y

    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleBiasedMNISTR


def get_biased_mnistr(bias):
    """
    Args:
        bias: dict of class and domain relationship.
              example:{'0' : 1,
                      '15': 0.85,
                      '30': 0.7,
                      '45': 0.55,
                      '60': 0.40,
                      '75': 0.25}
    """
    _SingleBiasedMNISTR.bias = bias
    BiasedMNISTR.SingleDataset = _SingleBiasedMNISTR
    return BiasedMNISTR


if __name__ == '__main__':
    dataset = MNISTR(domain_keys=['0', '30', '60'])
    dataset = MNISTR(domain_keys='0,15,30')
    dataset = MNISTR(domain_keys=MNISTR.get_disjoint_domains(['0', '30', '60']))
    print(len(np.where(dataset.datasets[0].y < 5)[0]))
    print(len(np.where(dataset.datasets[0].y >= 5)[0]))

    dataset = BiasedMNISTR(domain_keys=['15', '30', '45'])
    print(len(np.where(dataset.datasets[0].y < 5)[0]))
    print(len(np.where(dataset.datasets[0].y >= 5)[0]))

    UnnaturalMNISTR = get_biased_mnistr(
        {'0' : 1,
        '15': 0.85,
        '30': 0.7,
        '45': 0.55,
        '60': 1,
        '75': 0.25})
    dataset = UnnaturalMNISTR(domain_keys=['60', '75'])
    print(len(np.where(dataset.datasets[0].y < 5)[0]))
    print(len(np.where(dataset.datasets[0].y >= 5)[0]))
    print(len(np.where(dataset.datasets[1].y < 5)[0]))
    print(len(np.where(dataset.datasets[1].y >= 5)[0]))
