# -*- coding: utf-8 -*-

import copy
import random

import torch
from torch import nn, optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn import metrics

from dataset import Subset


class Discriminator(nn.Module):
    def __init__(self, num_domains, input_shape):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_shape[1], 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, num_domains),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data):
        return self.discriminator(input_data)


def check_invariance(E, train_loader, num_iterations):
    def validation(E, D, loader):
        ds = []
        pred_ds = []
        for X, _, d in loader:
            X = Variable(X.float().cuda(), volatile=True)
            d = Variable(d.long().cuda(), volatile=True)
            pred_d = D(E(X))
            pred_d = np.argmax(pred_d.data.cpu(), axis=1)
            ds.append(d.data.cpu().numpy())
            pred_ds.append(pred_d.numpy())

        d = np.concatenate(ds)
        pred_d = np.concatenate(pred_ds)
        acc = metrics.accuracy_score(d, pred_d)
        return acc

    E.eval()
    # FIXME: We cannot measure true invariance because valid_loader.dataset = train_loader.dataset
    train_loader, valid_loader = get_joint_valid_dataloader(train_loader.dataset, 0.2, train_loader.batch_size)
    valid_loader, test_loader = get_joint_valid_dataloader(valid_loader.dataset, 0.5, valid_loader.batch_size)
    D = Discriminator(len(train_loader.dataset.domain_keys), E.output_shape()).cuda()
    optimizer = optim.RMSprop(D.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    best_valid_acc = 0.
    final_test_acc = 0.

    # ===== train =====
    for i in range(1, num_iterations + 1):
        optimizer.zero_grad()
        X, _, d = train_loader.__iter__().__next__()
        X = Variable(X.float().cuda(), volatile=True)
        d = Variable(d.long().cuda())
        z = Variable(E(X).data)
        d_pred = D(z)
        loss = criterion(d_pred, d)

        loss.backward()
        optimizer.step()

        if i % (num_iterations // 10) == 0:
            # ==== validation ====
            valid_acc = validation(E, D, valid_loader)
            test_acc = validation(E, D, test_loader)

            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                final_test_acc = test_acc

            print('(check invariance) iter: %.3f, Acc: (v)%.3f, (t)%.3f' % (i, valid_acc, test_acc))

    E.train()
    return final_test_acc, best_valid_acc


def get_activation_layer(activation, sequential):
    if activation == 'log_softmax':
        sequential.add_module('c_log_softmax', nn.LogSoftmax(dim=1))
    elif activation == 'softmax':
        sequential.add_module('c_softmax', nn.Softmax(dim=1))
    elif activation == 'relu':
        sequential.add_module('c_relu', nn.ReLU())
    elif activation is None:
        pass
    else:
        raise Exception()
    return sequential


def split_dataset(dataset, train_size=0.9):
    all_size = len(dataset)
    train_size = int(all_size * train_size)
    indices = range(all_size)
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return train_indices, test_indices


def get_joint_valid_dataloader(dataset, valid_size, batch_size):
    """ get joint domain validation dataset """
    train_indices, valid_indices = split_dataset(dataset, 1-valid_size)
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_sampler = SubsetRandomSampler(valid_indices)
    valid_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, valid_loader


def get_split_datasets(dataset, split_size):
    datasets1 = []
    datasets2 = []

    for single_dataset in dataset.datasets:
        len_data = len(single_dataset)
        len_split = int(len_data * split_size)
        indices = torch.randperm(len_data)
        dataset1, dataset2 = Subset(single_dataset, indices[:len_split]), Subset(single_dataset, indices[len_split:])
        datasets1.append(dataset1)
        datasets2.append(dataset2)

    train_dataset = dataset
    valid_dataset = copy.deepcopy(dataset)
    train_dataset.datasets = datasets1
    valid_dataset.datasets = datasets2
    train_dataset.cumulative_sizes = train_dataset.cumsum(train_dataset.datasets)
    valid_dataset.cumulative_sizes = valid_dataset.cumsum(valid_dataset.datasets)
    return train_dataset, valid_dataset


def prepare_datasets(left_out_key, validation, dataset_class, require_domain, random_state=123):
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)

    if isinstance(left_out_key, str):
        left_out_key = left_out_key.split(',')

    train_domain_keys = dataset_class.get_disjoint_domains(left_out_key)

    train_dataset = dataset_class(train_domain_keys, require_domain)
    test_dataset = dataset_class(left_out_key, require_domain)

    if validation == 'test':
        # split test dataset into validation/test dataset. Two datasets comes from the same domain.
        valid_dataset, test_dataset = get_split_datasets(test_dataset, 0.5)

    elif validation == 'train':
        # split test dataset into train/validation dataset. Two datasets comes from the same domain.
        train_dataset, valid_dataset = get_split_datasets(train_dataset, 0.8)

    elif validation == 'disjoint':
        # split test dataset into train/validation dataset. Two datasets comes from different domains.
        valid_domain_keys = [random.choice(train_domain_keys)]
        train_domain_keys = list(set(train_domain_keys) - set(valid_domain_keys))
        train_dataset = dataset_class(train_domain_keys, require_domain)
        valid_dataset = dataset_class(valid_domain_keys, require_domain)

    return train_dataset, valid_dataset, test_dataset


def cross_entropy(q, p):
    """
    :param q: (batch_size, num_class), must be normalized by softmax
    :param p: (batch_size, num_class), must be normalized by softmax
    :return:
    """
    return - torch.mean(torch.sum(q * torch.log(p + 1e-8), 1))


def D_KL(q, p):
    """
    :param q: (batch_size, num_class), must be normalized by softmax
    :param p: (batch_size, num_class), must be normalized by softmax
    :return:
    """
    return - cross_entropy(q, q) + cross_entropy(q, p)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


def calc_acc(outputs, targets):
    _, idxs = outputs.max(1)
    return (idxs == targets).float().sum().data[0] / len(targets)
