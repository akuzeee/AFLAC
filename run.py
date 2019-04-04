#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random

import torch
from sacred import Experiment
from sacred import Ingredient
from sacred.observers import FileStorageObserver
from sacred.commandline_options import CommandLineOption
from tensorboardX import SummaryWriter

from dataset import BiasedMNISTR, get_biased_mnistr, MNISTR
from utils import prepare_datasets, get_split_datasets


class CustomFileStorageObserver(FileStorageObserver):
    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        if _id is None:
            # timestamp = "{:%Y-%m-%d-%H-%M-%S}_".format(start_time)
            options = '_'.join(meta_info['options']['UPDATE'][5:])
            run_id = options

            # avoid os error
            assert len(run_id) < 256

            # update the basedir of the observer
            self.basedir = os.path.join(self.basedir, run_id)
            global log_dir
            log_dir = self.basedir

            # and again create the basedir
            os.makedirs(self.basedir)
        return super(CustomFileStorageObserver, self).started_event(
            ex_info, command, host_info, start_time, config, meta_info, _id)


meta_ingredient = Ingredient('meta_cfg')
train_ingredient = Ingredient('train_cfg', ingredients=[meta_ingredient])
ex = Experiment('pytransfer', ingredients=[meta_ingredient, train_ingredient])
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')


class MyFileStorageOption(CommandLineOption):
    """Add a file-storage observer to the experiment."""
    short_flag = 'o'
    arg = 'BASEDIR'
    arg_description = "Base-directory to write the runs to"

    @classmethod
    def apply(cls, args, run):
        path = os.path.join(LOG_DIR, args)
        run.observers.append(CustomFileStorageObserver.create(path))


@meta_ingredient.config
def meta_cfg():
    seed = 0
    gpu = 0
    model = 'DAN_alt'
    dataset_name = 'MNISTR'
    biased = 1
    validation = 'train'

    if dataset_name == 'MNISTR':
        if biased == 1:
            dataset_class = BiasedMNISTR
        elif biased == 2:
            dataset_class = get_biased_mnistr(
                                                       {'0' : 1,
                                                        '15': 0.9,
                                                        '30': 0.8,
                                                        '45': 0.7,
                                                        '60': 0.6,
                                                        '75': 0.5})

        elif biased == 3:
            dataset_class = get_biased_mnistr(
                                                       {'0' : 1,
                                                        '15': 0.25,
                                                        '30': 1,
                                                        '45': 0.25,
                                                        '60': 1,
                                                        '75': 0.25})

        else:
            dataset_class = MNISTR
        test_key = '0'

    else:
        raise Exception()

    if model == 'DAN_sim' or model == 'DAN_alt':
        from DAN import train
    elif model == 'AFLAC':
        from AFLAC import train
    elif model == 'CIDDG':
        from CIDDG import train
    else:
        raise Exception()


@meta_ingredient.capture
def init_exp(seed, gpu):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.deterministic = True  # required to obtain stable results for each script run


@train_ingredient.config
def train_cfg(meta_cfg):
    dataset_name = meta_cfg['dataset_name']
    weight_decay = 0
    if meta_cfg['dataset_name'] == 'MNISTR':
        lr = 0.0005
        batch_size = 128
        n_iter = 10000
    else:
        raise Exception()

    if meta_cfg['model'] in ['DAN_sim', 'DAN_alt', 'CIDDG']:
        alpha = 0.1
        lr_scheduler = False
        alpha_scheduler = True
        num_train_d = 1
        model = meta_cfg['model']
        n_checks = 50

    elif meta_cfg['model'] in ['AFLAC']:
        alpha = 0.1
        lr_scheduler = False
        alpha_scheduler = True
        num_train_d = 1
        model = meta_cfg['model']
        p_d = 'dependent_y'  # ['dependent_y', 'independent_y']
        n_checks = 50
        num_train_e = 1
    else:
        raise Exception()


@ex.capture
def run_experiment(meta_cfg, train_cfg):
    writer = SummaryWriter(log_dir=log_dir)
    dataset_class = meta_cfg['dataset_class']
    test_key = str(meta_cfg['test_key'])
    train = meta_cfg['train']
    results = []

    train_dataset, valid_dataset, test_dataset = prepare_datasets(test_key, meta_cfg['validation'],
                                                                  dataset_class, True, meta_cfg['seed'])
    best_acc = train(writer, train_dataset, valid_dataset, test_dataset, **train_cfg)
    results.append([test_key, best_acc])

    for test_domain_key, test_acc in results:
        result = '| %s | %.3f |' % (test_domain_key, test_acc)
        print(result)
        writer.add_scalar(meta_cfg['dataset_name'] + '_' + test_domain_key, test_acc, 0)


@ex.automain
def main():
    init_exp()
    run_experiment()
