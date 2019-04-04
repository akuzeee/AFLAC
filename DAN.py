# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import metrics
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo

from learner import DAN
from utils import calc_acc, Flatten
from utils import check_invariance, get_activation_layer


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ MODEL ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
class MNISTR_Encoder(nn.Module):
    def __init__(self, input_shape):
        super(MNISTR_Encoder, self).__init__()
        _, col, row = input_shape
        latent_col = ((col - 4) - 4) / 2
        latent_row = ((row - 4) - 4) / 2
        self.latent_dim = 48 * latent_col * latent_row

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 32, kernel_size=5))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_flat', Flatten())
        self.feature.add_module('fc_fc1', nn.Linear(self.latent_dim, 100))
        self.feature.add_module('fc_relu1', nn.ReLU(True))
        self.feature.add_module('fc_fc2', nn.Linear(100, 100))
        self.feature.add_module('fc_relu2', nn.ReLU(True))

    def forward(self, input_data):
        feature = self.feature(input_data)
        return feature

    def output_shape(self):
        return (None, 100)


class MNISTR_Classifier(nn.Module):
    def __init__(self, num_classes, input_shape, activation='log_softmax'):
        super(MNISTR_Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(input_shape[1], 100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, num_classes))
        self.class_classifier = get_activation_layer(activation, self.class_classifier)

    def forward(self, input_data):
        return self.class_classifier(input_data)


class MNISTR_Discriminator(nn.Module):
    def __init__(self, num_domains, input_shape, activation='log_softmax'):
        super(MNISTR_Discriminator, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(input_shape[1], 100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, num_domains))
        self.domain_classifier = get_activation_layer(activation, self.domain_classifier)

    def forward(self, input_data):
        return self.domain_classifier(input_data)


# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ MODEL ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def get_learner(dataset, dataset_name, alpha):
    num_domains = len(dataset.domain_keys)

    if dataset_name == 'MNISTR':
        E = MNISTR_Encoder(dataset.get('input_shape'))
        M = MNISTR_Classifier(dataset.get('num_classes'), E.output_shape())
        D = MNISTR_Discriminator(num_domains, E.output_shape())

    else:
        raise NameError()

    learner = DAN(E, M, D, alpha).cuda()
    return learner


def validate(learner, loader, criterion):
    y_loss_li = []
    y_acc_li = []
    ys = []
    pred_ys = []
    for i, (X_batch, y_batch, _) in enumerate(loader):
        X_batch = Variable(X_batch.float(), volatile=True).cuda()
        y_batch = Variable(y_batch.long()).cuda()
        y_pred, _ = learner(X_batch)

        y_loss = criterion(y_pred, y_batch)
        y_acc  = calc_acc(y_pred, y_batch)

        y_loss_li.append(y_loss.data[0])
        y_acc_li.append(y_acc)

        ys.append(y_batch.cpu().data.numpy())
        pred_ys.append(np.argmax(y_pred.cpu().data.numpy(), axis=1))

    y = np.concatenate(ys)
    pred_y = np.concatenate(pred_ys)
    cm = metrics.confusion_matrix(y, pred_y)
    cm_pre = cm.astype(np.float32) / cm.sum(axis=0) * 100
    cm_rec = cm.astype(np.float32) / cm.sum(axis=1) * 100
    cm_f = (2*cm_pre*cm_rec) / (cm_pre+cm_rec)
    fvalue = np.diag(cm_f)
    return y_loss_li, y_acc_li, fvalue


def validation_step(learner, train_loader, valid_loader, test_loader, y_criterion, writer, epoch,
                    y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, n_epochs, best_valid_acc, final_test_acc,
                    d_grad_mean, acc_invariant_curve, n_checks, dataset_name_and_test_key):
    learner.eval()
    # ========== validate ==========
    y_loss_li_valid, y_acc_li_valid, fvalue = validate(learner, valid_loader, y_criterion)
    y_loss_mean_valid = np.mean(y_loss_li_valid)
    y_acc_mean_valid = np.mean(y_acc_li_valid)
    for i, f in enumerate(fvalue):
        writer.add_scalar('%s_valid_fvalue_%d' % (dataset_name_and_test_key, i), f, epoch)

    # ========== test ==========
    y_loss_li_test, y_acc_li_test, fvalue = validate(learner, test_loader, y_criterion)
    y_loss_mean_test = np.mean(y_loss_li_test)
    y_acc_mean_test = np.mean(y_acc_li_test)
    for i, f in enumerate(fvalue):
        writer.add_scalar('%s_test_fvalue_%d' % (dataset_name_and_test_key, i), f, epoch)

    # ========== check invariance =============
    if (n_epochs // 10 != 0) and (epoch % (n_epochs // 10)) == 0:
        d_test_acc, d_valid_acc = check_invariance(learner.E, train_loader, n_checks)
        writer.add_scalar('%s_d_acc_valid' % dataset_name_and_test_key, d_valid_acc, epoch)
        writer.add_scalar('%s_d_acc_test' % dataset_name_and_test_key, d_test_acc, epoch)
        acc_invariant_curve.update({d_test_acc: y_acc_mean_test})

    # ========== tracking ==========
    writer.add_scalar('%s_y_loss_train' % dataset_name_and_test_key, y_loss_mean, epoch)
    writer.add_scalar('%s_y_acc_train' % dataset_name_and_test_key, y_acc_mean, epoch)
    writer.add_scalar('%s_d_loss_train' % dataset_name_and_test_key, d_loss_mean, epoch)
    writer.add_scalar('%s_d_acc_train' % dataset_name_and_test_key, d_acc_mean, epoch)
    writer.add_scalar('%s_d_grad' % dataset_name_and_test_key, d_grad_mean, epoch)
    writer.add_scalar('%s_y_loss_valid' % dataset_name_and_test_key, y_loss_mean_valid, epoch)
    writer.add_scalar('%s_y_acc_valid' % dataset_name_and_test_key, y_acc_mean_valid, epoch)
    writer.add_scalar('%s_y_loss_test' % dataset_name_and_test_key, y_loss_mean_test, epoch)
    writer.add_scalar('%s_y_acc_test' % dataset_name_and_test_key, y_acc_mean_test, epoch)

    if best_valid_acc < y_acc_mean_valid:
        best_valid_acc = y_acc_mean_valid
        final_test_acc = y_acc_mean_test

    print("[%d/%d] train, Y: %.4f(%.3f), D: %.4f(%.3f) |  valid, Y: %.4f(%.3f) | test, Y: %.4f(%.3f)" %
          (epoch, n_epochs,
           y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean,
           y_loss_mean_valid, y_acc_mean_valid,
           y_loss_mean_test, y_acc_mean_test))

    learner.train()
    return best_valid_acc, final_test_acc


def train_simultaneously(writer, learner, lr, lr_scheduler, n_epochs, train_loader, valid_loader, test_loader,
                         y_criterion, d_criterion, best_valid_acc, final_test_acc, acc_invariant_curve,
                         weight_decay, n_checks, dataset_name_and_test_key):
    optimizer = optim.RMSprop(learner.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler:
        learner.set_lr_scheduler(optimizer, n_epochs * len(train_loader), lr)
    for epoch in range(1, n_epochs+1):
        # ====== train ======
        y_loss_li = []
        y_acc_li = []
        d_loss_li = []
        d_acc_li = []
        d_grad_li = []
        for i, (X, y, d) in enumerate(train_loader):
            optimizer.zero_grad()
            learner.scheduler_step()
            X = Variable(X.float().cuda())
            y = Variable(y.long().cuda())
            d = Variable(d.long().cuda())
            y_pred, d_pred = learner(X)

            y_loss = y_criterion(y_pred, y)
            y_acc = calc_acc(y_pred, y)
            d_loss = d_criterion(d_pred, d)
            d_acc = calc_acc(d_pred, d)
            loss = y_loss + d_loss
            loss.backward()
            d_grad_li.append(np.mean([x.grad.abs().mean().data.cpu().numpy() for x in list(learner.D.parameters())]))
            optimizer.step()

            y_loss_li.append(y_loss.data[0])
            y_acc_li.append(y_acc)
            d_loss_li.append(d_loss.data[0])
            d_acc_li.append(d_acc)
        y_loss_mean = np.mean(y_loss_li)
        y_acc_mean = np.mean(y_acc_li)
        d_loss_mean = np.mean(d_loss_li)
        d_acc_mean = np.mean(d_acc_li)
        d_grad_mean = np.mean(d_grad_li)

        best_valid_acc, final_test_acc = validation_step(learner, train_loader, valid_loader, test_loader, y_criterion,
            writer, epoch, y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, n_epochs, best_valid_acc,
            final_test_acc, d_grad_mean, acc_invariant_curve, n_checks, dataset_name_and_test_key)
    return final_test_acc


def train_alternately(writer, learner, lr, lr_scheduler, n_epochs, num_train_d, train_loader, valid_loader, test_loader,
                      y_criterion, d_criterion, best_valid_acc, final_test_acc, acc_invariant_curve,
                      weight_decay, n_checks, dataset_name_and_test_key):
    y_optimizer = optim.RMSprop(list(learner.E.parameters()) + list(learner.M.parameters()), lr=lr, weight_decay=weight_decay)
    d_optimizer = optim.RMSprop(learner.D.parameters(), lr=lr)
    if lr_scheduler:
        learner.set_lr_scheduler(y_optimizer, n_epochs * len(train_loader), lr)
    for epoch in range(1, n_epochs+1):
        # ====== train ======
        y_loss_li = []
        y_acc_li = []
        d_loss_li = []
        d_acc_li = []
        d_grad_li = []
        for i in range(len(train_loader)):
            learner.scheduler_step()
            y_optimizer.zero_grad()

            # update Domain Discriminator
            for _ in range(num_train_d):
                d_optimizer.zero_grad()
                X, _, d = train_loader.__iter__().__next__()
                X = Variable(X.float().cuda())
                d = Variable(d.long().cuda())
                d_pred = learner(X, freeze_E=True, require_class=False)
                d_loss = d_criterion(d_pred, d)
                d_acc = calc_acc(d_pred, d)
                d_loss.backward()
                d_grad_li.append(np.mean([x.grad.abs().mean().data.cpu().numpy() for x in list(learner.D.parameters())]))
                d_optimizer.step()
                d_loss_li.append(d_loss.data[0])
                d_acc_li.append(d_acc)

            # update Encoder and Classifier
            X, y, d = train_loader.__iter__().__next__()
            X = Variable(X.float().cuda())
            y = Variable(y.long().cuda())
            d = Variable(d.long().cuda())
            y_pred, d_pred = learner(X, freeze_E=False)
            y_loss = y_criterion(y_pred, y)
            d_loss = d_criterion(d_pred, d)
            loss = y_loss + d_loss
            loss.backward()
            y_optimizer.step()
            y_acc = calc_acc(y_pred, y)
            y_loss_li.append(y_loss.data[0])
            y_acc_li.append(y_acc)

        y_loss_mean = np.mean(y_loss_li)
        y_acc_mean = np.mean(y_acc_li)
        d_loss_mean = np.mean(d_loss_li)
        d_acc_mean = np.mean(d_acc_li)
        d_grad_mean = np.mean(d_grad_li)

        best_valid_acc, final_test_acc = validation_step(learner, train_loader, valid_loader, test_loader, y_criterion,
            writer, epoch, y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, n_epochs, best_valid_acc,
            final_test_acc, d_grad_mean, acc_invariant_curve, n_checks, dataset_name_and_test_key)
    return final_test_acc


def train(writer, train_dataset, valid_dataset, test_dataset, model, batch_size, dataset_name, lr, n_iter,
          lr_scheduler, alpha, num_train_d, alpha_scheduler, weight_decay, n_checks):
    learner = get_learner(train_dataset, dataset_name, alpha)
    y_criterion = nn.NLLLoss()
    d_criterion = nn.NLLLoss()
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    itr_per_epoch = len(train_loader)
    n_epochs = n_iter // itr_per_epoch
    if alpha_scheduler:
        learner.set_alpha_scheduler(n_epochs * len(train_loader), annealing_func='exp')
    best_valid_acc = 0.
    final_test_acc = 0.
    acc_invariant_curve = {}
    dataset_name_and_test_key = dataset_name + '_' + str(test_dataset.domain_keys[0])

    if model == 'DAN_sim':
        final_test_acc = train_simultaneously(writer, learner, lr, lr_scheduler, n_epochs, train_loader, valid_loader,
                                              test_loader, y_criterion, d_criterion, best_valid_acc, final_test_acc,
                                              acc_invariant_curve, weight_decay, n_checks,
                                              dataset_name_and_test_key)
    elif model == 'DAN_alt':
        final_test_acc = train_alternately(writer, learner, lr, lr_scheduler, n_epochs, num_train_d, train_loader,
                                           valid_loader, test_loader, y_criterion, d_criterion, best_valid_acc,
                                           final_test_acc, acc_invariant_curve, weight_decay, n_checks,
                                           dataset_name_and_test_key)
    else:
        raise Exception()

    for d_acc, y_acc in sorted(acc_invariant_curve.items(), key=lambda x: x[0]):
        writer.add_scalar('%s_acc_fair_curve' % dataset_name_and_test_key, y_acc, d_acc*10000)
    return final_test_acc
