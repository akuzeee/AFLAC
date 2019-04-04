# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data

from utils import D_KL, calc_acc, check_invariance
from DAN import validate, MNISTR_Encoder, MNISTR_Classifier, MNISTR_Discriminator
from learner import DAN


def mle_for_p_d_given_y(dataset):
    ys = []
    ds = []
    for _, y, d in dataset:
        ys.append(y)
        ds.append(d)
    y = np.array(ys)
    d = np.array(ds)

    num_y_keys = len(np.unique(y))
    num_d_keys = len(np.unique(d))
    p_d_given_y = torch.zeros(num_y_keys, num_d_keys)

    for y_key in np.unique(y):
        indices = np.where(y == y_key)
        d_given_key = d[indices]
        d_keys, d_counts = np.unique(d_given_key, return_counts=True)

        d_keys = torch.from_numpy(d_keys)
        d_counts = torch.from_numpy(d_counts).float()
        p_d_given_key = torch.zeros(p_d_given_y.size(1))
        p_d_given_key[d_keys] = d_counts

        p_d_given_y[y_key] = p_d_given_key

    p_d_given_y /= p_d_given_y.norm(p=1, dim=1, keepdim=True)
    return p_d_given_y


def mle_for_p_d(dataset):
    ds = []
    for _, _, d in dataset:
        ds.append(d)
    d = np.array(ds)

    d_keys, d_counts = np.unique(d, return_counts=True)
    d_keys = torch.from_numpy(d_keys)
    d_counts = torch.from_numpy(d_counts).float()
    p_d = torch.zeros(d_keys.size(0))
    p_d[d_keys] = d_counts
    p_d /= p_d.norm(p=1, dim=0, keepdim=True)
    return p_d


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


def validation_step(learner, train_loader, valid_loader, test_loader, y_criterion, writer, epoch, y_loss_mean,
                    y_acc_mean, d_loss_mean, d_acc_mean, kl_loss_mean, n_epochs, best_valid_acc, final_test_acc,
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
    writer.add_scalar('%s_kl_loss_train' % dataset_name_and_test_key, kl_loss_mean, epoch)
    writer.add_scalar('%s_d_grad' % dataset_name_and_test_key, d_grad_mean, epoch)
    writer.add_scalar('%s_y_loss_valid' % dataset_name_and_test_key, y_loss_mean_valid, epoch)
    writer.add_scalar('%s_y_acc_valid' % dataset_name_and_test_key, y_acc_mean_valid, epoch)
    writer.add_scalar('%s_y_loss_test' % dataset_name_and_test_key, y_loss_mean_test, epoch)
    writer.add_scalar('%s_y_acc_test' % dataset_name_and_test_key, y_acc_mean_test, epoch)

    if best_valid_acc < y_acc_mean_valid:
        best_valid_acc = y_acc_mean_valid
        final_test_acc = y_acc_mean_test

    print("[%d/%d] train, Y: %.4f(%.3f), D: %.4f(%.3f), KL: %.4f |  valid, Y: %.4f(%.3f) | test, Y: %.4f(%.3f)" %
          (epoch, n_epochs,
           y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, kl_loss_mean,
           y_loss_mean_valid, y_acc_mean_valid,
           y_loss_mean_test, y_acc_mean_test))

    learner.train()
    return best_valid_acc, final_test_acc


def train(writer, train_dataset, valid_dataset, test_dataset, model, batch_size, dataset_name, lr, n_iter,
          lr_scheduler, alpha, num_train_d, num_train_e, alpha_scheduler, p_d, weight_decay, n_checks):
    learner = get_learner(train_dataset, dataset_name, alpha)
    y_criterion = nn.NLLLoss()
    d_criterion = nn.NLLLoss()
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    itr_per_epoch = len(train_loader)
    n_epochs = n_iter // itr_per_epoch
    best_valid_acc = 0.
    final_test_acc = 0.
    acc_invariant_curve = {}
    dataset_name_and_test_key = dataset_name + '_' + 'wisdm'

    y_optimizer = optim.RMSprop(list(learner.E.parameters()) + list(learner.M.parameters()), lr=lr, weight_decay=weight_decay)
    d_optimizer = optim.RMSprop(learner.D.parameters(), lr=lr)
    if lr_scheduler:
        learner.set_lr_scheduler(y_optimizer, n_epochs * len(train_loader), lr)
    if alpha_scheduler:
        learner.set_alpha_scheduler(n_epochs * len(train_loader), annealing_func='exp')
    p_d_given_y = Variable(mle_for_p_d_given_y(train_dataset)).cuda()
    p_d_not_given_y = Variable(mle_for_p_d(train_dataset)).cuda()

    # ====== train ======
    for epoch in range(1, n_epochs+1):
        y_loss_li = []
        y_acc_li = []
        d_loss_li = []
        kl_loss_li = []
        d_acc_li = []
        d_grad_li = []
        for i in range(len(train_loader)):
            learner.scheduler_step()

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
            for _ in range(num_train_e):
                y_optimizer.zero_grad()
                X, y, d = train_loader.__iter__().__next__()
                X = Variable(X.float().cuda())
                y = Variable(y.long().cuda())
                y_pred, d_pred = learner(X, use_reverse_layer=False, freeze_E=False)
                d_pred = torch.exp(d_pred)
                y_loss = y_criterion(y_pred, y)
                if p_d == 'dependent_y':
                    d_true = p_d_given_y[y]
                elif p_d == 'independent_y':
                    d_true = p_d_not_given_y.expand_as(d_pred)
                else:
                    raise Exception()
                kl_loss = D_KL(d_true, d_pred)
                loss = y_loss + kl_loss
                loss.backward()
                y_optimizer.step()
            y_acc = calc_acc(y_pred, y)
            y_loss_li.append(y_loss.data[0])
            kl_loss_li.append(kl_loss.data[0])
            y_acc_li.append(y_acc)

        y_loss_mean = np.mean(y_loss_li)
        y_acc_mean = np.mean(y_acc_li)
        d_loss_mean = np.mean(d_loss_li)
        d_acc_mean = np.mean(d_acc_li)
        d_grad_mean = np.mean(d_grad_li)
        kl_loss_mean = np.mean(kl_loss_li)

        best_valid_acc, final_test_acc = validation_step(learner, train_loader, valid_loader, test_loader, y_criterion,
            writer, epoch, y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, kl_loss_mean, n_epochs, best_valid_acc,
            final_test_acc, d_grad_mean, acc_invariant_curve, n_checks, dataset_name_and_test_key)

    for d_acc, y_acc in sorted(acc_invariant_curve.items(), key=lambda x: x[0]):
        writer.add_scalar('%s_acc_fair_curve' % dataset_name_and_test_key, y_acc, d_acc*10000)

    return final_test_acc
