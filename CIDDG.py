# -*- coding: utf-8 -*-

from itertools import chain

import numpy as np
import torch
from sklearn import metrics
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data

from learner import CIDDG
from utils import calc_acc, check_invariance
from DAN import MNISTR_Encoder, MNISTR_Classifier, MNISTR_Discriminator


def mle_for_p_y_given_d(dataset):
    ys = []
    ds = []
    for _, y, d in dataset:
        ys.append(y)
        ds.append(d)
    y = np.array(ys)
    d = np.array(ds)

    num_y_keys = len(np.unique(y))
    num_d_keys = len(np.unique(d))
    p_y_given_d = torch.zeros(num_d_keys, num_y_keys)

    for d_key in np.unique(d):
        indices = np.where(d == d_key)
        y_given_key = y[indices]
        y_keys, y_counts = np.unique(y_given_key, return_counts=True)

        y_keys = torch.from_numpy(y_keys)
        y_counts = torch.from_numpy(y_counts).float()
        p_y_given_key = torch.zeros(p_y_given_d.size(1))
        p_y_given_key[y_keys] = y_counts

        p_y_given_d[d_key] = p_y_given_key

    p_y_given_d /= p_y_given_d.norm(p=1, dim=1, keepdim=True)
    return p_y_given_d


def get_learner(dataset, dataset_name, alpha):
    num_domains = len(dataset.domain_keys)

    if dataset_name == 'MNISTR':
        E = MNISTR_Encoder(dataset.get('input_shape'))
        M = MNISTR_Classifier(dataset.get('num_classes'), E.output_shape())
        D = MNISTR_Discriminator(num_domains, E.output_shape())
    else:
        raise NameError()
    D.cuda()
    learner = CIDDG(E, M, D, alpha, dataset.get('num_classes')).cuda()
    return learner


def validate(learner, loader, criterion):
    y_loss_li = []
    y_acc_li = []
    ys = []
    pred_ys = []
    for i, (X_batch, y_batch, _) in enumerate(loader):
        X_batch = Variable(X_batch.float(), volatile=True).cuda()
        y_batch = Variable(y_batch.long()).cuda()
        y_pred = learner(X_batch, require_domain=False)

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
                    y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, d_norm_loss_mean, d_norm_acc_mean, n_epochs,
                    best_valid_acc, final_test_acc, acc_invariant_curve, n_checks, dataset_name_and_test_key):
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
    writer.add_scalar('%s_d_norm_loss_train' % dataset_name_and_test_key, d_norm_loss_mean, epoch)
    writer.add_scalar('%s_d_norm_acc_train' % dataset_name_and_test_key, d_norm_acc_mean, epoch)
    writer.add_scalar('%s_y_loss_valid' % dataset_name_and_test_key, y_loss_mean_valid, epoch)
    writer.add_scalar('%s_y_acc_valid' % dataset_name_and_test_key, y_acc_mean_valid, epoch)
    writer.add_scalar('%s_y_loss_test' % dataset_name_and_test_key, y_loss_mean_test, epoch)
    writer.add_scalar('%s_y_acc_test' % dataset_name_and_test_key, y_acc_mean_test, epoch)

    if best_valid_acc < y_acc_mean_valid:
        best_valid_acc = y_acc_mean_valid
        final_test_acc = y_acc_mean_test

    print("[%d/%d] train, Y: %.4f(%.3f), D: %.4f(%.3f), D(norm): %.4f(%.3f) |  valid, Y: %.4f(%.3f) | test, Y: %.4f(%.3f)" %
          (epoch, n_epochs,
           y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, d_norm_loss_mean, d_norm_acc_mean,
           y_loss_mean_valid, y_acc_mean_valid,
           y_loss_mean_test, y_acc_mean_test))
    learner.train()
    return best_valid_acc, final_test_acc


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

    y_optimizer = optim.RMSprop(list(learner.E.parameters()) + list(learner.M.parameters()), lr=lr, weight_decay=weight_decay)
    d_optimizer = optim.RMSprop(list(chain.from_iterable([list(D.parameters()) for D in learner.Ds])) +
                                list(learner.D.parameters()), lr=lr)
    if lr_scheduler:
        learner.set_lr_scheduler(y_optimizer, n_epochs * len(train_loader), lr)
    p_y_given_d = Variable(mle_for_p_y_given_d(train_dataset)).cuda()
    L = train_dataset.get('num_classes')

    for epoch in range(1, n_epochs+1):
        # ====== train ======
        y_loss_li = []
        y_acc_li = []
        d_loss_li = []
        d_acc_li = []
        d_norm_loss_li = []
        d_norm_acc_li = []

        for i in range(len(train_loader)):
            learner.scheduler_step()
            y_optimizer.zero_grad()

            # update Domain Discriminator
            for _ in range(num_train_d):
                d_optimizer.zero_grad()
                X, y, d = train_loader.__iter__().__next__()
                X = Variable(X.float().cuda())
                y = Variable(y.float().cuda())
                d = Variable(d.long().cuda())
                d_preds = learner(X, freeze_E=True, require_class=False)
                d_loss = 0
                d_acc = 0
                for i, d_pred in enumerate(d_preds):
                    mask = y == i
                    d_pred_tmp = d_pred * mask.unsqueeze(1).float()
                    d_tmp = d * mask.long()
                    d_loss += d_criterion(d_pred_tmp, d_tmp)
                    _, idxs = d_pred_tmp.max(1)
                    d_acc += ((idxs == d_tmp).float().sum() - (mask.float()-1).abs().sum()).data[0] / len(d_tmp)

                d_pred = learner.pred_d_by_D(X, freeze_E=True)
                weight = (1 / p_y_given_d[d, y.long()]) / L
                d_norm_loss = (nn.NLLLoss(reduce=False)(d_pred, d) * weight).mean()
                _, idxs = d_pred.max(1)
                d_norm_acc = (idxs == d).float().sum() / len(d)

                (d_loss+d_norm_loss).backward()
                d_optimizer.step()
                d_loss_li.append(d_loss.data[0])
                d_acc_li.append(d_acc)
                d_norm_loss_li.append(d_norm_loss.data[0])
                d_norm_acc_li.append(d_norm_acc)

            # update Encoder and Classifier
            X, y, d = train_loader.__iter__().__next__()
            X = Variable(X.float().cuda())
            y = Variable(y.long().cuda())
            d = Variable(d.long().cuda())
            y_pred, d_preds = learner(X, freeze_E=False)
            y_loss = y_criterion(y_pred, y)
            d_loss = 0
            for i, d_pred in enumerate(d_preds):
                mask = y == i
                d_pred_tmp = d_pred * mask.unsqueeze(1).float()
                d_tmp = d * mask.long()
                d_loss += d_criterion(d_pred_tmp, d_tmp)

            d_pred = learner.pred_d_by_D(X, freeze_E=False)
            weight = (1 / p_y_given_d[d, y]) / L
            d_norm_loss = (nn.NLLLoss(reduce=False)(d_pred, d) * weight).mean()
            loss = y_loss + d_loss + d_norm_loss
            loss.backward()
            y_optimizer.step()
            y_acc = calc_acc(y_pred, y)
            y_loss_li.append(y_loss.data[0])
            y_acc_li.append(y_acc)

        y_loss_mean = np.mean(y_loss_li)
        y_acc_mean = np.mean(y_acc_li)
        d_loss_mean = np.mean(d_loss_li)
        d_acc_mean = np.mean(d_acc_li)
        d_norm_loss_mean = np.mean(d_norm_loss_li)
        d_norm_acc_mean = np.mean(d_norm_acc_li)

        best_valid_acc, final_test_acc = validation_step(learner, train_loader, valid_loader, test_loader, y_criterion,
            writer, epoch, y_loss_mean, y_acc_mean, d_loss_mean, d_acc_mean, d_norm_loss_mean, d_norm_acc_mean,
            n_epochs, best_valid_acc, final_test_acc, acc_invariant_curve, n_checks, dataset_name_and_test_key)

    for d_acc, y_acc in sorted(acc_invariant_curve.items(), key=lambda x: x[0]):
        writer.add_scalar('%s_acc_fair_curve' % dataset_name_and_test_key, y_acc, d_acc*10000)
    return final_test_acc
