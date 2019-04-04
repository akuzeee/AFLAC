# -*- coding: utf-8 -*-

import os
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Function
from torch.autograd import Variable


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradMultiplyLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        return output, None


class DAN(nn.Module):
    def __init__(self, E, M, D, alpha=1.0):
        """
        Input:
            E: encoder
            M: classifier
            D: discriminator
            alpha: weighting parameter of label classifier and domain classifier
         """
        super(DAN, self).__init__()

        self.E = E
        self.M = M
        self.D = D
        self._alpha = alpha
        self._current_step = None
        self._n_steps = None
        self._annealing_func = None
        self._lr_scheduler = None
        self.loader = None

    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return self.loader

    def get_batch(self, as_variable=True):
        assert self.loader is not None, "Please set loader before call this function"
        X, y, d = self.loader.__iter__().__next__()
        if as_variable:
            X = Variable(X.float().cuda())
            y = Variable(y.long().cuda())
            d = Variable(d.long().cuda())
        if hasattr(self.D, 'label_linear'):
            X = [X, y]
        return X, y, d

    def encode(self, input_data, freeze_E):
        if freeze_E:
            if type(input_data) is list:
                input_data = [Variable(x.data, volatile=True) for x in input_data]
            else:
                input_data = Variable(input_data.data, volatile=True)
            feature = self.E(input_data)
            feature = Variable(feature.data)
        else:
            feature = self.E(input_data)
        return feature

    def forward(self, input_data, freeze_E=False, use_reverse_layer=True, require_class=True):
        if isinstance(input_data, list):
            X, Y = input_data[0], input_data[1]
            feature = self.encode(X, freeze_E)
            feature_list = [ReverseLayerF.apply(feature, self._current_alpha), Y]
            if use_reverse_layer:
                domain_output = self.D(feature_list)
            else:
                domain_output = self.D(feature_list)
        else:
            feature = self.encode(input_data, freeze_E)
            if use_reverse_layer:
                domain_output = self.D(ReverseLayerF.apply(feature, self._current_alpha))
            else:
                domain_output = self.D(GradMultiplyLayerF.apply(feature, self._current_alpha))

        if not require_class:
            return domain_output
        class_output = self.M(feature)
        return class_output, domain_output

    def predict_y(self, input_data):
        return self.M(self.E(input_data))

    def predict_d(self, input_data):
        return self.D(self.E(input_data))

    def set_alpha_scheduler(self, n_steps, annealing_func='exp'):
        self._current_step = 0
        self._n_steps = n_steps
        self._annealing_func = annealing_func

    def alpha_scheduler_step(self):
        self._current_step += 1

    def set_lr_scheduler(self, optimizer, n_steps, lr0, lamb=None):
        if lamb is None:
            lamb = lambda current_step: lr0 / ((1 + 10 * (current_step / n_steps)) ** 0.75)
        scheduler = LambdaLR(optimizer, lr_lambda=[lamb])
        self._lr_scheduler = scheduler

    def lr_scheduler_step(self):
        self._lr_scheduler.step()

    def scheduler_step(self):
        if self._annealing_func is not None:
            self.alpha_scheduler_step()
        if self._lr_scheduler is not None:
            self.lr_scheduler_step()

    @property
    def _current_alpha(self):
        if self._annealing_func is None:
            return self._alpha
        elif self._annealing_func == 'exp':
            p = float(self._current_step) / self._n_steps
            return float((2. / (1. + np.exp(-10 * p)) - 1) * self._alpha)
        else:
            raise Exception()

    def save(self, out, prefix=None):
        names = ['E.pth', 'M.pth', 'D.pth']
        if prefix is not None:
            names = [prefix + '-' + x for x in names]
        names = [os.path.join(out, x) for x in names]

        for net, name in zip([self.E, self.M, self.D], names):
            torch.save(net.state_dict(), name)
        return names


class CIDDG(nn.Module):
    def __init__(self, E, M, D, alpha, num_classes):
        """
        Input:
            E: encoder
            M: classifier
            D: discriminator
            alpha: weighting parameter of label classifier and domain classifier
            num_classes: the number of classes
         """
        super(CIDDG, self).__init__()
        self.E = E
        self.M = M
        self.Ds = [deepcopy(D) for _ in range(num_classes)]
        self.D = deepcopy(D)
        self._alpha = alpha
        self._current_step = None
        self._n_steps = None
        self._annealing_func = None
        self._lr_scheduler = None

    def forward(self, input_data, freeze_E=False, use_reverse_layer=True, require_class=True, require_domain=True):
        if freeze_E:
            if type(input_data) is list:
                input_data = [Variable(x.data, volatile=True) for x in input_data]
            else:
                input_data = Variable(input_data.data, volatile=True)
            feature = self.E(input_data)
            feature = Variable(feature.data)
        else:
            feature = self.E(input_data)

        if require_class:
            class_output = self.M(feature)
            if not require_domain:
                return class_output

        if require_domain:
            domain_outputs = []
            for D in self.Ds:
                if use_reverse_layer:
                    domain_output = D(ReverseLayerF.apply(feature, self._current_alpha))
                else:
                    domain_output = D(GradMultiplyLayerF.apply(feature, self._current_alpha))
                domain_outputs.append(domain_output)

            if not require_class:
                return domain_outputs

        return class_output, domain_outputs

    def pred_d_by_D(self, input_data, freeze_E=False, use_reverse_layer=True):
        if freeze_E:
            if type(input_data) is list:
                input_data = [Variable(x.data, volatile=True) for x in input_data]
            else:
                input_data = Variable(input_data.data, volatile=True)
            feature = self.E(input_data)
            feature = Variable(feature.data)
        else:
            feature = self.E(input_data)

        if use_reverse_layer:
            domain_output = self.D(ReverseLayerF.apply(feature, self._current_alpha))
        else:
            domain_output = self.D(GradMultiplyLayerF.apply(feature, self._current_alpha))
        return domain_output

    def set_alpha_scheduler(self, n_steps, annealing_func='exp'):
        self._current_step = 0
        self._n_steps = n_steps
        self._annealing_func = annealing_func

    def alpha_scheduler_step(self):
        self._current_step += 1

    def set_lr_scheduler(self, optimizer, n_steps, lr0, lamb=None):
        if lamb is None:
            lamb = lambda current_step: lr0 / ((1 + 10 * (current_step / n_steps)) ** 0.75)
        scheduler = LambdaLR(optimizer, lr_lambda=[lamb])
        self._lr_scheduler = scheduler

    def lr_scheduler_step(self):
        self._lr_scheduler.step()

    def scheduler_step(self):
        if self._annealing_func is not None:
            self.alpha_scheduler_step()
        if self._lr_scheduler is not None:
            self.lr_scheduler_step()

    @property
    def _current_alpha(self):
        if self._annealing_func is None:
            return self._alpha
        elif self._annealing_func == 'exp':
            p = float(self._current_step) / self._n_steps
            return float((2. / (1. + np.exp(-10 * p)) - 1) * self._alpha)
        else:
            raise Exception()
