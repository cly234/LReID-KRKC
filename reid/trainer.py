from __future__ import print_function, absolute_import
import time

import torch.nn as nn
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss
from .utils.meters import AverageMeter
from .utils.my_tools import *
import numpy as np
from torch.nn import functional as F

class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.trip_hard = TripletLoss(margin=margin).cuda()
        self.T=2

    def train(self, epoch, data_loader_train, data_loader_replay, optimizer, old_optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None,replay=False):
        self.model.train()
        if old_model is not None:
            old_model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_tr_r = AverageMeter()
        losses_kd_r = AverageMeter()
        #losses_DCL = AverageMeter()
        #losses_PT_ID = AverageMeter()
        #losses_PT_KD = AverageMeter()

        end = time.time()

        for i in range(train_iters):

            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains = self._parse_data(train_inputs)
            targets += add_num
            
            s_features, bn_features, s_cls_out = self.model(s_inputs, domains, training_phase)
            loss_ce, loss_tp = self._forward(s_features, s_cls_out, targets)

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())

            loss = loss_ce + loss_tp

            if replay is True:
                imgs_r, fnames_r, pid_r, cid_r, domain_r = next(iter(data_loader_replay))
                imgs_r = imgs_r.cuda()
                pid_r = pid_r.cuda()

                features_r, bn_features_r, cls_out_r = \
                    self.model(imgs_r, domain_r, training_phase)

                loss_tr_r = self.trip_hard(features_r, pid_r)[0] 

                loss += loss_tr_r 

                s_features_old, bn_features_old, s_cls_out_old = old_model(s_inputs, domains, training_phase)
                features_r_old, bn_features_r_old, cls_out_r_old = old_model(imgs_r, domain_r, training_phase,fkd=True)
                KD_loss_r = self.criterion_ce(s_cls_out_old, targets) + self.criterion_triple(s_features_old, s_features_old, targets) + self.trip_hard(features_r_old, pid_r)[0] 

                KD_loss_r += self.loss_kd_js(s_cls_out, s_cls_out_old)
               
                old_optimizer.zero_grad()
                KD_loss_r.backward()
                old_optimizer.step()
                del bn_features, bn_features_old
                
                loss += self.loss_kd_js(s_cls_out_old, s_cls_out)
                del s_cls_out_old, s_cls_out, cls_out_r_old, cls_out_r, bn_features_r, bn_features_r_old, s_features_old, s_features
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) == train_iters or (i + 1)%(train_iters//4)==0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'
                      'Loss_sce {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_kd_r.val, losses_kd_r.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        return loss_ce, loss_tr

    def loss_kd_js(self, old_logits, new_logits):
        old_logits = old_logits.detach()
        p_s = F.log_softmax((new_logits + old_logits)/(2*self.T), dim=1)
        p_t = F.softmax(old_logits/self.T, dim=1)
        p_t2 = F.softmax(new_logits/self.T, dim=1)
        loss = 0.5*F.kl_div(p_s, p_t, reduction='batchmean')*(self.T**2) + 0.5*F.kl_div(p_s, p_t2, reduction='batchmean')*(self.T**2)

        return loss