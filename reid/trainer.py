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
        losses_rehearsal = AverageMeter()
        losses_refresh = AverageMeter()
        
        end = time.time()

        for i in range(train_iters):

            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            imgs, targets, cids, domains = self._parse_data(train_inputs)
            targets += add_num
            
            features, bn_features, cls_out = self.model(imgs, domains, training_phase)
            loss_ce, loss_tp = self._forward(features, cls_out, targets)

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())

            loss_rehearsal = loss_ce + loss_tp

            if replay is True:
                imgs_r, fnames_r, pid_r, cid_r, domain_r = next(iter(data_loader_replay))
                imgs_r = imgs_r.cuda()
                pid_r = pid_r.cuda()

                features_r, bn_features_r, cls_out_r = self.model(imgs_r, domain_r, training_phase)

                loss_tr_r = self.trip_hard(features_r, pid_r)[0] 

                loss_rehearsal += loss_tr_r 

                features_old, bn_features_old, cls_out_old = old_model(imgs, domains, training_phase)
                features_r_old, bn_features_r_old, cls_out_r_old = old_model(imgs_r, domain_r, training_phase)
                
                loss_memo = self.criterion_ce(cls_out_old, targets) + self.criterion_triple(features_old, features_old, targets) + self.trip_hard(features_r_old, pid_r)[0] 
                loss_cali= self.loss_kd_js(cls_out, cls_out_old)
                loss_refresh = loss_memo + loss_cali

                losses_refresh.update(loss_refresh)

                old_optimizer.zero_grad()
                loss_refresh.backward()
                old_optimizer.step()
                del bn_features, bn_features_old
                
                loss_rehearsal += self.loss_kd_js(cls_out_old, cls_out)
                del cls_out_old, cls_out, cls_out_r_old, cls_out_r, bn_features_r, bn_features_r_old, features_old, features
            
            losses_rehearsal.update(loss_rehearsal)
            
            optimizer.zero_grad()
            loss_rehearsal.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) == train_iters or (i + 1)%(train_iters//4)==0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_rehearsal {:.3f} ({:.3f})\t'
                      'Loss_refresh {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_rehearsal.val, losses_rehearsal.avg,
                              losses_refresh.val, losses_refresh.avg))

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