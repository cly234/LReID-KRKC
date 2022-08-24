from __future__ import print_function, absolute_import
import time

import torch.nn as nn
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, \
    CrossEntropyLabelSmooth_weighted, SoftTripletLoss_weight
from .utils.meters import AverageMeter
from .utils.my_tools import *
from reid.metric_learning.distance import cosine_similarity, cosine_distance


class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_ce_weight = CrossEntropyLabelSmooth_weighted(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.criterion_triple_weight = SoftTripletLoss_weight(margin=margin).cuda()
        self.trip_hard = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_train, data_loader_replay, optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None,replay=False, num_mark=None, num_vip=None, back_distill=False):

        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_kd_r = AverageMeter()
        #losses_DCL = AverageMeter()
        #losses_PT_ID = AverageMeter()
        #losses_PT_KD = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            if(back_distill==False):
                train_inputs = data_loader_train.next()
                data_time.update(time.time() - end)

                s_inputs, targets, cids, domains = self._parse_data(train_inputs)
                targets += add_num

                s_features, bn_features, s_cls_out = self.model(s_inputs, domains, training_phase)

                # We-ID
                loss_ce, loss_tp = self._forward(s_features, s_cls_out, targets)

                loss = loss_ce + loss_tp

            if replay is True:
                imgs_r, fnames_r, pid_r, cid_r, domain_r = next(iter(data_loader_replay))
                imgs_r = imgs_r.cuda()
                pid_r = pid_r.cuda()
                features_r, bn_features_r, cls_out_r = \
                    self.model(imgs_r, domain_r, training_phase)

                #loss_tr_r = self.trip_hard(features_r, pid_r)[0]
                loss_ce_r, loss_tp_r = self._forward(features_r, cls_out_r, pid_r)

                loss += (loss_ce_r + loss_tp_r) 
                with torch.no_grad():
                    old_features_r, old_features_bn_r, old_logits_r = \
                        old_model(imgs_r, domain_r, training_phase,fkd=True)
                assert cls_out_r.shape[1] == old_logits_r.shape[1] ==  num_mark + num_vip 

                KD_loss_r = self.loss_kd_logit(cls_out_r[:num_vip], old_logits_r[num_vip])

                del s_features, bn_features, s_cls_out, features_r, bn_features_r, cls_out_r, old_features_r, old_features_bn_r, old_logits_r
                
                losses_kd_r.update(KD_loss_r.item())
                loss += KD_loss_r

                '''
                # PT-KD
                loss_PT_KD = self.PT_KD(old_fake_feat_list_r[:(training_phase-1)], fake_feat_list_r[:(training_phase-1)])
                losses_PT_KD.update(loss_PT_KD.item())
                loss += loss_PT_KD

                # PT-ID
                loss_PT_ID = self.PT_ID(fake_feat_list, bn_features, targets)
                losses_PT_ID.update(loss_PT_ID.item())
                loss += loss_PT_ID
                '''

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #for bn in self.model.module.task_specific_batch_norm:
                #bn.weight.data.copy_(self.model.module.bottleneck.weight.data)

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

    def forward_weight(self, s_features, s_outputs, targets, weigths):
        loss_ce = self.criterion_ce_weight(s_outputs, targets, weigths)
        loss_tr = self.criterion_triple_weight(s_features, s_features, targets, weigths)
        return loss_ce, loss_tr

    def DCL(self, features, feature_list_bn, pids):
        loss = []
        uniq_pid = torch.unique(pids)
        for pid in uniq_pid:
            pid_index = torch.where(pid == pids)[0]
            global_bn_feat_single = features[pid_index]
            for feats in feature_list_bn:
                speci_bn_feat_single = feats[pid_index]
                distance_matrix = -torch.mm(F.normalize(global_bn_feat_single, p=2, dim=1),
                                            F.normalize(speci_bn_feat_single, p=2, dim=1).t().detach())
                loss.append(torch.mean(distance_matrix))
        loss = torch.mean(torch.stack(loss))
        return loss

    def loss_kd_L1(self, new_features, old_features):

        L1 = torch.nn.L1Loss()

        old_simi_matrix = cosine_distance(old_features, old_features)
        new_simi_matrix = cosine_distance(new_features, new_features)

        simi_loss = L1(old_simi_matrix, new_simi_matrix)

        return simi_loss

    def PT_KD(self, fake_feat_list_old, fake_feat_list_new):
        loss_cross = []
        for i in range(len(fake_feat_list_old)):
            for j in range(i, len(fake_feat_list_old)):
                loss_cross.append(self.loss_kd_L1(fake_feat_list_old[i], fake_feat_list_new[j]))
        loss_cross = torch.mean(torch.stack(loss_cross))
        return loss_cross

    def loss_kd_logit(new_logits, old_logits):
        logsoftmax = nn.LogSoftmax(dim=1).cuda()
 
        loss_ke_ce = (- F.softmax(old_logits, dim=1).detach() * logsoftmax(new_logits)).mean(0).sum() 
        return loss_ke_ce 
        
    def loss_kd_old(self, new_features, old_features, new_logits, old_logits):

        logsoftmax = nn.LogSoftmax(dim=1).cuda()

        L1 = torch.nn.L1Loss()

        old_simi_matrix = cosine_distance(old_features, old_features)
        new_simi_matrix = cosine_distance(new_features, new_features)

        simi_loss = 0.5*L1(old_simi_matrix, new_simi_matrix) 
        loss_ke_ce = 0.5*(- F.softmax(old_logits, dim=1).detach() * logsoftmax(new_logits)).mean(0).sum() 
        return loss_ke_ce + simi_loss

    def loss_kd_old_mutual(self, new_features, old_features, new_logits, old_logits):

        logsoftmax = nn.LogSoftmax(dim=1).cuda()

        L1 = torch.nn.L1Loss()

        old_simi_matrix = cosine_distance(old_features, old_features)
        new_simi_matrix = cosine_distance(new_features, new_features)

        simi_loss = 0.5*L1(old_simi_matrix.detach(), new_simi_matrix) + 0.5*L1(old_simi_matrix, new_simi_matrix.detach())
        loss_ke_ce = 0.5*(- F.softmax(old_logits, dim=1).detach() * logsoftmax(new_logits)).mean(0).sum() + 0.5*(- F.softmax(new_logits, dim=1).detach() * logsoftmax(old_logits)).mean(0).sum()

        return loss_ke_ce + simi_loss

    def PT_ID(self, feature_list_bn, bn_features, pids):

        loss = []
        for features in feature_list_bn:
            loss.append(self.trip_hard(features, pids)[0])
        loss.append(self.trip_hard(bn_features, pids)[0])
        loss = torch.mean(torch.stack(loss))

        loss_cross = []
        for i in range(len(feature_list_bn)):
            for j in range(i + 1, len(feature_list_bn)):
                loss_cross.append(self.trip_hard(feature_list_bn[i], pids, feature_list_bn[j]))
        loss_cross = torch.mean(torch.stack(loss_cross))
        loss = 0.5 * (loss + loss_cross)

        return loss



