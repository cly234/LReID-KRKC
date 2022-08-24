from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

from .evaluation_metrics import cmc, mean_ap, mean_ap_cuhk03
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking


def extract_features(model, data_loader):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    pid_list = []
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domians) in enumerate(data_loader):
            pid_list.append(pids)
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels

def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m
    

    x = torch.cat([features[f].unsqueeze(0) for f, _, _,_ in query], 0)
    
    y = torch.cat([features[f].unsqueeze(0) for f, _, _,_ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, cuhk03=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _,_ in query]
        gallery_ids = [pid for _, pid, _,_ in gallery]
        query_cams = [cam for _, _, cam,_ in query]
        gallery_cams = [cam for _, _, cam,_ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    if cuhk03:
        mAP = mean_ap_cuhk03(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    else:
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
        'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    if cuhk03:
        print('CUHK03 CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                .format(k,
                        cmc_scores['cuhk03'][k-1]))
        return cmc_scores['cuhk03'][0], mAP
    
    else:
        print('CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                .format(k,
                        cmc_scores['market1501'][k-1]))
        return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False,
                 rerank=False, pre_features=None, cuhk03=False):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader)
        else:
            features = pre_features
        #print("ddddddddddddddddddd")
        #print([query_i[1] for query_i in query])
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, cuhk03=cuhk03)
        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq = pairwise_distance(features, query, query, metric=metric)
        distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

