import torch
import torch.nn.functional as F

from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from .data.sampler import RandomIdentitySampler, MultiDomainRandomIdentitySampler

import collections
import numpy as np
import copy

def extract_features(model, data_loader):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            features = model(imgs)
            for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
    model.train()
    return features_all, labels_all, fnames_all, camids_all


def initial_classifier(model, data_loader):
    pid2features = collections.defaultdict(list)
    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)
    for feature, pid in zip(features_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    return F.normalize(class_centers, dim=1).float().cuda()

def select_replay_samples(model, dataset, training_phase=0, add_num=0, old_datas=None, select_samples=2):
    replay_data = []
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_loader = DataLoader(Preprocessor(dataset.train, root=dataset.images_dir,transform=transformer),
                              batch_size=128, num_workers=4, shuffle=True, pin_memory=True, drop_last=False)

    features_all, labels_all, fnames_all, camids_all = extract_features(model, train_loader)

    pid2features = collections.defaultdict(list)
    pid2fnames = collections.defaultdict(list)
    pid2cids = collections.defaultdict(list)

    for feature, pid, fname, cid in zip(features_all, labels_all, fnames_all, camids_all):
        pid2features[pid].append(feature)
        pid2fnames[pid].append(fname)
        pid2cids[pid].append(cid)

    labels_all = list(set(labels_all))

    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = F.normalize(torch.stack(class_centers), dim=1)
    select_pids = np.random.choice(labels_all, 250, replace=True)
    for pid in select_pids:
        feautures_single_pid = F.normalize(torch.stack(pid2features[pid]), dim=1, p=2)
        center_single_pid = class_centers[pid]
        simi = torch.mm(feautures_single_pid, center_single_pid.unsqueeze(0).t())
        simi_sort_inx = torch.sort(simi, dim=0)[1][:2]
        for id in simi_sort_inx:
            replay_data.append((pid2fnames[pid][id], pid+add_num, pid2cids[pid][id], training_phase-1))

    if old_datas is None:
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=128,num_workers=8, sampler=RandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)
    else:
        replay_data.extend(old_datas)
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=128,num_workers=8,
                             sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)

    return data_loader_replay, replay_data


def eval_func(epoch, evaluator, model, test_loader, name, old_model=None, use_fsc=True):
    evaluator.reset()
    model.eval()
    if old_model is not None:
        evaluator_old = copy.deepcopy(evaluator)
        evaluator_fuse = copy.deepcopy(evaluator)
        evaluator_old.reset()
        evaluator_fuse.reset()
        old_model.eval()
    device = 'cuda'
    pid_list = []
    for n_iter, (imgs, fnames, pids, cids, domians) in enumerate(test_loader):
        with torch.no_grad():
            pid_list.append(pids)
            imgs = imgs.to(device)
            cids = cids.to(device)
            feat = model(imgs)
            if old_model is not None:
                old_feat = old_model(imgs)
                fuse_feat = torch.cat([feat, old_feat], dim=1)

            evaluator.update((feat, pids, cids))
            if old_model is not None:
                evaluator_old.update((old_feat, pids, cids))
                evaluator_fuse.update((fuse_feat, pids, cids))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if old_model is not None:
        cmc_old, mAP_old, _, _, _, _, _ = evaluator_old.compute()
        cmc_fuse, mAP_fuse, _, _, _, _, _ = evaluator_fuse.compute()

    if not use_fsc:
        print("Validation Results - Epoch: {}".format(epoch))
        print("mAP_{}: {:.1%}".format(name, mAP))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()
    
        return cmc, mAP

    if old_model is not None:
        print("Validation Results - Epoch: {}".format(epoch))
        print("mAP_{}: {:.1%}".format(name+"_FSC", mAP_fuse))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_fuse[r - 1]))
        torch.cuda.empty_cache()

        return cmc, mAP, mAP_old, mAP_fuse
    
