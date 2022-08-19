from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import copy
import torch.nn as nn
import random

from reid import datasets
from reid.evaluators import Evaluator
from reid.utils.metrics import R1_mAP_eval
from reid.utils.data import IterLoader
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.my_tools import *
from reid.models.resnet import build_resnet_backbone
from reid.models.layers import DataParallel
from reid.trainer import Trainer
from torch.nn.parallel import DistributedDataParallel
import copy

def eval_func(epoch, evaluator, model, test_loader, name, old_model=None):
    evaluator_old = copy.deepcopy(evaluator)
    evaluator_both = copy.deepcopy(evaluator)
    evaluator.reset()
    evaluator_old.reset()
    evaluator_both.reset()
    model.eval()
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
                both_feat = torch.cat([feat, old_feat], dim=1)

            evaluator.update((feat, pids, cids))
            evaluator_old.update((old_feat, pids, cids))
            evaluator_both.update((both_feat, pids, cids))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    cmc_old, mAP_old, _, _, _, _, _ = evaluator_old.compute()
    cmc_both, mAP_both, _, _, _, _, _ = evaluator_both.compute()

    print("Validation Results - Epoch: {}".format(epoch))
    print("mAP_{}: {:.1%}".format(name, mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    torch.cuda.empty_cache()

    print("Validation Results - Epoch: {}".format(epoch))
    print("mAP_{}: {:.1%}".format(name+"old", mAP_old))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_old[r - 1]))
    torch.cuda.empty_cache()

    print("Validation Results - Epoch: {}".format(epoch))
    print("mAP_{}: {:.1%}".format(name+"both", mAP_both))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_both[r - 1]))
    torch.cuda.empty_cache()

    return cmc, mAP

def get_data(name, data_dir, height, width, batch_size, workers, num_instances):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(dataset.query + dataset.gallery),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_loader, test_loader, init_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):

    cudnn.benchmark = True
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_viper, num_classes_viper, train_loader_viper, test_loader_viper, _ = \
        get_data('viper', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
   
    dataset_market, num_classes_market, train_loader_market, test_loader_market, init_loader_market = \
        get_data('market1501', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_prid, num_classes_prid, train_loader_prid, test_loader_prid, init_loader_prid = \
        get_data('prid', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhksysu, num_classes_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk_sysu', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_msmt17, num_classes_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    '''
    # Data loaders for test only
    dataset_cuhk03, _, _, test_loader_cuhk03, _ = \
        get_data('cuhk03', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    
    dataset_cuhk01, _, _, test_loader_cuhk01, _ = \
        get_data('cuhk01', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_grid, _, _, test_loader_grid, _ = \
        get_data('grid', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_sense, _, _, test_loader_sense, _ =\
        get_data('sense', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    '''
    # Create model
    model = build_resnet_backbone(num_class=num_classes_viper, depth='50x')
    model.cuda()
    model = DataParallel(model)

    # Evaluator
    start_epoch = 0
    evaluator = Evaluator(model)

    # Opitimizer initialize
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, [40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Start training
    print('Continual training starts!')

    # Train Market-1501
    trainer = Trainer(model, num_classes_viper, margin=args.margin)
    for epoch in range(start_epoch, 60):

        train_loader_viper.new_epoch()
        trainer.train(epoch, train_loader_viper, None, optimizer, old_optimizer=None, training_phase=1,
                      train_iters=150, add_num=0, old_model=None, replay=False)
        lr_scheduler.step()

        if ((epoch + 1) % 60 == 0):
            _, mAP = evaluator.evaluate(test_loader_viper, dataset_viper.query, dataset_viper.gallery, cmc_flag=True)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, 'viper_checkpoint_v1.pth.tar'))

            print('Finished epoch {:3d}  VIPeR mAP: {:5.1%} '.format(epoch, mAP))
    # Select replay data of market-1501
    replay_dataloader, viper_replay_dataset = select_replay_samples(model, dataset_viper, training_phase=1)

    # Expand the dimension of classifier
    org_classifier_params = model.module.classifier.weight.data
    model.module.classifier = nn.Linear(2048, num_classes_market + num_classes_viper, bias=False)
    model.cuda()
    model.module.classifier.weight.data[:num_classes_viper].copy_(org_classifier_params)
    add_num = num_classes_viper

    # Initialize classifer with class centers
    class_centers = initial_classifier(model, init_loader_market)
    model.module.classifier.weight.data[num_classes_viper:].copy_(class_centers)

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.train()

    num_query = len(dataset_market.query)
    evaluator_market = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)
    evaluator_viper = R1_mAP_eval(len(dataset_viper.query), max_rank=50, feat_norm=True)
    evaluators = [evaluator_viper, evaluator_market]
    names = ['viper_norm', 'market_norm']
    test_loaders = [test_loader_viper, test_loader_market]

    evaluator_viper.reset()
    model.eval()
    device = 'cuda'
    pid_list = []
    for n_iter, (imgs, fnames, pids, cids, domians) in enumerate(test_loader_viper):
        with torch.no_grad():
            pid_list.append(pids)
            imgs = imgs.to(device)
            cids = cids.to(device)
            feat = model(imgs)
            evaluator_viper.update((feat, pids, cids))

    cmc, mAP_viper, _, _, _, _, _ = evaluator_viper.compute()
    print("*******************")
    print(mAP_viper)
 
    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": 1*args.lr , "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)

    old_params = []
    for key, value in old_model.named_params(old_model):
        if not value.requires_grad:
            continue
        old_params += [{"params": [value], "lr": 0.1*args.lr , "weight_decay": args.weight_decay}]
    old_optimizer = torch.optim.Adam(old_params)

    lr_scheduler = WarmupMultiStepLR(optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    old_lr_scheduler = WarmupMultiStepLR(old_optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    trainer = Trainer(model, num_classes_market + num_classes_viper, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):

        train_loader_market.new_epoch()
        trainer.train(epoch, train_loader_market, replay_dataloader, optimizer, old_optimizer, training_phase=2,
                      train_iters=len(train_loader_market), add_num=add_num, old_model=old_model, replay=True)
        lr_scheduler.step()
        old_lr_scheduler.step()

        if (epoch == 0 or epoch == 1 or epoch == args.epochs-1):
            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                cmc, mAP_market = eval_func(epoch, evaluator, model, test_loader, name, old_model)
        
        if epoch==args.epochs - 1:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP_market,
                }, True, fpath=osp.join(args.logs_dir, 'market_checkpoint_bilearn_daxiao.pth.tar'))

    # Select replay data of market-1501
    replay_dataloader, market_replay_dataset = select_replay_samples(model, dataset_market, training_phase=2,
                                                  add_num=num_classes_viper, old_datas=viper_replay_dataset)

    model = model.module
    old_model = old_model.module
    alpha = 1/2
    tmp_state_dict = model.state_dict()

    for k in model.state_dict().keys():
        tmp_state_dict[k] = alpha * model.state_dict()[k] + (1-alpha) * old_model.state_dict()[k]
    
    model.load_state_dict(tmp_state_dict)

    for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
        cmc, mAP_cuhk = eval_func(epoch, evaluator, model, test_loader, name, old_model)

    model = DataParallel(model)

    # Expand the dimension of classifier
    org_classifier_params = model.module.classifier.weight.data
    model.module.classifier = nn.Linear(2048, num_classes_viper + num_classes_market + num_classes_cuhksysu, bias=False)
    model.module.classifier.weight.data[:(num_classes_viper + num_classes_market)].copy_(org_classifier_params)
    model.cuda()
    add_num = num_classes_market + num_classes_viper

    # Initialize classifer with class centers
    class_centers = initial_classifier(model, init_loader_chuksysu)
    model.module.classifier.weight.data[(num_classes_market + num_classes_viper):].copy_(class_centers)
    model.cuda()

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.train()
    
    evaluator_cuhksysu_norm = R1_mAP_eval(len(dataset_cuhksysu.query), max_rank=50, feat_norm=True)
    #evaluator_cuhksysu = R1_mAP_eval(len(dataset_cuhksysu.query), max_rank=50, feat_norm=False)
    test_loaders.append(test_loader_cuhksysu) 
    evaluators.append(evaluator_cuhksysu_norm)
    names.append('cuhksysu_norm')
    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr * 1, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)

    old_params = []
    for key, value in old_model.named_params(old_model):
        if not value.requires_grad:
            continue
        old_params += [{"params": [value], "lr": 0.1*args.lr , "weight_decay": args.weight_decay}]
    old_optimizer = torch.optim.Adam(old_params)

    lr_scheduler = WarmupMultiStepLR(optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    old_lr_scheduler = WarmupMultiStepLR(old_optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    trainer = Trainer(model, num_classes_cuhksysu + add_num, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):

        train_loader_cuhksysu.new_epoch()
        trainer.train(epoch, train_loader_cuhksysu, replay_dataloader, optimizer, old_optimizer, training_phase=3,
                      train_iters=len(train_loader_cuhksysu), add_num=add_num, old_model=old_model, replay=True)
        lr_scheduler.step()
        old_lr_scheduler.step()

        if (epoch == 0 or epoch ==1 or epoch == args.epochs-1):

            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                cmc, mAP_cuhk = eval_func(epoch, evaluator, model, test_loader, name, old_model)
            
            if epoch == args.epochs - 1:
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'mAP': mAP_cuhk,
                }, True, fpath=osp.join(args.logs_dir, 'cuhksysu_checkpoint_bilearn_daxiao.pth.tar'))

            print('Finished epoch {:3d}  CUHKSYSU mAP: {:5.1%}'.format(epoch, mAP_cuhk))
    
    
    replay_dataloader, cuhksysu_replay_dataset = select_replay_samples(model, dataset_cuhksysu, training_phase=3, 
                                                   add_num=add_num, old_datas=market_replay_dataset)


    # Expand the dimension of classifier
    #org_classifier_params = model.module.classifier.weight.data
    #model.module.classifier = nn.Linear(2048, num_classes_viper + num_classes_market + num_classes_cuhksysu + num_classes_msmt17, bias=False)
    #model.module.classifier.weight.data[:(num_classes_viper + num_classes_market + num_classes_cuhksysu)].copy_(org_classifier_params)
    #odel.cuda()
    #add_num = num_classes_market + num_classes_viper + num_classes_cuhksysu

    # Initialize classifer with class centers
    #class_centers = initial_classifier(model, init_loader_msmt17)
    #model.module.classifier.weight.data[(num_classes_market + num_classes_viper +num_classes_cuhksysu):].copy_(class_centers)
    #model.cuda()

    model = model.module
    old_model = old_model.module
    alpha = 1/3
    tmp_state_dict = model.state_dict()

    for k in model.state_dict().keys():
        tmp_state_dict[k] = alpha * model.state_dict()[k] + (1-alpha) * old_model.state_dict()[k]
    
    model.load_state_dict(tmp_state_dict)

    for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
        cmc, mAP_cuhk = eval_func(epoch, evaluator, model, test_loader, name, old_model)

    model = DataParallel(model)

    org_classifier_params = model.module.classifier.weight.data
    model.module.classifier = nn.Linear(2048, num_classes_viper + num_classes_market + num_classes_cuhksysu + num_classes_msmt17, bias=False)
    model.module.classifier.weight.data[:(num_classes_viper + num_classes_market + num_classes_cuhksysu)].copy_(org_classifier_params)
    model.cuda()
    add_num = num_classes_market + num_classes_viper + num_classes_cuhksysu

    # Initialize classifer with class centers
    class_centers = initial_classifier(model, init_loader_msmt17)
    model.module.classifier.weight.data[(num_classes_market + num_classes_viper + num_classes_cuhksysu):].copy_(class_centers)
    model.cuda()

    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.train()
    model.train()

    #evaluators.extend([R1_mAP_eval(len(dataset_msmt17.query), max_rank=50, feat_norm=True)])
    #names.extend(["msmt17_norm"])
    test_loaders.extend([test_loader_msmt17])

    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr * 1, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)


    old_params = []
    for key, value in old_model.named_params(old_model):
        if not value.requires_grad:
            continue
        old_params += [{"params": [value], "lr": 0.01*args.lr , "weight_decay": args.weight_decay}]
    old_optimizer = torch.optim.Adam(old_params)

    lr_scheduler = WarmupMultiStepLR(optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    old_lr_scheduler = WarmupMultiStepLR(old_optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    trainer = Trainer(model, num_classes_msmt17 + add_num, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):

        train_loader_msmt17.new_epoch()
        trainer.train(epoch, train_loader_msmt17, replay_dataloader, optimizer, old_optimizer, training_phase=4,
                      train_iters=len(train_loader_msmt17), add_num=add_num, old_model=old_model, replay=True)
        lr_scheduler.step()
        old_lr_scheduler.step()

        if (epoch == 0 or epoch ==1):
            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                cmc, mAP_msmt = eval_func(epoch, evaluator, model, test_loader, name, old_model)
            
        if epoch == args.epochs - 1:
            evaluators.extend([R1_mAP_eval(len(dataset_msmt17.query), max_rank=50, feat_norm=True)])
            names.extend(["msmt17_norm"])
            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                cmc, mAP_msmt = eval_func(epoch, evaluator, model, test_loader, name, old_model)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP_msmt,
            }, True, fpath=osp.join(args.logs_dir, 'msmt17_checkpoint_bilearn_daxiao.pth.tar'))

            print('Finished epoch {:3d}  MSMT17 mAP: {:5.1%}'.format(epoch, mAP_msmt))
    
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-br', '--replay-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='/public/home/yuchl/PTKP/logs/viper_checkpoint.pth.tar', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('/public/home/yuchl/', 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    main()