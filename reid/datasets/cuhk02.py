from __future__ import division, print_function, absolute_import
import glob
import os.path as osp

import random


from ..utils.serialization import write_json, read_json

import numpy as np
from reid.utils.data.dataset1 import ImageDataset
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable
from easydict import EasyDict
from collections import defaultdict, OrderedDict
import operator
class IncrementalPersonReIDSamples:

    def _relabels_incremental(self, samples, label_index, is_mix=False):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        pid2label = {}
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()

        # reorder
        for sample in samples:
            sample = list(sample)
            pid2label[sample[label_index]] = ids.index(sample[label_index])
        new_samples = copy.deepcopy(samples)
        for i, sample in enumerate(samples):
            new_samples[i] = list(new_samples[i])
            new_samples[i][label_index] = pid2label[sample[label_index]]
        if is_mix:
            return samples, pid2label
        else:
            return new_samples

    def _load_images_path(self, folder_dir, domain_name='market', is_mix=False):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name, is_mix=is_mix)
                samples.append([root_path + file_name, identi_id, camera_id, domain_name])
        return samples

    def _analysis_file_name(self, file_name, is_mix=False):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return: 0844, 3
        '''

        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        if is_mix:
            identi_id, camera_id = int(split_list[0]), int(split_list[2])
        else:
            identi_id, camera_id = int(split_list[0]), int(split_list[1])

        return identi_id, camera_id

    def _show_info(self, train, query, gallery, name=None, if_show=True):
        if if_show:
            def analyze(samples):
                pid_num = len(set([sample[1] for sample in samples]))
                cid_num = len(set([sample[2] for sample in samples]))
                sample_num = len(samples)
                return sample_num, pid_num, cid_num

            train_info = analyze(train)
            query_info = analyze(query)
            gallery_info = analyze(gallery)

            # please kindly install prettytable: ```pip install prettyrable```
            table = PrettyTable(['set', 'images', 'identities', 'cameras'])
            table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])
            table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
            table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
            table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
            print(table)
        else:
            pass

class CUHK02(IncrementalPersonReIDSamples):
    '''
    Market Dataset
    '''
    dataset_dir = ''
    cam_pairs = ['P1', 'P2', 'P3', 'P4', 'P5']
    test_cam_pair = 'P5'
    def __init__(self, datasets_root, relabel=True, combineall=False, split_id = 0):
        self.root = datasets_root
        self.relabel = relabel
        self.combineall = combineall
        self.dataset_dir = osp.join(self.root, 'Dataset')
        train, query, gallery = self.get_data_list()

        self.train, self.query, self.gallery = train, query, gallery
        print("888")
        print(len(train), len(query))
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        self._show_info(self.train, self.query, self.gallery)
    
    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid,_ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
    @property
    def images_dir(self):
        return self.dataset_dir

    def get_data_list(self):
        num_train_pids, camid = 0, 0
        train, query, gallery = [], [], []
        print('camdir')
        flag=0
        for cam_pair in self.cam_pairs:
            if(flag==0):
                print(osp.join(self.dataset_dir, cam_pair))
                flag = flag + 1
            cam_pair_dir = osp.join(self.dataset_dir, cam_pair)

            cam1_dir = osp.join(cam_pair_dir, 'cam1')
            cam2_dir = osp.join(cam_pair_dir, 'cam2')

            impaths1 = glob.glob(osp.join(cam1_dir, '*.png'))
            impaths2 = glob.glob(osp.join(cam2_dir, '*.png'))

            if cam_pair == self.test_cam_pair:
                # add images to query
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    pid = int(pid)
                    query.append((impath, pid, camid, 4))
                camid += 1

                # add images to gallery
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    pid = int(pid)
                    gallery.append((impath, pid, camid, 4))
                camid += 1

            else:
                pids1 = [
                    osp.basename(impath).split('_')[0] for impath in impaths1
                ]
                pids2 = [
                    osp.basename(impath).split('_')[0] for impath in impaths2
                ]
                pids = set(pids1 + pids2)
                pid2label = {
                    pid: label + num_train_pids
                    for label, pid in enumerate(pids)
                }

                # add images to train from cam1
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    pid = pid2label[pid]
                    train.append((impath, pid, camid, 4))
                camid += 1

                # add images to train from cam2
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    pid = pid2label[pid]
                    train.append((impath, pid, camid, 4))
                camid += 1
                num_train_pids += len(pids)

        return train, query, gallery


