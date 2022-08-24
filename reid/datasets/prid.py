from __future__ import division, print_function, absolute_import
import random
import os.path as osp

from ..utils.serialization import write_json, read_json
import glob

import numpy as np
import os.path as osp
from reid.utils.data.dataset1 import ImageDataset
import numpy as np
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable
from easydict import EasyDict
import random
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

class PRID(IncrementalPersonReIDSamples):
    """PRID (single-shot version of prid-2011)

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and Discriminative
        Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_

    Dataset statistics:
        - Two views.
        - View A captures 385 identities.
        - View B captures 749 identities.
        - 200 identities appear in both views.
    """
    dataset_dir = 'prid2011'
    dataset_url = None

    def __init__(self, datasets_root, relabel=True, combineall=False, split_id=0):
        self.dataset_dir = osp.join(datasets_root)

        self.cam_a_dir = osp.join(
            self.dataset_dir, 'single_shot', 'cam_a'
        )
        self.cam_b_dir = osp.join(
            self.dataset_dir, 'single_shot', 'cam_b'
        )
        self.split_path = osp.join(self.dataset_dir, 'splits_single_shot.json')

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]

        train, query, gallery = self.process_split(split)
        self.train, self.query, self.gallery = train, query, gallery
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

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating splits ...')

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {'train': train_pids, 'test': test_pids}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def process_split(self, split):
        train_pids = split['train']
        test_pids = split['test']

        train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

        # train
        train = []
        for pid in train_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            pid = train_pid2label[pid]
            img_a_path = osp.join(self.cam_a_dir, img_name)
            train.append((img_a_path, pid, 0, 2))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            train.append((img_b_path, pid, 1, 2))

        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_a_path = osp.join(self.cam_a_dir, img_name)
            query.append((img_a_path, pid, 0, 2))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1, 2))
        for pid in range(201, 750):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1, 2))

        return train, query, gallery


