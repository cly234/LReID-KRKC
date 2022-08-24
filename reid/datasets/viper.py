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


class VIPeR(IncrementalPersonReIDSamples):
    """VIPeR.

    Reference:
        Gray et al. Evaluating appearance models for recognition, reacquisition, and tracking. PETS 2007.

    URL: `<https://vision.soe.ucsc.edu/node/178>`_

    Dataset statistics:
        - identities: 632.
        - images: 632 x 2 = 1264.
        - cameras: 2.
    """
    dataset_dir = 'viper'
    dataset_url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'

    def __init__(self, datasets_root, relabel=True, combineall=False, split_id=0):

        self.root = datasets_root
        self.relabel = relabel
        self.combineall = combineall
        self.dataset_dir = datasets_root
        self.cam_a_dir = osp.join(self.dataset_dir, 'VIPeR', 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir, 'VIPeR', 'cam_b')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']  # query and gallery share the same images
        gallery = split['gallery']

        train = [tuple(item + [0]) for item in train]
        print(train[0])
 
        query = [tuple(item + [0] ) for item in query]
        gallery = [tuple(item + [0] ) for item in gallery]

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
        return self.root

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')

            cam_a_imgs = sorted(glob.glob(osp.join(self.cam_a_dir, '*.bmp')))
            cam_b_imgs = sorted(glob.glob(osp.join(self.cam_b_dir, '*.bmp')))
            assert len(cam_a_imgs) == len(cam_b_imgs)
            num_pids = len(cam_a_imgs)
            print('Number of identities: {}'.format(num_pids))
            num_train_pids = num_pids // 2
            """
            In total, there will be 20 splits because each random split creates two
            sub-splits, one using cameraA as query and cameraB as gallery
            while the other using cameraB as query and cameraA as gallery.
            Therefore, results should be averaged over 20 splits (split_id=0~19).

            In practice, a model trained on split_id=0 can be applied to split_id=0&1
            as split_id=0&1 share the same training data (so on and so forth).
            """
            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                test_idxs = order[num_train_pids:]
                assert not bool(set(train_idxs) & set(test_idxs)), \
                    'Error: train and test overlap'

                train = []
                for pid, idx in enumerate(train_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    train.append((cam_a_img, pid, 0))
                    train.append((cam_b_img, pid, 1))

                test_a = []
                test_b = []
                for pid, idx in enumerate(test_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    test_a.append((cam_a_img, pid, 0))
                    test_b.append((cam_b_img, pid, 1))

                # use cameraA as query and cameraB as gallery
                split = {
                    'train': train,
                    'query': test_a,
                    'gallery': test_b,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {
                    'train': train,
                    'query': test_b,
                    'gallery': test_a,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))