from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    #dataset_dir = 'Market-1501-v15.09.15'

    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.replay_dir = osp.join(self.dataset_dir, 'replay_sample')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        replay = self._process_dir(self.replay_dir, relabel=True)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.replay = replay

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    
    @property
    def images_dir(self):
        return self.dataset_dir

    def _process_dir(self, dir_path, relabel=False):
        dict_cam_seq_max = {
        11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
        21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
        31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
        41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
        51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
        61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_([-\d]+)_([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _, _, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid, seq, frame, count = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            new_frame = 0
            for i in range(1, int(seq)):
                new_frame = new_frame + dict_cam_seq_max[int(str(camid+1) + str(i))]
            new_frame = new_frame + int(frame)
            dataset.append((img_path, pid, camid, 1))

        return dataset
