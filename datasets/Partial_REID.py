import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

# Sources :
# https://github.com/hh23333/PVPM
# Zhuo, J., Chen, Z., Lai, J., & Wang, G. (2018). Occluded Person Re-identification.


class Partial_REID(BaseImageDataset):
    dataset_dir = 'Partial_REID'
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Partial_REID, self).__init__()
        self.train_dataset_dir = osp.join(root, 'market1501')
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.train_dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'partial_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')

        self.pid_begin = pid_begin
        train = self._train_process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False,is_query=False)

        if verbose:
            print("=> Partial_REID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def _train_process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
    
    def _process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
         
        if is_query:
            camid = 0
        else:
            camid = 1
        
        
        pid_container = set()
        for img_path in sorted(img_paths):
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            
            if relabel:
                pid = pid2label[pid]
            
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

             
        return dataset

    # def _process_dir(self, dir_path, relabel=False):
    #     img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    #     pattern = re.compile(r'([-\d]+)_c(\d)')

    #     pid_container = set()
    #     for img_path in sorted(img_paths):
    #         pid, _ = map(int, pattern.search(img_path).groups())
    #         if pid == -1: continue  # junk images are just ignored
    #         pid_container.add(pid)
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}
    #     dataset = []
    #     for img_path in sorted(img_paths):
    #         pid, camid = map(int, pattern.search(img_path).groups())
    #         if pid == -1: continue  # junk images are just ignored
    #         assert 0 <= pid <= 1501  # pid == 0 means background
    #         assert 1 <= camid <= 6
    #         camid -= 1  # index starts from 0
    #         if relabel: pid = pid2label[pid]

    #         dataset.append((img_path, self.pid_begin + pid, camid, 1))
    #     return dataset
    
