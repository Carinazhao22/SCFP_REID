from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

import datasets.utils as utils
import torchvision.transforms as transforms

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        if train !=None:
            num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        if train !=None:
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

def read_evalimage(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_evalimage(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]


class Train_ImageDataset(Dataset):
    def __init__(self, dataset, grids, transform=None, transform2=None, occPath=None,pattern=4, ratio=0):
        self.dataset = dataset
        self.transform = transform
        self.transform2 = transform2
        self.ratio = ratio
        self.occPath = occPath
        self.pattern = pattern
        self.length = len(self.dataset)
        self.item = 0
        self.grids = grids
       
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img, occ_img, mask_labels, mask = self.read_trainimage(img_path)

        if self.transform2 is not None:
            img = self.transform2(img)
            occ_img = self.transform2(occ_img)
            img = torch.cat((img,occ_img),dim=0)
            
        return img, pid, mask_labels, mask, camid, trackid,img_path.split('/')[-1]
    
    def PIL_reader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)

    def occlude_img(self, img):
        # occlude img
        occluder = str(random.choice(range(37)))
        occ = self.PIL_reader(osp.join(self.occPath,occluder+'.jpg'))
        factor = random.choice(np.linspace(1, 5, 9, endpoint=True))
        img_occ, mask, _ = utils.occluded_image_ratio(img.copy(), occ, factor)
         
        # W, H = * img_occ.size, 
 
        # cal mask label
        mask_labels = utils.cal_similarity_label(self.grids, mask)
    
        return img_occ, mask_labels, mask

    def read_trainimage(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform!=None:
                    img = self.transform(img)
                got_img = True
                # H,W = img.size[1],img.size[0]
                # # grids, label_smoothing = utils.get_grids(H,W, self.pattern)
                img_occ, mask_label, mask = self.occlude_img(img)
                mask = torch.tensor(mask)
                 
                return img, img_occ, mask_label, mask, 
                # if random.choice(range(self.ratio)) == 0:
                #     mask = torch.zeros((img.size[1],img.size[0]))
                #     # return [0,0,0] represent head torso legs
                #     return img, 0, mask
                # else:
                #     H,W = img.size[1],img.size[0]
                #     # grids, label_smoothing = utils.get_grids(H,W, self.pattern)
                #     img_occ, mask_label, mask = self.occlude_img(img)
                #     mask = torch.tensor(mask)
                     
                #     return img_occ, mask_label, mask, 
            
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass