# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
from basicsr.data.transforms import stereo_augment, stereo_random_crop_hw
from basicsr.utils import FileClient, imfrombytes, img2tensor
import os
import numpy as np
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(suffix='basicsr')
class PairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(PairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'

        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)
        self.gt_files = list(filter(lambda file: file.find('L')!=-1, self.gt_files ))
        self.lq_files = list(filter(lambda file: file.find('L')!=-1, self.lq_files ))

        self.nums = len(self.gt_files)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        L_filename = self.gt_files[index]
        R_filename = L_filename.replace('L', 'R')

        gt_path_L = os.path.join(self.gt_folder, L_filename)
        gt_path_R = os.path.join(self.gt_folder, R_filename)

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            # img_gt_L = cv2.imread(gt_path_L)/255.0
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            # img_gt_R = cv2.imread(gt_path_R)/255.0
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_folder, L_filename)
        lq_path_R = os.path.join(self.lq_folder, R_filename)


        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            # img_lq_L = cv2.imread(lq_path_L)/255.0
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            # img_lq_R = cv2.imread(lq_path_R)/255.0
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_gt, img_lq = stereo_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale)
            # flip, rotation
            imgs, status = stereo_augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt, img_lq = imgs


        img_gt = torch.cat([img2tensor(img_gt[:,:,:3],bgr2rgb=True,float32=True), img2tensor(img_gt[:,:,3:],bgr2rgb=True,float32=True)], dim=0)
        img_lq = torch.cat([img2tensor(img_lq[:,:,:3],bgr2rgb=True,float32=True), img2tensor(img_lq[:,:,3:],bgr2rgb=True,float32=True)], dim=0)



        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums




@DATASET_REGISTRY.register(suffix='basicsr')
class TestPairedStereoImageDataset(data.Dataset):
    '''
        Testdata set for filckr1024 KITTI2012 KITTI2015 Middlebury
    '''
    def __init__(self, opt):
        super(TestPairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')


        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        img_gt = torch.cat([img2tensor(img_gt[:, :, :3], bgr2rgb=True, float32=True),
                            img2tensor(img_gt[:, :, 3:], bgr2rgb=True, float32=True)], dim=0)
        img_lq = torch.cat([img2tensor(img_lq[:, :, :3], bgr2rgb=True, float32=True),
                            img2tensor(img_lq[:, :, 3:], bgr2rgb=True, float32=True)], dim=0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums


@DATASET_REGISTRY.register(suffix='basicsr')
class InferStereoImageDataset(data.Dataset):
    '''
        Testdata set for filckr1024 test
    '''
    def __init__(self, opt):
        super(InferStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.lq_folder = opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        self.lq_files = os.listdir(self.lq_folder)

        self.nums = len( self.lq_files)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')


        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        img_lq = torch.cat([img2tensor(img_lq[:, :, :3], bgr2rgb=True, float32=True),
                            img2tensor(img_lq[:, :, 3:], bgr2rgb=True, float32=True)], dim=0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index])
        }

    def __len__(self):
        return self.nums

@DATASET_REGISTRY.register(suffix='basicsr')
class InferStereoImageV2Dataset(data.Dataset):
    '''
        Testdata set for filckr1024 KITTI2012 KITTI2015 Middlebury
    '''
    def __init__(self, opt):
        super(InferStereoImageV2Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.lq_folder = opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        self.lq_files = os.listdir(self.lq_folder)
        self.lq_files = list(filter(lambda file: file.find('L') != -1, self.lq_files))
        self.nums = len(self.lq_files)
        print(self.nums)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        L_filename = self.lq_files[index]
        R_filename = L_filename.replace('L', 'R')
        lq_path_L = os.path.join(self.lq_folder, L_filename)
        lq_path_R = os.path.join(self.lq_folder, R_filename)

        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        img_lq = torch.cat([img2tensor(img_lq[:, :, :3], bgr2rgb=True, float32=True),
                            img2tensor(img_lq[:, :, 3:], bgr2rgb=True, float32=True)], dim=0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index])
        }

    def __len__(self):
        return self.nums
