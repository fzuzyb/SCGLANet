#!/usr/bin/env python
# encoding: utf-8
#@author: ZYB
#@license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
#@file: model_preparation.py
#@time: 3/28/23 2:38 PM

import torch

from collections import OrderedDict
if __name__ == '__main__':
    ckpt_path = "/home/bobo/Temp_new/ZYB/IV_WORKING/codes/GeneralSR/SCGLANet/experiments/1_SCGLANet_L_x4_Flickr1024/models/merge_1850_2025-2175-2150.pth"
    cktp = torch.load(ckpt_path)
    new_ckpt = OrderedDict()

    for k,v in cktp['params'].items():
        if "up" not in k :
            new_ckpt[k] = v
        else:
            print(k)
    out = {'params': new_ckpt}
    torch.save(out,"/home/bobo/Temp_new/ZYB/IV_WORKING/codes/GeneralSR/SCGLANet/experiments/1_SCGLANet_L_x4_Flickr1024/models/merge_1850_2025-2175-2150_x2_pretrain.pth")
