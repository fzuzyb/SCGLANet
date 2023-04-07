#!/usr/bin/env python
# encoding: utf-8
#@author: ZYB
#@license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
#@file: test_vgg_dino.py
#@time: 12/16/22 12:05 PM

from basicsr.archs.vgg_arch import VGGFeatureExtractor
import cv2
import torch
import numpy as np
import os
from einops import rearrange
def img2tensor(img):
    img = img[:, :, (2, 1, 0)]  # to RGB
    img = (img / 255.0).astype(np.float32)
    img = torch.from_numpy(np.transpose(img,(2,0,1))).unsqueeze(0)
    return img

def visual_cnn():
    layer_weights = { 'conv1_2': 0.1,
                      'conv2_2': 0.1,
                      'conv3_4': 1,
                      'conv4_4': 1,
                      'conv5_4': 1}

    vgg = VGGFeatureExtractor(
        layer_name_list=list(layer_weights.keys()),
        use_input_norm=True,
        range_norm=False).cuda()
    img = cv2.imread("/home/bobo/Temp_new/ZYB/IV_WORKING/codes/contrastive_learning/STEGO/src/dino/dog.png")
    img = cv2.resize(img,dsize=(512,512))
    imgtensor = img2tensor(img).cuda()
    features = vgg(imgtensor)
    b1s = [8, 8, 16, 16, 16]
    b2s = [8, 16, 16, 32, 32]
    for index,(k,v) in enumerate(features.items()):
        print(k,v.size())
        show_fea = v.detach().cpu().squeeze().numpy()
        show_fea = (show_fea - show_fea.min()) / (show_fea.max() - show_fea.min())
        show_fea = (rearrange(show_fea, '(b1 b2 1) h w -> (b1 h) (b2 w)', b1=b1s[index], b2=b2s[index])*255).astype(np.uint8)
        # show_fea = cv2.applyColorMap(show_fea,colormap=cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(
            "/home/bobo/Temp_new/ZYB/IV_WORKING/codes/GeneralSR/BasicSR/experiments/vit_cnn_visualization",'gray_{}.png'.format(k)),show_fea)
        # cv2.imshow('fea', show_fea)
        # cv2.waitKey(0)
def visual_vit():
    pass
if __name__ == '__main__':
    pass


