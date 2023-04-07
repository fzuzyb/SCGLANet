#!/usr/bin/env python
# encoding: utf-8
#@author: ZYB
#@license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
#@file: data_file_process.py
#@time: 3/24/23 11:51 AM


import os
import shutil
from concurrent.futures import ThreadPoolExecutor as Pool
import multiprocessing
from tqdm import tqdm
import os

def copy_images(filename):
    # 遍历源文件夹中的所有文件

    # 构造源文件路径和目标文件路径
    source_file_path = os.path.join(source_folder_path, filename)
    destination_file_path = os.path.join(destination_folder_path, filename)
    # 拷贝文件
    shutil.copyfile(source_file_path, destination_file_path)
    return filename

if __name__ == '__main__':

    # 定义源文件夹路径
    source_folder_paths = ['/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Flickr1024', '/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Middlebury']

    # 定义目标文件夹路径
    destination_folder_path = '/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/HR'


    for source_folder_path in source_folder_paths:
        print(source_folder_path)
        pool = Pool(max_workers=10)
        filename = os.listdir(source_folder_path)
        pbar = tqdm(total=len(filename))
        results = pool.map(copy_images,filename)
        for result in results:
            pbar.update()
            pbar.set_description(result)

