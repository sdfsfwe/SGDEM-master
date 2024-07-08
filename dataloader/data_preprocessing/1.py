import os
import sys
import wget
import zipfile
import pandas as pd
import shutil
import numpy as np
import glob
import cv2

import dataloader.file_io.get_path as gp
import dataloader.file_io.dir_lister as dl
from kitti_utils import pcl_to_depth_map


kitti_path = 'E:\\SGDepth-master\\Datasets\\kitti_download'  # 替换为实际的KITTI数据文件夹路径
kitti_path_depth_annotated = os.path.join(kitti_path, 'Depth_improved')

# 检查data_depth_annotated.zip是否存在
depth_zip_path = 'E:\\temp\\data_depth_annotated.zip' # 替换为实际的data_depth_annotated.zip文件路径
if os.path.isfile(depth_zip_path):
    # 创建目标文件夹
    if not os.path.isdir(kitti_path_depth_annotated):
        os.makedirs(kitti_path_depth_annotated)

    # 解压缩data_depth_annotated.zip
    unzipper = zipfile.ZipFile(depth_zip_path, 'r')
    unzipper.extractall(kitti_path_depth_annotated)
    unzipper.close()

    # 删除data_depth_annotated.zip
    os.remove(depth_zip_path)

    # 移动文件夹到指定位置
    trainval_folder = os.listdir(kitti_path_depth_annotated)
    kitti_drives_list = []
    for sub_folder in trainval_folder:
        sub_folder = os.path.join(kitti_path_depth_annotated, sub_folder)
        kitti_drives_list.extend([os.path.join(sub_folder, i) for i in os.listdir(sub_folder)])

    for sub_folder in kitti_dirs_days:
        sub_folder = os.path.join(kitti_path_depth_annotated, sub_folder)
        if not os.path.isdir(sub_folder):
            os.makedirs(sub_folder)
        for drive in kitti_drives_list:
            if os.path.split(sub_folder)[1] in drive:
                shutil.move(drive, sub_folder)

    # 删除临时文件夹
    for sub_folder in trainval_folder:
        sub_folder = os.path.join(kitti_path_depth_annotated, sub_folder)
        shutil.rmtree(sub_folder)
else:
    print("data_depth_annotated.zip 文件不存在。请检查文件路径是否正确。")
