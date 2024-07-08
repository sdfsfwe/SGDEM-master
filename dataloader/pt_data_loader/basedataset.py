# MIT License
#
# Copyright (c) 2020 Marvin Klingner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from torch.utils.data import Dataset
from torchvision import transforms
import os

import json
import cv2
import numpy as np

import dataloader.pt_data_loader.mytransforms as mytransforms
import dataloader.pt_data_loader.dataset_parameterset as dps
import dataloader.file_io.get_path as gp
import dataloader.file_io.dir_lister as dl


class BaseDataset(Dataset):
    """Image Dataset which can be used to load several images and their corresponding additional information"""

    def __init__(self,
                 dataset,    #数据集的名称
                 trainvaltest_split,  #可以是train、validation或test，用于指定加载的是训练集、验证集还是测试集
                 video_mode='mono',   #可以是mono或video，定义是否仅加载图像或图像序列
                 stereo_mode='mono',  #可以是mono或stereo，定义是否加载立体图像
                 cluster_mode=None,
                 simple_mode=False,  #如果为True，则直接从文件夹中读取数据，而不使用.json文件
                 labels=None,   #以Cityscapes中的命名元组风格定义的标签。从definitions文件夹中获取标签
                 labels_mode=None,   #可以是fromid或fromrgb，定义分割掩码是以id还是颜色的形式给出
                 data_transforms=None,  #接受transforms.compose列表，用于数据变换
                 scales=None,   #图像应该加载的所有比例的列表（以2的幂为指数的列表）
                 keys_to_load=None,  #定义应该加载的所有键
                 keys_to_video=None,  #定义应该加载序列的键
                 keys_to_stereo=None,  #定义应该加载立体图像的键
                 split=None,   #应该加载的数据集拆分，默认为完整的数据集本身
                 video_frames=None,  #应该加载的序列的所有帧（相对于主帧的帧号列表，例如[0, -2, -1, 1, 2]）
                 disable_const_items=True,  #从加载过程中移除常量项，如相机校准
                 folders_to_load=None,  #应该从中加载数据的文件夹列表；未提及的文件夹将在相应的集合中被跳过。只考虑路径中的最后一个文件夹；过滤器不区分大小写。
                                        #默认值：None -> 从数据集加载所有文件夹
                 files_to_load=None,  #应该加载的文件列表；未提及的文件将在相应的集合中被跳过。文件名不需要完整；过滤器不区分大小写。
                                      # 默认值：None -> 从数据集加载所有文件
                 n_files=None,   #应该加载的文件数量。如果文件数量超过n_files，则随机选择文件。使用numpy.random.seed()种子
                 output_filenames=False,
                 flow_validation_mode=True
                 ):

        super(BaseDataset, self).__init__()
        assert isinstance(dataset, str)
        assert trainvaltest_split in ('train', 'validation', 'test'), '''trainvaltest_split must be train,
        validation or test'''
        assert video_mode in ('mono', 'video'), 'video_mode must be mono or video'
        assert stereo_mode in ('mono', 'stereo'), 'stereo_mode must be mono or stereo'
        assert isinstance(simple_mode, bool)
        if data_transforms is None:
            data_transforms = [mytransforms.CreateScaledImage(),
                               mytransforms.CreateColoraug(),
                               mytransforms.ToTensor()]
        if scales is None:
            scales = [0]
        if keys_to_load is None:
            keys_to_load = ['color']
        if keys_to_stereo is None and stereo_mode == 'stereo':
            keys_to_stereo = ['color']
        if keys_to_video is None and video_mode == 'video':
            keys_to_video = ['color']
        if video_frames is None:
            video_frames = [0, -1, 1]

        self.dataset = dataset
        self.video_mode = video_mode
        self.stereo_mode = stereo_mode
        self.scales = scales
        self.disable_const_items = disable_const_items
        self.output_filenames = output_filenames
        self.parameters = dps.DatasetParameterset(dataset)
        if labels is not None:
            self.parameters.labels = labels
        if labels_mode is not None:
            self.parameters.labels_mode = labels_mode
        path_getter = gp.GetPath()
        dataset_folder = path_getter.get_data_path()
        datasetpath = os.path.join(dataset_folder, self.dataset)
        self.datasetpath = datasetpath
        if split is None:
            splitpath = None
        else:
            splitpath = os.path.join(dataset_folder, self.dataset + '_' + split)

        if simple_mode is False:
            self.data = self.read_json_file(datasetpath, splitpath, trainvaltest_split,
                                            keys_to_load, keys_to_stereo, keys_to_video,
                                            video_frames, folders_to_load, files_to_load, n_files)
        else:
            self.data = self.read_from_folder(datasetpath, keys_to_load, video_mode, video_frames)

        self.load_transforms = transforms.Compose(   #加载数据时的转换
            [mytransforms.LoadRGB(),
             mytransforms.LoadSegmentation(),
             mytransforms.LoadDepth(),
             mytransforms.LoadFlow(validation_mode=flow_validation_mode),
             mytransforms.LoadNumerics()
             ])

        # IMPORTANT to create a new list if the same list is passed to multiple datasets. Otherwise, due to the
        # mutability of lists, ConvertSegmentation will only be added once. Hence, the labels may be wrong for the 2nd,
        # 3rd, ... dataset!
        self.data_transforms = list(data_transforms)

        # Error if CreateColorAug and CreateScaledImage not in transforms.
        if mytransforms.CreateScaledImage not in data_transforms:
            raise Exception('The transform CreateScaledImage() has to be part of the data_transforms list')
        if mytransforms.CreateColoraug not in data_transforms:
            raise Exception('The transform CreateColoraug() has to be part of the data_transforms list')

        # Error if depth, segmentation or flow keys are given but not the corresponding Convert-Transform
        if any([key.startswith('segmentation') for key in keys_to_load]) and \
                mytransforms.ConvertSegmentation not in self.data_transforms:
            raise Exception('When loading segmentation images, please add mytransforms.ConvertSegmentation() to '
                            'the data_transforms')
        if any([key.startswith('depth') for key in keys_to_load]) and \
                mytransforms.ConvertDepth not in self.data_transforms:
            raise Exception('When loading depth images, please add mytransforms.ConvertDepth() to the data_transforms')
        if any([key.startswith('flow') for key in keys_to_load]) and \
                mytransforms.ConvertFlow not in self.data_transforms:
            raise Exception('When loading flow images, please add mytransforms.ConvertFlow() to the data_transforms')

        # In the flow validation mode, it is not allowed to use data-altering transforms
        if any([key.startswith('flow') for key in keys_to_load]) and flow_validation_mode:
            allowed_transforms = [mytransforms.CreateScaledImage,
                                  mytransforms.CreateColoraug,
                                  mytransforms.ConvertSegmentation,
                                  mytransforms.ConvertDepth,
                                  mytransforms.ConvertFlow,
                                  mytransforms.RemoveOriginals,
                                  mytransforms.ToTensor,
                                  mytransforms.Relabel,
                                  mytransforms.OneHotEncoding,
                                  mytransforms.NormalizeZeroMean,
                                  mytransforms.AdjustKeys,
                                  mytransforms.RemapKeys,
                                  mytransforms.AddKeyValue
                                  ]
            for transform in self.data_transforms:
                if transform not in allowed_transforms:
                    raise Exception('In flow validation mode, it is not allowed to use data-altering transforms')

        # Set the correct parameters to the ConvertDepth and ConvertSegmentation transforms
        for i, transform in zip(range(len(self.data_transforms)), self.data_transforms):
            if isinstance(transform, mytransforms.ConvertDepth):
                transform.set_mode(self.parameters.depth_mode)
            elif isinstance(transform, mytransforms.ConvertSegmentation):
                transform.set_mode(self.parameters.labels, self.parameters.labels_mode)
            elif isinstance(transform, mytransforms.ConvertFlow):
                transform.set_mode(self.parameters.flow_mode, flow_validation_mode)

        self.data_transforms = transforms.Compose(self.data_transforms)

    def __len__(self): #返回数据集中元素的数量
        """Return the number of elements inside the dataset"""
        dict_keys = list(self.data.keys())  #获取数据字典的键列表
        return len(self.data[dict_keys[0]])  #返回第一个键对应的值的长度来确定元素数量

    def __getitem__(self, number):  #用于按索引获取数据集中的元素
        """Dataset element with index number 'number' is loaded"""
        sample = {}
        for item in list(self.data.keys()):
            if isinstance(self.data[item][number], str):  #根据索引number从数据字典中获取对应的元素
                element = self.read_image_file(self.data[item][number]) #字符串类型，则调用read_image_file方法读取图像文件并将其作为numpy数组返回
            else:
                element = self.data[item][number]
            sample.update({item: element})    #否则直接将元素添加到sample字典中
        if not self.disable_const_items:
            sample = self.add_const_dataset_items(sample)
        sample = self.load_transforms(sample) #依次应用load_transforms和data_transforms对sample进行转换和数据变换
        sample = self.data_transforms(sample)
        if self.output_filenames:   #如果output_filenames为True，它将每个键对应的元素的文件名添加到sample字典的filename键下，并返回最终的sample字典作为数据集的元素。
            sample['filename'] = {}
            for item in list(self.data.keys()):
                if isinstance(self.data[item][number], str):
                    sample['filename'][item] = (self.data[item][number])
        return sample

    def add_const_dataset_items(self, sample):  #用于向sample字典中添加数据集特定的常量或项。子类需要根据具体的数据集实现该方法。
        """Add dataset specific constants or items"""
        raise NotImplementedError

    def read_image_file(self, filepath):  #用于从文件路径中读取图像，并将其作为numpy数组返回
        """Returns an image as a numpy array"""
        filepath = os.path.join(self.datasetpath, filepath)
        filepath = filepath.replace('/', os.sep)
        filepath = filepath.replace('\\', os.sep)
        image = cv2.imread(filepath, -1)
        return image

    def read_json_file(self, datasetpath, splitpath, trainvaltest_split, keys_to_load,
                       keys_to_stereo, keys_to_video, video_frames, folders_to_load, files_to_load, n_files):
                       #datasetpath：数据集路径，splitpath：拆分文件的路径，trainvaltest_split：用于拆分的训练/验证/测试数据的标识
                       #keys_to_load：要加载的数据键列表，keys_to_stereo：要加载的立体数据键列表，keys_to_video：要加载的视频数据键列表
                       #video_frames：视频帧的索引列表，folders_to_load：要加载的文件夹列表，files_to_load：要加载的文件列表，n_files：要加载的文件数量
        """Reads a json file from a dataset and outputs its data for the data loader
        here one might include filtering by folders for video data"""

        assert self.video_mode in ['mono', 'video'], 'video mode is not supported'
        assert self.stereo_mode in ['mono', 'stereo'], 'stereo mode is not supported'

        if splitpath is None:
            splitpath = datasetpath

        #函数打开指定的JSON文件并加载其中的数据
        splitpath = os.path.join(splitpath, trainvaltest_split + '.json')
        datasetpath = os.path.join(datasetpath, 'basic_files' + '.json')
        assert os.path.isfile(datasetpath), 'Path to basic files is not valid'
        assert os.path.isfile(splitpath), 'Path to the split is not valid. Please use another argument for split.'

        # Load the basic filenames file to get the video data
        if self.video_mode == 'video':
            with open(datasetpath) as file:
                basic_json_data = json.load(file)
            basic_names = basic_json_data['names']
            basic_files = basic_json_data['files']
            basic_numerics = basic_json_data['numerical_values']

        # Load the split file
        with open(splitpath) as file:
            split_json_data = json.load(file)
        split_names = split_json_data['names']
        split_types = split_json_data['types']
        split_folders = split_json_data['folders']
        split_files = split_json_data['files']
        split_positions = split_json_data['positions']

        # Load all data necessary. Create dictionaries data_files and data_positions of the form
        #加载所有必要的数据。创建字典data_files和data_positions，其格式为：
        # data_files = {('color', 0, -1): ['rgb_images/file_001_001', 'rgb_images/file_001_002', ...],
        #               ('depth', 0, -1): ['depth_images/file_001_001', 'depth_images/file_001_001', ...],
        #               ...}
        # data_positions = {('color', 0, -1): [[0, 0, 0, 0], [1, 0, 0, 1], ...],
        #                   ('depth', 0, -1): [[0, 0, 0, 0], [1, 0, 0, 1], ...],
        #                   ...}
        #键有三个条目：数据类别，帧索引（当前为0，对于视频数据可以有不同的值），以及分辨率参数，其默认值为-1，并在进行某些数据转换后将发生变化。
        # The keys have three entries: the data category, a frame index (0 at the moment, can take different values
        # for video data) and a resolution parameter which has -1 as a default value und will be different after certain
        # data transforms have been performed.
        data_files = {}
        data_positions = {}
        frame_index = 0
        resolution = -1
        existing_positions_all = []

        keys_to_load = list(keys_to_load)
        # If the stereo mode is switched on, then all stereo keys are added if they exist in the split_names.
        # If this is not the case then nothing is added
        #如果立体声模式被打开，则将所有立体声键添加到split_names（如果它们存在）。如果不是这种情况，则不添加任何内容。
        if self.stereo_mode == 'stereo':
            for name in split_names:
                if name in keys_to_stereo:
                    if name[-6:] == '_right':
                        stereo_name = name[:-6]
                    else:
                        stereo_name = name + '_right'
                    if stereo_name in split_names and stereo_name not in keys_to_load:
                        keys_to_load.append(stereo_name)
        keys_to_load = tuple(keys_to_load)

        for name, filetype, folder, file, position in \
                zip(split_names, split_types, split_folders, split_files, split_positions):
            if name in keys_to_load:
                data_files.update({(name, frame_index, resolution): file})
                data_positions.update({(name, frame_index, resolution): position})
        # Keep only files where all data is available. existing_positions_all contains all global positions that are
        # present in every data category. Then, remove all entries in the value lists of data_files and data_positions
        # that do not occur in every data category.
        #保留仅当所有数据可用时的文件。existing_positions_all包含每个数据类别中都存在的所有全局位置。然后，删除data_files和data_positions的值列表中在每个数据类别中不存在的所有条目。
        for i, name in zip(range(len(data_files.keys())), data_files.keys()):
            if i == 0:
                existing_positions_all = np.array(data_positions[name])[:, 0]   #如果i等于0，将data_positions[name]的第一列提取出来并存储在existing_positions_all变量中
            else:
                existing_positions_one = np.array(data_positions[name])[:, 0]   #将data_positions[name]的第一列提取出来并存储在existing_positions_one变量中。
                existing_positions_all = list(set(existing_positions_all).intersection(existing_positions_one)) #使用set()函数将existing_positions_all和existing_positions_one两个列表取交集，
        existing_positions_all = sorted(existing_positions_all)  #existing_positions_all列表进行排序

        for name in data_files.keys():
            existing_positions_one = np.array(data_positions[name])[:, 0]
            indices_to_keep = []
            index_one = 0
            index_all = 0
            while index_all < len(existing_positions_all):
                if existing_positions_one[index_one] == existing_positions_all[index_all]:
                    indices_to_keep.append(index_one)
                    index_all += 1
                index_one += 1
            data_files[name] = [data_files[name][j] for j in indices_to_keep]
            data_positions[name] = [data_positions[name][j] for j in indices_to_keep]

        # Keep only files in certain folders. Lists in data_files and data_positions must have the same size and order!
        if folders_to_load is not None:
            assert isinstance(folders_to_load, list), "please provide a list for folders_to_load"
            assert len(folders_to_load) > 0, "please provide a non-empty list for folders_to_load"

            # Get a key from data_files; which key does not matter as lists (dict values) should be of same size and
            # order.
            key = next(iter(data_files.keys()))
            list_to_work_on = data_files[key]
            indices_to_keep = []

            # Iterate over list_to_work_on as well as folders_to_load. If the last folder of a path "i" in
            # list_to_work_on is equal to any folder in folders_to_load, its index "i" is appended to indices_to_keep.
            folders_to_load = set([folder.lower() for folder in folders_to_load])
            for i in range(len(list_to_work_on)):
                dir_to_compare = list_to_work_on[i].split(os.sep)
                if not set(dir_to_compare).isdisjoint(folders_to_load):
                    indices_to_keep.append(i)

            # Filter all lists based on indices_to_keep.
            assert len(indices_to_keep) > 0, "given folders_to_load is/are not existing in dataset"
            for name in data_files.keys():
                data_files[name] = [data_files[name][j] for j in indices_to_keep]
                data_positions[name] = [data_positions[name][j] for j in indices_to_keep]

        # Keep only certain files. Lists in data_files and data_positions must have the same size and order!
        if files_to_load is not None:
            assert isinstance(files_to_load, list), "please provide a list for files_to_load"
            assert len(files_to_load) > 0, "please provide a non-empty list for files_to_load"

            # Get a key from data_files; which key does not matter as lists (dict values) should be of same size and
            # order.
            key = next(iter(data_files.keys()))
            list_to_work_on = data_files[key]
            indices_to_keep = []

            # Iterate over list_to_work_on as well as files_to_load. If the filename of a file "i" in
            # list_to_work_on contains any file from files_to_load, its index "i" is appended to indices_to_keep.
            files_to_load = [file.lower() for file in files_to_load]
            for i in range(len(list_to_work_on)):
                file_to_compare = list_to_work_on[i].split(os.sep)[-1].lower()
                for file in files_to_load:
                    if file_to_compare[:len(file)] == file:
                        indices_to_keep.append(i)
                        break

            # Filter all lists based on indices_to_keep.
            assert len(indices_to_keep) > 0, 'given files_to_load is/are not existing in dataset'
            for name in data_files.keys():
                data_files[name] = [data_files[name][j] for j in indices_to_keep]
                data_positions[name] = [data_positions[name][j] for j in indices_to_keep]

        # Keep only files with compatible video modes
        if self.video_mode == 'video':
            existing_positions_all = []
            min_frame = np.min(np.array(video_frames))
            max_frame = np.max(np.array(video_frames))

            # Create a list existing_positions_all containing all global positions for which a sufficient amount of
            # preceding and succeeding frames is available in every data category
            for name in data_files.keys():
                if name[0] in keys_to_video:
                    # Create boolean arrays indicating whether each frame has enough preceding/succeeding frames
                    existing_min_frames = np.array(data_positions[name])[:, 1] >= -min_frame
                    existing_max_frames = np.array(data_positions[name])[:, 2] >= max_frame
                    # Get the position entries of all frames that have enough preceding and succeeding frames
                    existing_positions_one = np.array(data_positions[name])[:, 0]
                    existing_positions_one = existing_positions_one[np.logical_and(existing_min_frames,
                                                                                   existing_max_frames)]
                    # existing_positions_all is the list of global positions that have occured in every list
                    # existing_positions_one so far
                    if len(existing_positions_all) == 0:
                        existing_positions_all = existing_positions_one
                    else:
                        existing_positions_all = list(set(existing_positions_all).intersection(existing_positions_one))
            existing_positions_all = sorted(existing_positions_all)

            # Remove all entries in the value lists of every data category which do not have enough preceding and
            # succeeding frames
            for name in data_files.keys():
                existing_positions_one = np.array(data_positions[name])[:, 0]
                indices_to_keep = []
                index_one = 0
                index_all = 0
                while index_all < len(existing_positions_all):
                    if existing_positions_one[index_one] == existing_positions_all[index_all]:
                        indices_to_keep.append(index_one)
                        index_all += 1
                    index_one += 1
                data_files[name] = [data_files[name][j] for j in indices_to_keep]
                data_positions[name] = [data_positions[name][j] for j in indices_to_keep]

        # Add video files to data
        if self.video_mode == 'video':
            original_keys = list(data_files.keys())
            for name in original_keys:
                if isinstance(name, tuple) and name[0] in keys_to_video:
                    basic_name_index = basic_names.index(name[0])
                    indices = np.array(data_positions[name])[:, 3]
                    # For every video frame index,
                    for frame_index in video_frames:
                        if frame_index == 0:
                            continue
                        else:
                            # If a numerical entry exists, it will be loaded instead of a file (e.g. for camera
                            # intrinsic parameters).
                            if basic_numerics[basic_name_index] is not None:
                                frame_file = [basic_numerics[basic_name_index][j + frame_index] for j in indices]
                            else:
                                frame_file = [basic_files[basic_name_index][j + frame_index] for j in indices]
                            data_files.update({(name[0], frame_index, resolution): frame_file})

        # Select n_files to load, based on a uniform distribution
        if n_files is not None:
            # Get a key from data_files; which key does not matter as lists (dict values) should be of same size and
            # order.
            key = next(iter(data_files.keys()))
            list_to_work_on = data_files[key]

            # If n_files is greater or equal to len(list_to_work_on), we don't have to do anything.
            if n_files < len(list_to_work_on):
                indices_to_keep = np.random.choice(len(list_to_work_on), size=n_files, replace=False)
                for name in data_files.keys():
                    data_files[name] = [data_files[name][j] for j in indices_to_keep]

        return data_files

    def read_from_folder(self, path, keys_to_load, video_mode, video_frames):
        """
        Creates the data dictionary directly from the folder without a .json-File. Only suitable for simple datasets.

        Folders should have the same name as keys. Folder structure is assumed to be as follows:
        <path>
          color
            <image_01>
            ...
            <image_n>
          sgementation
            <image_01>
            ...
            <image_n>
          ...

        :param path: path of the dataset/dataset split to use
        :param keys_to_load:
        :param video_mode:
        :return: a dictionary with all files for each key, sorted alphabetically by filename
        """

        assert keys_to_load is not None, 'in simple mode, the keys must be specified'
        root_stringlength = len(path) + 1
        folders = dl.DirLister.get_directories(path)
        folders = sorted(folders, key=str.lower)
        keys_to_load = sorted(keys_to_load, key=str.lower)
        data_files = {}

        if video_mode == 'mono':
            frame_index = 0
            resolution = -1
            for key in keys_to_load:
                files = []
                # include all folders containing the key
                key_folders = dl.DirLister.include_dirs_by_name(folders, key)
                for folder in key_folders:
                    new_files = dl.DirLister.list_files_in_directory(folder)
                    # remove root path from file
                    for i in range(len(new_files)):
                        new_files[i] = new_files[i][root_stringlength:]
                    files.extend(new_files)
                    # sort alphabetically by file name (not by the full path)
                    files = sorted(files, key=lambda file: os.path.split(file)[1].lower())
                data_files.update({(key, frame_index, resolution): files})

        elif video_mode == 'video':
            # determine how many frames before and after each frame should be loaded. Based on that, determine the first
            # frame to load with index 0
            min_frame = np.min(np.array(video_frames))
            max_frame = np.max(np.array(video_frames))
            resolution = -1

            for key in keys_to_load:
                # include all folders containing the key
                key_folders = dl.DirLister.include_dirs_by_name(folders, key)
                all_files = {}
                # Create a dictionary that contains all file names for every folder
                for folder in key_folders:
                    new_files = dl.DirLister.list_files_in_directory(folder)
                    # remove root path from file
                    for i in range(len(new_files)):
                        new_files[i] = new_files[i][root_stringlength:]
                    all_files.update({folder: new_files})

                # Update the data files dictionary for every frame index
                for frame_index in video_frames:
                    files = []
                    for folder in key_folders:
                        first_frame = -min_frame + frame_index
                        last_frame = len(all_files[folder]) - max_frame + frame_index
                        new_files = all_files[folder][first_frame:last_frame]
                        files.extend(new_files)
                    data_files.update({(key, frame_index, resolution): files})
        for key in data_files.keys():
            print(str(key) + ': ' + str(data_files[key]))
            print()
        return data_files
