from torch.utils.data import DataLoader, ConcatDataset

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as tf


def kitti_zhou_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    #接收 resize_hheight:调整后图像高度 、resize_width（调整后的图像宽度）、crop_height（裁剪后的图像高度）、crop_width（裁剪后的图像宽度）、batch_size（批次大小）和num_workers（工作线程数）。
    """A loader that loads image sequences for depth training from the
    kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [                     #数据转换操作
        tf.RandomHorizontalFlip(),              #随机水平翻转
        tf.CreateScaledImage(),                 #创建缩放图像
        tf.Resize(                              #调整图像大下
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),                      #将深度图像进行转换
        tf.CreateColoraug(new_element=True),    #创建颜色增强图案
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),  #颜色抖动，对图像的亮度、对比度、饱和度、色调和gamma进行随机变换。
        tf.RemoveOriginals(),       #移除原始图像
        tf.ToTensor(),              #转换为张量
        tf.NormalizeZeroMean(),     #零均值归一化
        tf.AddKeyValue('domain', 'kitti_zhou_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'mono',
        'split': 'zhou_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color', ),         #左右相机的配置参数
                'keys_to_video': ('color', )}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,         #tf.ExchangeStereo()用于交换左右相机图像
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))        #将左右相机的数据集合成一个数据集对象

    loader = DataLoader(                               #数据集加载到内存中
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (zhou_split) train split for depth training", flush=True)

    return loader


def kitti_kitti_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads image sequences for depth training from the kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_kitti_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'mono',
        'split': 'kitti_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color',),
                'keys_to_video': ('color',)}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (kitti_split) train set for depth training", flush=True)

    return loader


def kitti_odom09_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads image sequences for depth training from the
    kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_odom09_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'stereo',
        'split': 'odom09_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color', ),
                'keys_to_video': ('color', )}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (odom09_split) train split for depth training", flush=True)

    return loader


def kitti_benchmark_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads image sequences for depth training from the
    kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_benchmark_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'stereo',
        'split': 'benchmark_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color', ),
                'keys_to_video': ('color', )}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (benchmark_split) train split for depth training",
          flush=True)

    return loader
