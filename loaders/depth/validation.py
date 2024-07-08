from torch.utils.data import DataLoader

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as tf


def kitti_zhou_validation(img_height, img_width, batch_size, num_workers):
    #函数接受一些参数，包括img_height（图像的高度）、img_width（图像的宽度）、batch_size（批次大小）和num_workers（工作线程数）。
    """A loader that loads images and depth ground truth for
    depth validation from the kitti validation set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_zhou_val_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_zhou'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='zhou_split',
        trainvaltest_split='validation',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (zhou_split) validation set for depth validation",
          flush=True)

    return loader


def kitti_zhou_test(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth evaluation from the kitti test set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_zhou_test_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_zhou'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='zhou_split',
        trainvaltest_split='test',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (zhou_split) test set for depth evaluation", flush=True)

    return loader


def kitti_kitti_validation(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth validation from the kitti validation set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_kitti_val_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_kitti'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='kitti_split',
        trainvaltest_split='validation',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (kitti_split) validation set for depth validation",
          flush=True)

    return loader


def kitti_2015_train(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth evaluation from the kitti_2015 training set (but for evaluation).
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_2015_train_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_kitti'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti_2015',
        trainvaltest_split='train',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti_2015 test set for depth evaluation", flush=True)

    return loader

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, video_path, img_height, img_width):
        self.video_path = video_path
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        cap.release()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_width, self.img_height))
        frame = self.transform(frame)

        # Return a dummy depth value since the real depth values are not available
        depth = torch.zeros((1, self.img_height, self.img_width))

        return frame, depth

def custom_data_loader(video_path, img_height, img_width, batch_size, num_workers):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    dataset = CustomDataset(video_path, img_height, img_width)
    dataset.num_frames = num_frames

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=False)

    return loader

