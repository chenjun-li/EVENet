# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import nibabel as nib
import torch
from data_loader import dataset as dset
from data_loader.augmentation import ToTensor, ZeroPad2D, AddGaussianNoise
from torch.utils.data import DataLoader, Dataset
# IMPORTS
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from utils import logging

logger = logging.getLogger(__name__)


class HCPDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root_dir, self.file_list[idx])
        orig_path = os.path.join(folder_path, 'orig.nii.gz')
        gt_path = os.path.join(folder_path, 'heatmap.nii.gz')

        orig_data = self.load_nii_to_tensor(orig_path) * 256
        gt_data = self.load_nii_to_tensor(gt_path)

        return orig_data, gt_data

    def load_nii_to_tensor(self, file_path):
        import nibabel as nib
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        tensor_data = to_tensor(nii_data)
        return tensor_data


def create_dataloaders(root_dir, batch_size, train_ratio=0.8):
    dataset = HCPDataset(root_dir)
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


class SwinbdyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        e3_path = os.path.join(self.root_dir, self.file_list[index], 'orig.nii.gz')
        heatmap_path = os.path.join(self.root_dir, self.file_list[index], 'heatmap.nii.gz')

        e3_data = nib.load(e3_path).get_fdata()
        heatmap_data = nib.load(heatmap_path).get_fdata()

        # Extract neighboring channels
        num_channels = e3_data.shape[-1]
        if index < 10:
            padding = torch.zeros((e3_data.shape[0], e3_data.shape[1], 10 - index))
            e3_data = torch.cat((padding, e3_data), dim=-1)
            heatmap_data = torch.cat((padding, heatmap_data), dim=-1)

        start_channel = max(0, index - 10)
        end_channel = min(num_channels, index + 10)
        e3_data = e3_data[..., start_channel:end_channel]
        heatmap_data = heatmap_data[..., start_channel:end_channel]

        return torch.from_numpy(e3_data), torch.from_numpy(heatmap_data)


def create_dataloaders_bdy(data_dir):
    dataset = SwinbdyDataset(data_dir)

    # Splitting the dataset into two subsets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Creating dataloaders for train and validation subsets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader


def get_dataloader(cfg, mode):
    """
        Creating the dataset and pytorch data loader
    :param cfg:
    :param mode: loading data for train, val and test mode
    :return:
    """
    assert mode in ['train', 'val'], f"dataloader mode is incorrect {mode}"

    padding_size = cfg.DATA.PADDED_SIZE

    if mode == 'train':

        if "None" in cfg.DATA.AUG:
            tfs = [ZeroPad2D((padding_size, padding_size)), ToTensor()]
            # old transform
            if "Gaussian" in cfg.DATA.AUG:
                tfs.append(AddGaussianNoise(mean=0, std=0.1))

            shuffle = True
            data_path = cfg.DATA.PATH_HDF5_TRAIN
            logger.info(f"Loading {mode.capitalize()} data ... from {data_path}. Using standard Aug")

            # dataset = dset.MultiScaleDatasetVal(data_path, cfg, transforms.Compose(tfs))

            hdf5_dir = "/data01/hdf5_seg50_1k/E3_train_axial"
            hdf5_files = [os.path.join(hdf5_dir, f"train{i}.hdf5") for i in range(1, 3)]

            dataset = dset.MultiScaleDatasetVal_1K(dataset_paths=hdf5_files, cfg=cfg,
                                                   transforms=transforms.Compose(tfs))

            # data_paths = ["/data/dataset/HCP/hdf5_set/train_set_coronal_30sub_md.hdf5",
            #               "/data/dataset/HCP/hdf5_set/train_set_coronal_30sub_e3.hdf5",
            #               "/data/dataset/HCP/hdf5_set/train_set_coronal_30sub_fa.hdf5"]
            # dataset = dset.FusionDatasetVal(data_paths, cfg, transforms.Compose(tfs))

    elif mode == 'val':
        data_path = cfg.DATA.PATH_HDF5_VAL
        shuffle = False
        transform = transforms.Compose([ZeroPad2D((padding_size, padding_size)),
                                        ToTensor(),
                                        ])
        logger.info(f"Loading {mode.capitalize()} data ... from {data_path}")

        # dataset = dset.MultiScaleDatasetVal(data_path, cfg, transform)

        hdf5_dir = "/data01/hdf5_seg50_1k/E3_val_axial"
        hdf5_files = [os.path.join(hdf5_dir, f"val{i}.hdf5") for i in range(1, 2)]
        dataset = dset.MultiScaleDatasetVal_1K(dataset_paths=hdf5_files, cfg=cfg, transforms=transform)

        # data_paths = ["/data/dataset/HCP/hdf5_set/val_set_coronal_30sub_md.hdf5",
        #               "/data/dataset/HCP/hdf5_set/val_set_coronal_30sub_e3.hdf5",
        #               "/data/dataset/HCP/hdf5_set/val_set_coronal_30sub_fa.hdf5"]
        # dataset = dset.FusionDatasetVal(data_paths, cfg, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True,
    )

    return dataloader, dataset
