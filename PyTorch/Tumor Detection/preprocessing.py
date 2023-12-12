import os
from glob2 import glob
import torch
from monai.transforms import Compose, LoadImaged, ToTensord, AddChanneld, Resized, Spacingd, Orientationd, CropForegroundd,    ScaleIntensityRanged
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
import matplotlib.pyplot as plt

def prepare(in_dir, pixdim = (1.5, 1.5, 1.0),
a_min = -200,
a_max = 200,
spatial_size=[128, 128, 64],
cache=False):

    set_determinism(seed=0)
    data_dir = 'D:\projects\Tumor Detection\Task03 Liver smaller\\fixed_data\\all_together'
    train_images = sorted(glob(os.path.join(data_dir, 'TrainData', '*.nii.gz')))
    train_labels = sorted(glob(os.path.join(data_dir, 'TrainLabels', '*.nii.gz')))

    val_images = sorted(glob(os.path.join(data_dir, 'ValData', '*.nii.gz')))
    val_labels = sorted(glob(os.path.join(data_dir, 'ValLabels', '*.nii.gz')))

    train_files = [{"image":image_name, 'label':label_name} for image_name, label_name in zip(train_images, train_labels)]
    test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]

    print(train_files)

    #load the images
    #do any transforms
    #convert to tensors



    train_transforms = Compose(
        [
            LoadImaged(keys=['image','labels']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=['image','label'], pixdim=pixdim, mode=('bilinear','neartest')),
            Orientationd(keys=['image','label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=['image', 'label'])
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=10)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(train_ds, batch_size=1)

        return train_loader, test_loader

    plt.figure('test', (12, 6))
    plt.subplit(1, 2, 2)
    plt.title('slice of patient')
    plt.imshow(test_patient['image'][0, 0:, :, 30], cmap='gray')

    plt.subplit(1,2,2)
    plt.title('label of patient')
    plt.imshow(test_patient['label'][0,0,:,:,30],cmap='gray')
    plt.show()

