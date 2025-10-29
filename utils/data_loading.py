import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.path_hyperparameter import ph
import cv2


class LabelProcessor:
    @classmethod
    def label2direction(cls, label):
        assert label.ndim == 2, "label must be a 2D array"

        gt_mask = label.copy()
        gt_mask[gt_mask == 255] = 1
        gt_mask += 1
        gt_mask = gt_mask.astype(np.float32)
        categories = np.unique(gt_mask)
        if 0 in categories:
            raise RuntimeError('invalid category')
        gt_mask = cv2.copyMakeBorder(gt_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        height, width = label.shape
        direction_field = np.zeros((2, height + 2, width + 2), dtype=np.float32)
        weight_matrix = np.zeros((height + 2, width + 2), dtype=np.float32)

        for category in categories:
            img = (gt_mask == category).astype(np.uint8)
            weight_matrix[img > 0] = 1. / np.sqrt(img.sum())
            _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                        labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[img > 0] = 0
            place = np.argwhere(index > 0)

            nearCord = place[labels - 1, :]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, height + 2, width + 2))
            nearPixel[0, :, :] = x
            nearPixel[1, :, :] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel

            direction_field[:, img > 0] = diff[:, img > 0]

        weight_matrix = weight_matrix[1:-1, 1:-1]
        direction_field = direction_field[:, 1:-1, 1:-1]

        return direction_field, weight_matrix


class BasicDataset(Dataset):
    def __init__(self, t1_images_dir: str, t2_images_dir: str, labels_dir: str,
                 t1_seg_dir: str, t2_seg_dir: str, all_seg_dir: str, train: bool,
                 t1_mean: list, t1_std: list, t2_mean: list, t2_std: list):

        self.t1_images_dir = Path(t1_images_dir)
        self.t2_images_dir = Path(t2_images_dir)
        self.labels_dir = Path(labels_dir)
        self.t1_seg_dir = Path(t1_seg_dir)
        self.t2_seg_dir = Path(t2_seg_dir)
        self.all_seg_dir = Path(all_seg_dir)
        self.train = train

        # image name without suffix
        self.t1_ids = [splitext(file)[0] for file in listdir(t1_images_dir) if not file.startswith('.')]
        self.t2_ids = [splitext(file)[0] for file in listdir(t2_images_dir) if not file.startswith('.')]
        self.t1_ids.sort()
        self.t2_ids.sort()

        if not self.t1_ids:
            raise RuntimeError(f'No input file found in {t1_images_dir}, make sure you put your images there')
        if not self.t2_ids:
            raise RuntimeError(f'No input file found in {t2_images_dir}, make sure you put your images there')
        assert len(self.t1_ids) == len(self.t2_ids), 'number of t1 images is not equivalent to number of t2 images'
        logging.info(f'Creating dataset with {len(self.t1_ids)} examples')

        self.train_transforms_all = A.Compose([
            #A.Flip(p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], p=0.5),
            A.Transpose(p=0.5),
            A.Rotate(45, p=0.3),
            A.ShiftScaleRotate(p=0.3),
            # ], additional_targets={'image1': 'image'})
        ], additional_targets={
            'image1': 'image',  # 第二幅图像
            'mask_t1_seg': 'mask',  # t1 的分割图
            'mask_t2_seg': 'mask',  # t2 的分割图
            'mask_all_seg': 'mask'  # 所有分割图
        })

        self.train_transforms_image = A.Compose(
            [A.OneOf([
                A.GaussNoise(p=1),
                A.HueSaturationValue(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.Emboss(p=1),
                A.MotionBlur(p=1),
            ], p=ph.noise_p)],
            additional_targets={'image1': 'image'})

        self.t1_normalize = A.Compose([
            A.Normalize(
                mean=t1_mean,
                std=t1_std)
        ])

        self.t2_normalize = A.Compose([
            A.Normalize(
                mean=t2_mean,
                std=t2_std)
        ])

        self.to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={
            'image1': 'image',  # 第二幅图像
            'mask_t1_seg': 'mask',  # t1 的分割图
            'mask_t2_seg': 'mask',  # t2 的分割图
            'mask_all_seg': 'mask'  # 所有分割图
        })

        self.weight_to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={
            'weight_t1': 'mask',  # t1
            'weight_t2': 'mask',  # t2
            'weight_all': 'mask'  # 所有
        })

    def __len__(self):
        """ Return length of dataset."""

        return len(self.t1_ids)

    @classmethod
    def label_preprocess(cls, label):
        """ Binaryzation label."""

        label[label != 0] = 1
        return label

    @classmethod
    def load(cls, filename):
        """Open image and convert image to array."""

        img = Image.open(filename)
        img = np.array(img)

        return img

    def __getitem__(self, idx):
        t1_name = self.t1_ids[idx]
        t2_name = self.t2_ids[idx]
        assert t1_name == t2_name, f't1 name{t1_name} not equal to t2 name{t2_name}'
        t1_img_file = list(self.t1_images_dir.glob(t1_name + '.*'))
        t2_img_file = list(self.t2_images_dir.glob(t2_name + '.*'))
        label_file = list(self.labels_dir.glob(t1_name + '.*'))
        t1_seg_file = list(self.t1_seg_dir.glob(t1_name + '.*'))
        t2_seg_file = list(self.t2_seg_dir.glob(t2_name + '.*'))
        all_seg_file = list(self.all_seg_dir.glob(t1_name + '.*'))
        assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {t1_name}: {label_file}'
        assert len(t1_img_file) == 1, f'Either no image or multiple images found for the ID {t1_name}: {t1_img_file}'
        t1_img = self.load(t1_img_file[0])
        t2_img = self.load(t2_img_file[0])

        label = self.load(label_file[0])
        label = self.label_preprocess(label)
        label_t1_seg = self.load(t1_seg_file[0])
        # label_t1_seg = self.load(label_file[0])
        label_t1_seg = self.label_preprocess(label_t1_seg)
        label_t1_seg = np.where(label == label_t1_seg, 1, 0)

        label_t2_seg = self.load(t2_seg_file[0])
        # label_t2_seg = self.load(label_file[0])
        label_t2_seg = self.label_preprocess(label_t2_seg)
        label_t2_seg = np.where(label == label_t2_seg, 1, 0)

        label_all_seg = self.load(label_file[0])
        label_all_seg = self.label_preprocess(label_all_seg)

        sample = {
            'image': t1_img,
            'image1': t2_img,
            'mask': label,
            'mask_t1_seg': label_t1_seg,
            'mask_t2_seg': label_t2_seg,
            'mask_all_seg': label_all_seg,
        }
        if self.train:
            sample = self.train_transforms_all(**sample)
            t1_img, t2_img, label, label_t1_seg, label_t2_seg, label_all_seg = sample['image'], sample['image1'], \
            sample['mask'], sample['mask_t1_seg'], sample['mask_t2_seg'], sample['mask_all_seg']
            sample = self.train_transforms_image(image=t1_img, image1=t2_img)
            t1_img, t2_img = sample['image'], sample['image1']

        t1_img = self.t1_normalize(image=t1_img)['image']
        t2_img = self.t2_normalize(image=t2_img)['image']

        if self.train:
            if random.choice([0, 1]):
                t1_img, t2_img, label_t1_seg, label_t2_seg = t2_img, t1_img, label_t2_seg, label_t1_seg
        label_t1_seg, weight_t1 = LabelProcessor.label2direction(label_t1_seg)  # 2*H*W
        label_t2_seg, weight_t2 = LabelProcessor.label2direction(label_t2_seg)
        label_all_seg, weight_all = LabelProcessor.label2direction(label_all_seg)
        label_t1_seg = label_t1_seg.transpose(1, 2, 0)  # 2*H*W->H*W*2
        label_t2_seg = label_t2_seg.transpose(1, 2, 0)
        label_all_seg = label_all_seg.transpose(1, 2, 0)

        sample = self.to_tensor(image=t1_img, image1=t2_img, mask=label, mask_t1_seg=label_t1_seg,
                                mask_t2_seg=label_t2_seg, mask_all_seg=label_all_seg)
        sample_weight = self.weight_to_tensor(image=t1_img, weight_t1=weight_t1, weight_t2=weight_t2,
                                              weight_all=weight_all)

        t1_tensor, t2_tensor, label_tensor, label_t1_seg_tensor, label_t2_seg_tensor, label_all_seg_tensor = sample[
            'image'].contiguous(), sample['image1'].contiguous(), sample['mask'].contiguous(), sample[
            'mask_t1_seg'].contiguous(), sample['mask_t2_seg'].contiguous(), sample['mask_all_seg'].contiguous()
        weight_t1_tensor, weight_t2_tensor, weight_all_tensor = sample_weight['weight_t1'].contiguous(), sample_weight[
            'weight_t2'].contiguous(), sample_weight['weight_all'].contiguous()
        name = t1_name
        label_t1_seg_tensor = label_t1_seg_tensor.permute(2, 0, 1)  # H*W*2->2*H*W
        label_t2_seg_tensor = label_t2_seg_tensor.permute(2, 0, 1)
        label_all_seg_tensor = label_all_seg_tensor.permute(2, 0, 1)

        return t1_tensor, t2_tensor, label_tensor, label_t1_seg_tensor, label_t2_seg_tensor, label_all_seg_tensor, weight_t1_tensor, weight_t2_tensor, weight_all_tensor, name
