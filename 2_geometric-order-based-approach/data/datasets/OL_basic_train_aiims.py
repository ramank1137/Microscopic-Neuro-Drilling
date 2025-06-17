import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A




class OLBasic_Train(Dataset):
    def __init__(self, imgs, labels, transform, tau, norm_age=True, logscale=False, is_filelist=False):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        multiply = {
            1:5,
            2:2,
            3:1,
            4:1,
            5:1,
            6:1,
            7:1,
            8:2,
            9:5,
            10:10

        }
        #imgs, labels = [], []
        #for img, label in zip(self.imgs, self.labels):
        #    imgs += [img,]*multiply[label]
        #    labels += [label]*multiply[label]
        #self.imgs = np.array(imgs)
        #self.labels = np.array(labels)
        self.transform = transform
        self.n_imgs = len(self.imgs)
        self.min_age_bf_norm = self.labels.min()
        if logscale:
            self.labels = np.log(labels.astype(np.float32))
        else:
            if norm_age:
                self.labels = self.labels - min(self.labels)

        self.max_age = self.labels.max()
        self.min_age = self.labels.min()
        self.tau = tau
        self.is_filelist = is_filelist

        # mapping age to rank : because there are omitted ages
        rank = 0
        self.mapping = dict()
        for cls in np.unique(self.labels):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])

    def __get_aug_image(self, image):
        
        transform = A.Compose(
            [#A.CLAHE(p=1),
                A.RandomRotate90(),
                #A.Transpose(),
                #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                #                   rotate_limit=45, p=.75),
                #A.Blur(blur_limit=10),
                A.GaussianBlur(blur_limit=(1, 9), p=0.5),
                #A.OpticalDistortion(p=1),
                A.GridDistortion(always_apply=True, p=0.5),
                A.HueSaturationValue(p=0.5)
                #A.CoarseDropout(2, 0.2, 0.2, None, 0.1, 0.1, 0,None,always_apply=True,p=0.5)
                #A.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True, p=1)
            ])
        augmented_image = transform(image=np.asarray(image))['image']
        return Image.fromarray(augmented_image)

    def get_transform(self):

        transform = transforms.Compose([
            #transforms.Resize((self.img_size, self.img_size)),
            #transforms.RandomCrop(self.img_size, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ])
        return transform

    def transform_new(self, img):
        img = self.__get_aug_image(img)
        transform = self.get_transform()
        return transform(img)

    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference(self.labels[item], self.labels, min_rank=self.min_age,
                                                   max_rank=self.max_age)

        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        base_img = self.transform_new(base_img)
        ref_img = self.transform_new(ref_img)

        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]
        
        return base_img, ref_img, order_label, [base_rank, ref_rank], item

    def __len__(self):
        return self.n_imgs

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        rng = np.random.default_rng()
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank + tau
                ref_range_min = min_rank
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx