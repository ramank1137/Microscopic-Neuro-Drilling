import os
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_lds_kernel_window

print = logging.info


class AgeDB(data.Dataset):
    def __init__(self, df, data_dir, img_size, group, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        group = group.split("_")
        group = group[0] + "_class_constraint/" + group[1]
        fld = "/home/raman/Work/big/scoring_data/drilling/10 Fold/" + group
        #if split=="val":
        #    split = "test"
        fld = fld + "/" + self.data_dir + "/" + split + "/image"
        self.files = [fld + "/" + i for i in os.listdir(fld) if ".png" in i]
        self.files = self.files + self._get_files_class(9, times = 6)
        self.labels = [float(int(i.split("_")[-1].replace(".png",""))/10) for i in self.files]
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)
    
    def _get_augmentation_files(self, fld):
        aug_fld = "/home/raman/Work/big/scoring_data/drilling/augmentation/aug/"
        if self.split == "train":
            fls = []
            for fl in os.listdir(aug_fld):
                if ".png" in fl and (fl.split("_")[0] + "_" +"_".join(fl.split("_")[2:])) in os.listdir(fld):
                    fls.append(aug_fld + fl)
            return fls
        else:
            return []
        
    def _get_files_class(self, clss, times=6):
        if self.split == "train":
            fls = [i for i in self.files if i.split("_")[-1].replace(".png", "") == str(clss)]
            fls = fls*6
        else:
            fls = []
        return fls
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #index = index % len(self.df)
        file = self.files[index]
        img = Image.open(os.path.join(self.data_dir, file)).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([float(int(file.split("_")[-1].replace(".png",""))/10),]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        if self.split == "val":
            return img, label, weight, file
        
        return img, label, weight

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

    def _prepare_weights(self, reweight, max_target=10, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):

        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.labels
        #labels = self.df['age'].values
        for label in labels:
            value_dict[min(max_target - 1, int(label*10) - 1)] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label*10) - 1)] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label*10) - 1)] for label in labels]
        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        import ipdb
        ipdb.set_trace()
        return weights
