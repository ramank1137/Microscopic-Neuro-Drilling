import pickle
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
import random

from data.datasets import OL_basic_train_aiims, basic


multiply_gen = {
            1:4,
            2:2,
            3:1,
            4:1,
            5:1,
            6:3,
            7:8,
            8:20,
            9:20,
            10:20

        }



def multipy_(imgs, labels, factor):
    list_for_class = {}
    total_list = []
    for im, lb in zip(imgs, labels):
        if lb not in list_for_class:
            list_for_class[lb] = []
        list_for_class[lb].append(im)
    for lb, imgs in list_for_class.items():
        #import ipdb
        #ipdb.set_trace()
        total_list+= imgs * int(factor[lb]//1) + random.sample(imgs, round(len(imgs)*(factor[lb]%1)))
    
    return total_list, [int(i.split("_")[-1][:-4]) for i in total_list]


def multiply(imgs, labels, factor):
    image_list = []
    label_list = []
    for im, lb in zip(imgs, labels):
        image_list+=[im,]*factor[lb]
        label_list += [lb,]*factor[lb]
    return np.array(image_list), np.array(label_list)


def get_datasets(cfg):
    tr_std = None
    te_std = None
    if cfg.dataset =='morph':
        img_root = cfg.img_root
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
        tr_ages = tr_list[:, cfg.lb_idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
        te_ages = te_list[:, cfg.lb_idx]

    elif cfg.dataset == 'aiims':
        img_root = cfg.img_root
        #import ipdb
        #ipdb.set_trace()
        df = pd.read_csv("/mnt/f/Work/Dataset/new_score_split.csv")
        #file_index = {}
        #for i, row in df.iterrows():
        #    file_index["/home/raman/Work/big/drilling_data/images/" + row["Index"]] = row["index"]
        #cfg.file_index = file_index
        #df = df[df["scapula or head"]=="h"]
        splt = "Split"
        tr_imgs = [img_root + str(i) + ".png" for i in list(df[~(df[splt]==int(cfg.fooled))]["Index"])]
        tr_labels = np.array([i for i in list(df[~(df[splt]==int(cfg.fooled))]["GT"])])
        tr_imgs, tr_labels = multiply(tr_imgs, tr_labels, multiply_gen)
        #tr_labels = np.array(tr_labels)
        #import ipdb
        #ipdb.set_trace()

        vl_imgs = [img_root + str(i) + ".png" for i in list(df[(df[splt]==int(cfg.fooled))]["Index"])]
        vl_labels =  np.array([i for i in list(df[(df[splt]==int(cfg.fooled))]["GT"])])
    else:
        pass

    loader_dict = dict()
    loader_dict['train'] = DataLoader(OL_basic_train_aiims.OLBasic_Train(tr_imgs, tr_labels, cfg.transform_tr, cfg.tau, logscale=cfg.logscale, is_filelist=cfg.is_filelist),
                                      batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    loader_dict['train_for_val'] = DataLoader(basic.Basic(tr_imgs, tr_labels, cfg.transform_te, is_filelist=cfg.is_filelist, norm_age=False),
                                    batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                                    num_workers=cfg.num_workers)

    loader_dict['val'] = DataLoader(basic.Basic(vl_imgs, vl_labels, cfg.transform_te, is_filelist=cfg.is_filelist, std=te_std, norm_age=False),
                                     batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    return loader_dict





