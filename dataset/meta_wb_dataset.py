import os
import cv2
import time
import scipy.io
import numpy as np
import json
import torch
import torch.utils.data as data

from .config import COLOR_CHECKER_PATH, NUS_8_PATH
from .utils import augmentation_im, augmentation_im_and_illumination

# todo: support cube+ dataset
class MetaWBDataset(data.Dataset):
    def __init__(self, camera_list, train=True, fold=0):
        self.train = train
        with open('data/color_checker.json', 'r') as f:
            meta_infos = json.load(f)#返回字典
        with open('data/nus.json', 'r') as f:
            meta_infos += json.load(f)

        self.meta_infos = []
        for m in meta_infos:
            if m['camera'] in camera_list and ( (train and m['fold'] != fold) or (not train and m['fold'] == fold) ):
                # for color checker
                if m['camera'] in ['canon1d', 'canon5d']:
                    m['img_path'] = os.path.join(COLOR_CHECKER_PATH, m['img_name'])
                # for nus-8
                else:
                    m['img_path'] = os.path.join(NUS_8_PATH, m['img_name'].split('_')[0], 'PNG', m['img_name'])
                self.meta_infos.append(m)

        if train:
            self.illumination_tables = np.array([m['illumination'] for m in self.meta_infos], dtype=np.float32)
            self.illumination_tables = self.illumination_tables/np.linalg.norm(self.illumination_tables, ord=2, axis=1, keepdims=True)

    def __getitem__(self, idx):
        meta = self.meta_infos[idx]
        
        # read img
        raw = cv2.imread(meta['img_path'], cv2.IMREAD_UNCHANGED)
        raw = raw[..., ::-1]#??
        raw = raw.astype(np.float32)
        
        # linearization
        blc, saturate = meta['black_level'], meta['saturation']
        raw = (raw-blc) / (saturate-blc)
        raw = np.clip(raw, 0, 1)
        
        # mask mcc
        polygon = np.array(meta['polygon'], dtype=np.float32)
        polygon = polygon * np.array([raw.shape[1], raw.shape[0]])
        polygon = polygon.astype(np.int32)
        cv2.fillPoly(raw, [polygon], (0, 0, 0,))
        
        # read illumination
        illumination = np.array(meta['illumination'], dtype=np.float32)
        illumination = illumination/np.linalg.norm(illumination, ord=2, axis=0, keepdims=True)

        raw = torch.from_numpy(raw.transpose([2, 0, 1]).copy())
        illumination = torch.from_numpy(illumination.copy())
        return raw, illumination, meta['img_name']

    def __len__(self):
        return(len(self.meta_infos)) #照片数目
