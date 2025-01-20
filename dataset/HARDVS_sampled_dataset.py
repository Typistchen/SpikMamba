
import sys
root_path = "/home/dsz/Documents/eventcamera/ExACT"
sys.path.append(root_path)


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import struct
import copy, json
import numpy as np
import matplotlib.pyplot as plt
import cv2, os
import pickle
import yaml
# from spikingjelly.datasets.cifar10_dvs import load_events
import spikingjelly.datasets as sjds
# from model.utils.yaml import read_yaml
from os.path import join, abspath, dirname

def read_yaml(yaml_path):
    # 读取Yaml文件方法
    with open(yaml_path, encoding="utf-8", mode="r") as f:
        result = yaml.load(stream=f, Loader=yaml.FullLoader)
        return result
THIS_DIR = abspath("/home/dsz/Documents/eventcamera/ExACT")

class HARDVS_sampled(Dataset):
    def __init__(self, txtPath, tfPath):
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)
        tf = open(tfPath, "r")
        self.classnames_dict = json.load(tf)  # class name idx start from 0
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx:
        :return: events_image 3,T,H,W
                 image 3,H,W
                 label_idx 0 to cls 1
        """
        event_stream_path = self.files[idx]
        events = []
        # for i in range(len(event_stream_path)):
        #     event_frame = cv2.imread(event_stream_path[i])
        #     events.append(np.array(event_frame))
        # events = np.array(events)
        data = np.load(event_stream_path)
        events = data['images']

        if events.ndim < 4:
            events = events[np.newaxis,:,:,:]

        events_data = np.array(events).transpose(3,0,1,2) / 255.0
        # events_data = torch.from_numpy(events_data)
        # print(event_stream_path)

        # label
        label_idx = int(event_stream_path.split('/')[7][7:]) -1
        # print(label_idx)

        return events_data, label_idx

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line.strip())
        # random.shuffle(self.files)
        return len(self.files)

def visualize_img(grayscale_img, classnames_list, labels):
    # print(image.shape)
    B,T,H,W,C = grayscale_img.shape
    plt_num = int((T + 1) / 5 + 1)
    for j in range(B):
        plt.figure()
        for i in range(T):
            img = grayscale_img[j, i, :, :, :]
            plt.subplot(plt_num, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            # plt.title('Frame No.'+str(i), loc='center')
        class_name = classnames_list[labels[j]]
        plt.title(class_name, loc='center')
        plt.show()

def pad_event(event, max_event_length):
    C, N, H, W = event.shape
    pad_num = max_event_length - N
    if pad_num > 0:
        pad_zeros = np.zeros((C, pad_num, H, W))
        event = np.concatenate((event, pad_zeros), axis=1)

    return event

def event_sampled_frames_collate_func(data):
    """
    Pad event data with various number of sampled frames among a batch.

    """
    events = [data[i][0] for i in range(len(data))]
    actual_event_length = [events[i].shape[1] for i in range(len(events))]
    max_event_length = max(actual_event_length)
    events = np.array(events)
    # padded_events = np.array([pad_event(events[i], 16) for i in range(len(events))])
    labels = np.array([data[i][1] for i in range(len(data))])
    actual_event_length = np.array(actual_event_length)

    return events, actual_event_length, labels



# cfg = read_yaml(THIS_DIR + '/Configs/HARDVS.yaml')
# val_dataset = HARDVS_sampled(cfg['Dataset']['Val']['Path'], cfg['Dataset']['Classnames'])
# val_loader = DataLoader(val_dataset, batch_size=cfg['Dataset']['Val']['Batch_size'],
#                             shuffle=False, drop_last=True, num_workers=16, prefetch_factor=2, pin_memory=True,
#                             collate_fn=event_sampled_frames_collate_func)

# a = 0
# i = 0
# for batch in val_loader:
#     events, actual_event_length, labels = batch
#     i  = i + 1
#     print(i)
#     print("events shape:", events.shape)
#     print("labels shape:", labels.shape)



