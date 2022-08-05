import os
import numpy as np
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, folder, data_name, label_name, transform=None):
        train_set, train_labels = load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder, data_name, label_name):
    """
    :param data_folder: 文件目录
    :param data_name: 数据文件
    :param label_name: 标签数据文件
    :return:
    """
    with open(os.path.join(data_folder, label_name), 'rb') as lbfile:
        y_train = np.frombuffer(lbfile.read(), np.uint8, offset=8)

    with open(os.path.join(data_folder, data_name), 'rb') as imgfile:
        x_train = np.frombuffer(imgfile.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    return x_train, y_train