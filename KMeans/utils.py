# -*- coding: utf-8 -*-
# @Time    : 2022-10-12 11:21
# @Author  : Zhikang Niu
# @FileName: utils.py
# @Software: PyCharm

import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def read_h5_file(file_path):
    """
    Read h5 file
    :param file_path:
    :return: tuple(train_data, train_label), tuple(test_data, test_label)
    """
    with h5py.File(file_path, 'r') as f:
        test_data = np.asarray(f['test']['data'])
        test_label = np.asarray(f['test']['target'])
        train_data = np.asarray(f['train']['data'])
        train_label = np.asarray(f['train']['target'])
        train_data = np.hstack((train_data, train_label.reshape(-1, 1)))
        test_data = np.hstack((test_data, test_label.reshape(-1, 1)))

    return train_data, test_data


def read_csv_file(file_path):
    """
    Read csv file
    :param file_path:
    :return: data, label
    """
    data = []
    label = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            data.append(list(map(float,line[:-1])))
            label.append(line[-1])

    data = normalize(data)
    return np.asarray(data), np.asarray(label)


def read_data(file_path, split_train_test=True, test_size=0.2):
    """
    Read data from file
    :param file_path:
    :return: data, label
    """
    assert os.path.exists(file_path), 'File not found'
    assert file_path.endswith('.csv') or file_path.endswith('.h5'), 'File format error'

    if file_path.endswith('.h5'):
        train_data, test_data = read_h5_file(file_path)
        return train_data, test_data

    elif file_path.endswith('.csv'):
        if split_train_test:
            data, label = read_csv_file(file_path)
            train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=test_size,
                                                                              random_state=42)
            train_data = np.hstack((train_data, train_label.reshape(-1, 1)))
            test_data = np.hstack((test_data, test_label.reshape(-1, 1)))
            return train_data, test_data
        else:
            data, label = read_csv_file(file_path)
            return data, label


def normalize(data):
    """
    Normalize data
    :param data:
    :return:
    """
    if isinstance(data, list):
        data = np.asarray(data)
    # print(np.max(data))
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


if __name__ == '__main__':
    file_path = './knn_data/iris.csv'
    train_data, test_data = read_data(file_path)
    print(train_data[:-1])
    # print(train_data[:,:-1])
    train_data = list(map(float, train_data[:,-1]))
    print(train_data)
    print(train_data[:, -1])
    print(test_data.shape)
    # print(train_data)
    # print(test_data)
