# -*- coding: utf-8 -*-
# @Time    : 2022-10-24 14:12
# @Author  : Zhikang Niu
# @FileName: Kmeans.py
# @Software: PyCharm
import os
import numpy as np
import random
from utils import read_data
from KNN import compute_distance

"""
KMeans代码的流程
创建k个点作为初始的质心点（随机选择） -> 如何选择一个合适的k值
当任意一个点的簇分配结果发生改变时
    对数据集中的每一个数据点
        对每一个质心
            计算质心与数据点的距离
        将数据点分配到距离最近的簇
    对每一个簇，计算簇中所有点的均值，并将均值作为质心
"""

class KMeans():
    def __init__(self, file_path,seed=2022):
        self.train_data, self.test_data = read_data(file_path)
        self.file_name = os.path.basename(file_path).split('.')[0]
        self.cluster = {}
        self._set_seed(seed)

    def _set_seed(self,seed=2022):
        np.random.seed(seed)
        random.seed(seed)

    def _initialize_ceter(self,K):
        k = 0
        indices = []

        while True:
            index = np.random.randint(self.train_data.shape[0])
            if index not in indices:
                indices.append(index)
                k += 1
                if k == K:
                    break
        init_center = self.train_data[indices]

        center = {str(i): list(map(float, center[:-1])) for i in range(K) for center in init_center}
        self.cluster = {str(i):[list(map(float,center[:-1]))]for i in range(K) for center in init_center}
        return center

    def _pred_label(self, pred_point, center):
        """
        预测单个点的标签
        :param pred_point:
        :param ceter:
        :return:
        """

        res = [
            {'label': center_label,
             'dist': compute_distance(pred_point, np.asarray(center_point))}
            for center_label,center_point in center.items()
        ]

        res = sorted(res, key=lambda x: x['dist'])

        distance,label = res[0]['dist'],res[0]['label']
        return distance,label

    def _update_center(self):
        """
        更新质心点
        :param center:
        :param cluster:
        :return:
        """
        k = len(self.cluster.keys())
        center = {str(i): np.mean(np.asarray(self.cluster[str(i)]), axis=0) for i in range(k)}
        return center

    def test(self,k):
        """
        预测测试集的标签
        :param k:
        :return:
        """

        # 1. 随机选择k个点作为初始的质心点
        cluserChange = True
        center = self._initialize_ceter(k)
        # 计算每个点到质心点的距离
        while cluserChange:
            for point in self.train_data:
                # 计算每个点到质心的距离
                point = list(map(float, point[:-1]))
                point = np.asarray(point)
                dist,label = self._pred_label(point, center)
                self.cluster[label].append(list(point))
            # 重新计算每个簇的质心点
            new_center = self._update_center()
            sum = 0
            for key in new_center.keys():
                sum += np.sum(np.abs(np.asarray(new_center[key]) - np.asarray(center[key])))
            if sum > 1e-5:
                cluserChange = True
                center = new_center
                self.cluster = {str(i): [c] for i, c in center.items()}
            else:
                cluserChange = False
        print('最终的质点为：',center)
        print('最终的簇为：',self.cluster)





if __name__ == '__main__':
    file_path = './knn_data/iris.csv'
    kmeans = KMeans(file_path)
    kmeans.test(3)



