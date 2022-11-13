#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       :utils.py
@Author     :zkniu
@IDE        :Vscode
@Time       :2022-11-11 11:35:22
"""

import numpy as np
from math import log
import pandas as pd
# 信息熵
def entropy(ele):
    """
    @description: 
    @param {ele}
    ele:包含类别取值的列表 
    @return: 
    """
    # 计算概率分布
    probs = [ele.count(i)/len(ele) for i in set(ele)]
    # 计算信息熵
    entropy = -sum([p*log(p,2) for p in probs])
    return entropy

# 信息增益


# 信息增益比

# 基尼系数
def gini(ele):
    """
    @description: 
    @param {type} 
    @return: 
    """
    probs = [ele.count(i)/len(ele) for i in set(ele)]
    gini = sum([p*(1-p) for p in probs])
    return gini

# 数据集划分函数
def df_split(df,col):
    """
    @description: 
    @param {type} 
    @return: 
    """
    # 获取依据特征的不同取值
    unique_col_val = df[col].unique()
    # 创建划分结果的数据框字典
    res_dict = {elem:pd.DataFrame for elem in unique_col_val}
    # 根据特征值进行划分
    for key in res_dict.keys():
        res_dict[key] = df[:][df[col] == key]

    return res_dict
