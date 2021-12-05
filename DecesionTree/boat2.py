import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import *
import seaborn as sns
import datetime

def read_feat(path, test_mode=False):
    df = pd.read_csv(path) #读入指定路径的文件
    df = df.iloc[::-1] #倒序读入数据，目的是使得时间数据正常读入
    if test_mode: #判断是否是测试模式，如果是，则提取标签项
        df['type'] = df['type'].map({'拖网': 0, '围网': 1, '刺网': 2})
        label = np.array(df['type'].iloc[0])
        df = df.drop(['type'], axis=1)
    else:
        label = None
    df.rename(columns = {'渔船ID':'ID','速度':'speed','方向':'direction'},inplace=True) 

    df['dis'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2) ##利用经纬度计算渔船距离
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column].fillna(mean_val, inplace=True) ##对缺失值进行插值

    df = df[["x", "y", "speed", 'direction', 'dis']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))##归一化处理
    features=np.array([df['x'].std(),df['x'].mean(), df['y'].std(),df['y'].mean(),df['speed'].mean(),
    df['speed'].std(),df['direction'].mean(),df['direction'].std(),df['dis'].mean(), df['dis'].std(),])#构建特征向量

    return features,label





def load_data(X_file="./npy/data_x2.npy", Y_file="./npy/data_y2.npy", new=False):
    if os.path.exists(X_file) and os.path.exists(Y_file) and not new:
        X = np.load(X_file)  #如果有特征文件则直接读入
        Y = np.load(Y_file)
        return np.array(X), np.array(Y)
    else:
        path = 'hy_round1_train_20200102'
        train_file = os.listdir(path)
        X = []
        Y = []
        for i, each in enumerate(train_file):
            if not i % 1000:
                print(i)
            each_path = os.path.join(path, each)
            x, y = read_feat(each_path, True) #调用read_feat函数
            if x is not None:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        np.save(X_file, X) #将特征向量存入文件，便于后续机器学习算法调用。
        np.save(Y_file, Y)
        return X, Y
load_data()

