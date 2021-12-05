import os
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import *
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

def feature_engineer(df, test=False):
    df['speed'] = df['speed']
    df['ori'] = df['ori'] / 180.0 * np.pi
    df['speed_sin'] = df['speed'] * np.sin(df['ori'])
    df['speed_cos'] = df['speed'] * np.cos(df['ori'])
    
    if test:
        df = df.groupby(['id']).agg({'x': ['std', 'min', 'max', 'mean'], 
                                     'y': ['std', 'min', 'max', 'mean'], 
                                     'speed_sin': ['std', 'min', 'max', 'mean'], 
                                     'speed_cos': ['std', 'min', 'max', 'mean'], 
                                     'speed': ['std', 'min', 'max', 'mean'], 
                                     'ori': ['std', 'min', 'max', 'mean']}).reset_index()

        df.columns = ['id', 
                      'x_std', 'x_min', 'x_max', 'x_mean',
                      'y_std', 'y_min', 'y_max', 'y_mean', 
                      'speed_sin_std', 'speed_sin_min', 'speed_sin_max', 'speed_sin_mean', 
                      'speed_cos_std', 'speed_cos_min', 'speed_cos_max', 'speed_cos_mean',
                      'speed_std', 'speed_min', 'speed_max', 'speed_mean', 
                      'ori_std', 'ori_min', 'ori_max', 'ori_mean']
        
    else:
        df = df.groupby(['id', 'type']).agg({'x': ['std', 'min', 'max', 'mean'], 
                                             'y': ['std', 'min', 'max', 'mean'], 
                                             'speed_sin': ['std', 'min', 'max', 'mean'], 
                                             'speed_cos': ['std', 'min', 'max', 'mean'],
                                             'speed': ['std', 'min', 'max', 'mean'], 
                                             'ori': ['std', 'min', 'max', 'mean']}).reset_index()
        df.columns = ['id', 'type', 
                      'x_std', 'x_min', 'x_max', 'x_mean',
                      'y_std', 'y_min', 'y_max', 'y_mean', 
                      'speed_sin_std', 'speed_sin_min', 'speed_sin_max', 'speed_sin_mean', 
                      'speed_cos_std', 'speed_cos_min', 'speed_cos_max', 'speed_cos_mean',
                      'speed_std', 'speed_min', 'speed_max', 'speed_mean', 
                      'ori_std', 'ori_min', 'ori_max', 'ori_mean']  
    return df

important = ['x_std', 'x_min', 'x_max', 'x_mean', 
            'y_std', 'y_min', 'y_max', 'y_mean', 
            'speed_sin_std', 'speed_sin_max', 'speed_sin_mean', 
            'speed_cos_std', 'speed_cos_max', 'speed_cos_mean',
            'speed_std', 'speed_max', 'speed_mean', 
            'ori_std', 'ori_max', 'ori_mean']

num = ['id', 
       'x_std', 'x_min', 'x_max', 'x_mean', 
       'y_std', 'y_min', 'y_max', 'y_mean', 
       'speed_sin_std', 'speed_sin_min', 'speed_sin_max', 'speed_sin_mean', 
       'speed_cos_std', 'speed_cos_min', 'speed_cos_max', 'speed_cos_mean',
       'speed_std', 'speed_min', 'speed_max', 'speed_mean', 
       'ori_std', 'ori_min', 'ori_max', 'ori_mean']

X = pd.read_feather('train.feather')
X.columns = ['id', 'x', 'y', 'speed', 'ori', 'time', 'type']
type_dict = {'围网':0, '拖网':1, '刺网':2}
X.type = X.type.map(type_dict)
X = feature_engineer(X)
scaler = StandardScaler()
X[important] = scaler.fit_transform(X[important])
Y = X['type']
X = X[important]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# 将参数写成字典下形式
parameters = [
    # {'kernel': ['rbf'],   #核函数
    #  'C': [10,50,100,500,1000], #惩罚因子
    #  'gamma': [0.1, 1, 10], #径向基函数的系数
    #  'class_weight': ['balanced'] #样本均衡度
    #  },
     {'kernel':['linear'],
     'C':[10,50,100,500,1000], 'class_weight': ['balanced']
     },
     # {'kernel':['poly'],
     # 'C': [50,100,500,1000],'class_weight': ['balanced']
     # },
     # {
     # 'kernel':['sigmoid'],
     # 'gamma':[0.1,1,10],
     # 'C': [10,50,100,500,1000],
     # }
]
# 参数调优 
clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=5, n_jobs=5, scoring='f1_macro')
clf.fit(X_train, y_train) 
print("Best set score:{:.2f}".format(clf.best_score_))
print("Best parameters:{}".format(clf.best_params_))
print("Test set score:{:.2f}".format(clf.score(X_test, y_test)))

#模型测试
predicted = clf.predict(X_test)  # 模型预测
accuracy = accuracy_score(y_test, predicted)
print("accuracy", accuracy)
print("f1_score",f1_score(y_test,predicted,labels=[0,1,2],average='macro'))