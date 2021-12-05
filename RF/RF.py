import pandas as pd
import numpy as np


#这里采用x，y，速度，方向四个属性中的最大值，最小值，平均值以及标准差作为每个渔船的特征
def get_feature(df, train_mode=True):
    """
    test_mode: 用于区分训练数据和测试数据，训练数据存在label而测试数据不存在label
    """
    df = df.iloc[::-1]

    if train_mode:
        df['type'] = df['type'].map({'拖网': 0, '围网': 1, '刺网': 2})# 将label由str类型转换为int类型
        label = np.array(df['type'].iloc[0])
        df = df.drop(['type'], axis=1)
    else:
        label = None
    features = np.array([df['x'].std(), df['x'].mean(), df['x'].max(), df['x'].min(),
                df['y'].std(), df['y'].mean(), df['y'].max(), df['y'].min(),
                df['速度'].mean(), df['速度'].std(), df['速度'].max(), df['速度'].min(),
                df['方向'].mean(), df['方向'].std(), df['方向'].max(), df['方向'].min(),
                ])
    return features, label

import os
def load_data():
    path = 'hy_round1_train_20200102'
    train_file = os.listdir(path)
    X = []
    Y = []
    for i, each in enumerate(train_file):
        if not i % 1000:  #每读1000个文件输出一次
            print(i)
        each_path = os.path.join(path, each)
        df = pd.read_csv(each_path)
        x, y = get_feature(df)
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y



# X,Y = load_data()
# np.save("./npy/x_array.npy", X)
# np.save("./npy/y_array.npy", Y)



X = np.load("./npy/x_array.npy")
Y = np.load("./npy/y_array.npy")





from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split



def eval(clf,X_train, X_test, y_train, y_test):
    predicted = clf.predict(X_train)  # 模型预测
    accuracy = accuracy_score(y_train, predicted)
    print("训练集准确率", accuracy)
    f1 = f1_score(y_train, predicted,average='macro')
    print("训练集f1_score", f1)
    
    predicted = clf.predict(X_test)  
    accuracy = accuracy_score(y_test, predicted)
    print("测试集准确率", accuracy)
    f1 = f1_score(y_test, predicted,average='macro')
    print("测试集f1_score", f1)
    
    plot_confusion_matrix(clf, X_test, y_test,cmap="GnBu") 


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

from sklearn.model_selection import GridSearchCV


print("\n\n===============随机森林==============\n")
from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators':[100,200,500],'max_depth':range(15,20),
          'criterion':['entropy','gini'],"class_weight":['balanced'],"random_state":[2021],}
clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, cv=8, n_jobs=5, scoring="f1_macro")
clf.fit(X_train,y_train)
print("Best set score:{:.2f}".format(clf.best_score_))
print("Best parameters:{}".format(clf.best_params_))
print("Test set score:{:.2f}".format(clf.score(X_test, y_test)))
eval(clf,X_train, X_test, y_train, y_test)
