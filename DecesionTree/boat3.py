import numpy as np
import graphviz
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split


def get_feature(df, train_mode=True):
 # test_mode: 用于区分训练数据和测试数据，训练数据存在label而测试数据不存在label
    df = df.iloc[::-1]
    if train_mode:
        df['type'] = df['type'].map({'拖网': 0, '围网': 1, '刺网': 2})
# 将label由str类型转换为int类型
        label = np.array(df['type'].iloc[0])
        df = df.drop(['type'], axis=1)
    else:
        label = None
    features = np.array([df['x'].std(), df['x'].mean(), df['x'].max(), df['x'].min(), df['y'].std(), df['y'].mean(), df['y'].max(), df['y'].min(), df['速度'].mean(), df['速度'].std(), df['速度'].max(), df['速度'].min(), df['方向'].mean(), df['方向'].std(), df['方向'].max(), df['方向'].min(), ])
    return features, label
#读入数据函数

def load_data():
    path = './hy_round1_train_20200102'
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

def eval(clf, X_train, X_test, y_train, y_test):
    predicted = clf.predict(X_train)  # 训练集上进行模型预测
    accuracy = accuracy_score(y_train, predicted)
    print("训练集准确率", accuracy)
    f1 = f1_score(y_train, predicted,average='macro')
    print("训练集f1_score", f1)
    
    predicted = clf.predict(X_test)  # 测试集上进行模型预测
    accuracy = accuracy_score(y_test, predicted)
    print("测试集准确率", accuracy)
    f1 = f1_score(y_test, predicted,average='macro')
    print("测试集f1_score", f1)



# X,Y = load_data()
# #将特征向量写入特征文件中，便于后续快速读取
# np.save("./npy/x_array.npy", X)
# np.save("./npy/y_array.npy", Y)

#读入特征文件数据
X = np.load("./npy/x_array.npy")
Y = np.load("./npy/y_array.npy")
#把数据切分为训练集与测试两个部分，一般采用8:2或7:3比例
#使用sklearn决策树对部分数据进行分类
#在深度学习中跑一次模型可能会花很长的时间，这时候先使用部分数据既可以方便程序的debug，对模型运行时间有个大致的概念，并理解训练数据太少导致过拟合现象。

#切分数据集
X_train, X_test, y_train, y_test = train_test_split(X[:], Y[:], test_size=0.2, random_state=0) 
#模型构建，参数random_state用来设置分枝中的随机模式的参数，赋值为任意整数可以使模型在同一个训练集和测试集下稳定
clf=tree.DecisionTreeClassifier(random_state=2021) 
#模型训练
clf=clf.fit(X_train,y_train)  
#可视化

feature_name = ['x方差','x均值','x最大值','x最小值','y方差','y均值','y最大值','y最小值','速度均值','速度方差','速度最大值','速度最小值','方向均值','方向方差','方向最大值','方向最小值']
dot_data = tree.export_graphviz(clf,out_file = None,feature_names= feature_name,class_names=["0","1","2"],filled=True,rounded=True)
graph = graphviz.Source(dot_data)

graph.render('tree',view=True)
    
plot_confusion_matrix(clf, X_test, y_test,cmap="GnBu")

eval(clf,X_train, X_test, y_train, y_test)