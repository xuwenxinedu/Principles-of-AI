import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import *
import seaborn as sns

path = 'hy_round1_train_20200102'
#读取数据
print("===读取数据===")
train_file = os.listdir(path) 
data=[]
for each in train_file:  
    each_path=os.path.join(path,each) 
    temp = pd.read_csv(each_path)[['渔船ID','x','y','速度','方向','time','type']] 
    data.append(temp)
train = pd.concat(data) 
train.rename(columns = {'渔船ID':'ID','速度':'speed','方向':'direction'},inplace=True) 

#数据预处理
print()
print("===数据预处理===")
train['type'] = train['type'].map({'拖网':0,'围网':1,'刺网':2}) 
train.head() 
train.describe()
print(train.isnull().sum(),'\n\n', train['type'].value_counts())
train.dropna(inplace=True)
print(train.isnull().sum())

#数据可视化
print()
print("===数据可视化===")
f,[ax1,ax2] = plt.subplots(1,2,figsize = (20,10)) 
train['type'].value_counts().plot.pie(autopct = '%1.2f%%',ax = ax1) 
sns.countplot(x = 'type',data = train,ax=ax2) 
f.suptitle('Fishing type') 
plt.show()

#回归模型训练及结果分析
print()
print("===回归模型训练及结果分析===")
X_train=train[['ID','x','y','speed','direction' ]] 
Y_train = train[['type']] 
print(X_train.head())
model = LinearRegression() ##生成对象
model.fit(X_train, Y_train) ##进行回归拟合
print(model.intercept_)  
print(model.coef_)

#双变量关系初步分析
train.groupby(['speed','type'])['type'].count()

plt.figure(figsize = (20,10)) ##构图
sns.countplot(x = 'speed',hue = 'type',data = train) 
ax1.set_title('Speed -> type influence')

interval = [0,1,5,10,20,30,50,110]
train['speed'] = pd.cut(train['speed'], interval, labels = ['0','1','2','3','4','5','6'])
train.groupby(['speed','type'])['type'].count() 
plt.figure(figsize = (20,10))
sns.countplot(x = 'speed',hue = 'type',data = train)
ax1.set_title('Speed -> type influence')

plt.show()