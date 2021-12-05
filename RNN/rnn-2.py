import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

LEN = 400
N = 5
MIN_LEN = 300

def read_feat(path, test_mode=False):
    df = pd.read_csv(path)
    df = df.iloc[::-1]

    if test_mode:
        df['type'] = df['type'].map({'拖网': 0, '围网': 1, '刺网': 2})
        Y = np.array(df['type'].iloc[0])
    else:
        Y = None

    df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
    X = df[["x", "y", "速度", '方向']].apply(lambda x: (x - np.mean(x)) / np.std(x))
    for column in list(X.columns[X.isnull().sum() > 0]):
        mean_val = X[column].mean()
        X[column].fillna(mean_val, inplace=True)
    X = X.dropna(axis=0)
    X = np.array(X)
    cols = X.shape[1]
    rows = X.shape[0]
    if rows < MIN_LEN:
        return None, None, 0
    for i in range(rows, LEN):
        b = np.zeros((1, cols))
        for j in range(N):
            b += X[i - j - 1]
        X = np.row_stack((X, b / N))
    return X[:LEN], Y


def load_data(X_file="./npy/data_x.npy", Y_file="./npy/data_y.npy", new=False):
    if os.path.exists(X_file) and os.path.exists(Y_file) and not new:
        X = np.load(X_file)
        Y = np.load(Y_file)
        return np.array(X), np.array(Y)
    else:
        path = './data/hy_round1_train_20200102'
        train_file = os.listdir(path)
        X = []
        Y = []
        for i, each in enumerate(train_file):
            if not i % 1000:
                print(i)
            each_path = os.path.join(path, each)
            x, y= read_feat(each_path, True)
            if x is not None:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        np.save(X_file, X)
        np.save(Y_file, Y)
        return X, Y
 
#按特定比例进行训练集切分
X, Y = load_data()
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# 模型构建
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Bidirectional, Dense, LSTM
model = Sequential()
model.add(Bidirectional(LSTM(32,input_shape=(X_train.shape[1],X_train.shape[2]))))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
# 训练模型
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
# 绘制准确率图像
data = {'accuracy': history.history['accuracy'], 'val_accuracy': history.history['val_accuracy']}
pd.DataFrame(data).plot(figsize=(8, 5))
plt.grid(True)
plt.axis([0, 10, 0, 1])
plt.show()
# 绘制损失图像
data = {}
data['loss'] = history.history['loss']
data['val_loss'] = history.history['val_loss']
pd.DataFrame(data).plot(figsize=(8, 5))
plt.grid(True)
plt.axis([0, 10, 0, 1])
plt.show()
predicted = model.predict(X_test)  # 模型预测
predicted = [list(x).index(max(x)) for x in predicted]
accuracy = accuracy_score(y_test, predicted)
print("accuracy", accuracy)
print("f1_score",f1_score(y_test,predicted,labels=[0,1,2],average='macro'))
