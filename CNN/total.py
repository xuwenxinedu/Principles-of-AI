import pandas as pd
import os

path = './hy_round1_testA_20200102'
total = pd.DataFrame()
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename == '7000.csv':
            part = pd.read_csv(f'./hy_round1_testA_20200102/{filename}')
            total = part
        else:
            print(filename)
            part = pd.read_csv(f'./hy_round1_testA_20200102/{filename}')
            total = pd.concat([total, part], axis=0, ignore_index=True)

total.to_feather("test.feather")
nn = pd.read_feather('./test.feather')
print(total)
print(nn)