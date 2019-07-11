import pickle
import numpy as np


label = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']
X_train = []
y_train = []
for l in label:
    print("Working on label: {}".format(l))
    with open('{}_data_resized_small.pkl'.format(l), 'rb') as f:
        arr = pickle.load(f)
    for a in arr:
        X_train.append(a)
        y_train.append(l)
X_train_arr = np.swapaxes(np.asarray(X_train), 1, 3)
print(X_train_arr.shape)
y_train_num = np.array([label.index(y) for y in y_train])
print(y_train_num.shape)

np.save('y_all', y_train_num)
print('Saved y_all')
np.save('X_all', X_train_arr)
print('Saved X_all')
