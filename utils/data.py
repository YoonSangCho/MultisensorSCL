# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 02:51:38 2021

@author: YoonSangCho
"""

#%% functions: data

def normalization(x_train, x_test):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    x_re_train = np.moveaxis(x_train, 1, 2).reshape(x_train.shape[0]*x_train.shape[2], x_train.shape[1])
    x_re_test = np.moveaxis(x_test, 1, 2).reshape(x_test.shape[0]*x_test.shape[2], x_test.shape[1])
    scaler = StandardScaler()
    scaler.fit(x_re_train)
    x_re_train_norm = scaler.transform(x_re_train)
    x_re_test_norm = scaler.transform(x_re_test)
    x_train_norm = np.reshape(x_re_train_norm, (x_train.shape[0], x_train.shape[2], x_train.shape[1]))
    x_test_norm = np.reshape(x_re_test_norm, (x_test.shape[0], x_test.shape[2], x_test.shape[1]))
    x_train_norm =  np.expand_dims(np.moveaxis(x_train_norm, 1, 2), -1)
    x_test_norm = np.expand_dims(np.moveaxis(x_test_norm, 1, 2), -1)
    return x_train_norm, x_test_norm

def data_loader(path_data, plant, line):
    import pickle
    import numpy as np

    with open(path_data+f'{plant}_{line}_x_sample.pkl', 'rb') as f:
        x = pickle.load(f)
    with open(path_data+f'{plant}_{line}_y_sample.pkl', 'rb') as f:
        y = pickle.load(f)
    if plant=='세타':
        with open(path_data+f'{plant}_{line}_class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
    elif plant=='신R':
        with open(path_data+f'{plant}_{line}_main_class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
    f.close()
    return x, y, class_names

    
def data_loader_train_valid_test(path_data, plant, line):
    import pickle
    import numpy as np

    if plant=='세타':
        with open(path_data+f'{plant}_{line}_X_train.pkl', 'rb') as f:
            x_train = pickle.load(f)
            x_train = np.expand_dims(x_train, axis=3)
            x_train = np.moveaxis(x_train, 1, 2)
        with open(path_data+f'{plant}_{line}_y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(path_data+f'{plant}_{line}_X_valid.pkl', 'rb') as f:
            x_valid = pickle.load(f)
            x_valid = np.expand_dims(x_valid, axis=3)
            x_valid = np.moveaxis(x_valid, 1, 2)
        with open(path_data+f'{plant}_{line}_y_valid.pkl', 'rb') as f:
            y_valid = pickle.load(f)
        with open(path_data+f'{plant}_{line}_X_test.pkl', 'rb') as f:
            x_test = pickle.load(f)
            x_test = np.expand_dims(x_test, axis=3)
            x_test = np.moveaxis(x_test, 1, 2)
        with open(path_data+f'{plant}_{line}_y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
        with open(path_data+f'{plant}_{line}_class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
    elif plant=='신R':
        with open(path_data+f'{plant}_{line}_main_X_train.pkl', 'rb') as f:
            x_train = pickle.load(f)
            x_train = np.expand_dims(x_train, axis=3)
            x_train = np.moveaxis(x_train, 1, 2)
        with open(path_data+f'{plant}_{line}_main_y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(path_data+f'{plant}_{line}_main_X_valid.pkl', 'rb') as f:
            x_valid = pickle.load(f)
            x_valid = np.expand_dims(x_valid, axis=3)
            x_valid = np.moveaxis(x_valid, 1, 2)
        with open(path_data+f'{plant}_{line}_main_y_valid.pkl', 'rb') as f:
            y_valid = pickle.load(f)
        with open(path_data+f'{plant}_{line}_main_X_test.pkl', 'rb') as f:
            x_test = pickle.load(f)
            x_test = np.expand_dims(x_test, axis=3)
            x_test = np.moveaxis(x_test, 1, 2)
        with open(path_data+f'{plant}_{line}_main_y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
        with open(path_data+f'{plant}_{line}_main_class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
    return x_train, x_valid, x_test, y_train, y_valid, y_test, class_names
