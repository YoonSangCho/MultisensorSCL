# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:52:47 2020

@author: yscho
"""
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score, accuracy_score, f1_score
#%% functions: evaluation
def get_class_prob(dataset, model):
    pred_scores = []
    for x in dataset:
        class_prob = model(x, training=False)
        pred_scores.append(class_prob.numpy())

    pred_scores = np.concatenate(pred_scores, axis=0)
    return pred_scores

def make_prediction(_scores, _T):
    _labels = _scores - _T > 0
    return _labels

def evaluation(dat_true, dat_pred):
        
    dat_sum = dat_pred+dat_true
    dat_fn = dat_true-dat_pred
    dat_fp = dat_pred-dat_true
    
    n_alarm_list = []
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    n_class = dat_true.shape[1]

    ##### get performances for each class
    for c in range(n_class):

        # c = 7
        TP = sum(dat_sum.iloc[:, c]==2)
        FN = sum(dat_fn.iloc[:, c]==1)
        FP = sum(dat_fp.iloc[:, c]==1)
        TN = sum(dat_sum.iloc[:, c]==0)
        
        n_alarm = TP+FN
        n_alarm_list.append(n_alarm)
        n = TP+FN+FP+TN
        T = TP+TN
        
        if (TP+FN) == 0:
            recall_list.append('-')
        else:
            r = TP/(TP+FN)
            recall = np.round(r, 2)
            recall_list.append(recall)

        if (TP+FP) == 0:
            precision_list.append('-')
        else:
            p = TP/(TP+FP)
            precision = np.round(p, 2)
            precision_list.append(precision)
            
        if (TP+FN) == 0 or (TP+FP) == 0:
            f1_list.append('-')
        else:
            f1 = (2*recall*precision)/(recall+precision)
            f1_score = np.round(f1, 2)
            f1_list.append(f1_score)
            
        if TP == 0:
            accuracy_list.append('-')
        else:
            acc = np.round((T/n), 4)
            accuracy_list.append(acc)
        
    colnames = list(dat_sum.columns)
    idxnames = ['count', 'accuracy', 'recall', 'precision', 'f1 score']
    performance = pd.DataFrame([n_alarm_list, accuracy_list, recall_list, precision_list, f1_list],
                               columns = colnames,
                               index = idxnames)
    
    ##### get average score exept non-value '-'
    list_acc = list(performance.iloc[1])
    list_recall = list(performance.iloc[2])
    list_precision = list(performance.iloc[3])
    list_f1_score = list(performance.iloc[4])
    list_performance = [list_acc, list_recall, list_precision, list_f1_score]
    for i in range(len(list_performance)):
        if '-' in list_performance[i]:
            n_non = len(np.where(np.array(list_performance[i])=='-')[0])
            for non in range(n_non):
                list_performance[i].remove('-')
        else:
            pass
    sum_count = int(sum(performance.iloc[0]))
    avg_acc = np.mean(list_performance[0])
    avg_acc = np.round(avg_acc, 3)
    avg_recall = np.mean(list_performance[1])
    avg_recall = np.round(avg_recall, 3)
    avg_precision = np.mean(list_performance[2])
    avg_precision = np.round(avg_precision, 3)
    avg_f1_score = np.mean(list_performance[3])
    avg_f1_score = np.round(avg_f1_score, 3)
    
    performance['performance(n_window={})'.format(n)] = [sum_count,
                                                         avg_acc,
                                                         avg_recall,
                                                         avg_precision,
                                                         avg_f1_score]
    
    return performance.T

def performance_evaluation_bench(classifier, threshold, x_train, x_valid, x_test, y_train, y_valid, y_test, class_names):
    pred_y_tr = classifier.predict([x_train])
    pred_y_vl = classifier.predict([x_valid])
    pred_y_ts = classifier.predict([x_test])

    train_pred_labels = make_prediction(pred_y_tr, threshold) #.astype(int)
    valid_pred_labels = make_prediction(pred_y_vl, threshold) #.astype(int)
    test_pred_labels = make_prediction(pred_y_ts, threshold) #.astype(int)

    # f1_macro_tr = f1_score(y_train, train_pred_labels, average='macro')
    # f1_macro_vl = f1_score(y_valid, valid_pred_labels, average='macro')
    # f1_macro_ts = f1_score(y_test, test_pred_labels, average='macro')
    
    f1_micro_tr = f1_score(y_train, train_pred_labels, average='micro').round(4)
    f1_micro_vl = f1_score(y_valid, valid_pred_labels, average='micro').round(4)
    f1_micro_ts = f1_score(y_test, test_pred_labels, average='micro').round(4)
    
    acc_tr = accuracy_score(y_train, train_pred_labels).round(4)
    acc_vl = accuracy_score(y_valid, valid_pred_labels).round(4)
    acc_ts = accuracy_score(y_test, test_pred_labels).round(4)
    
    jaccard_tr = jaccard_score(y_train, train_pred_labels, average='samples').round(4)
    jaccard_vl = jaccard_score(y_valid, valid_pred_labels, average='samples').round(4)
    jaccard_ts = jaccard_score(y_test, test_pred_labels, average='samples').round(4)

    print("acc_tr:", acc_tr)
    print("acc_vl:", acc_vl)
    print("acc_ts:", acc_ts)

    print("f1_micro_tr:", f1_micro_tr)
    print("f1_micro_vl:", f1_micro_vl)
    print("f1_micro_ts:", f1_micro_ts)

    print("jaccard_tr:", jaccard_tr)
    print("jaccard_vl:", jaccard_vl)
    print("jaccard_ts:", jaccard_ts)

    # test_y_df = pd.DataFrame(y_test, columns = class_names)
    # test_pred_labels_df = pd.DataFrame(test_pred_labels, columns = class_names)
    
    # results = evaluation(test_y_df, test_pred_labels_df)
    return acc_ts, f1_micro_ts, jaccard_ts # , results

def performance_evaluation(classifier, threshold, x_train, x_valid, x_test, y_train, y_valid, y_test, class_names):
    # import tensorflow as tf
    # import tensorflow_addons as tfa
    # from tensorflow.compat.v1 import ConfigProto, InteractiveSession
    # from tensorflow.python.client import device_lib
    # config = ConfigProto(device_count = {'GPU': 1})
    # session = InteractiveSession(config=config)

    pred_y_tr = classifier.predict([x_train])
    pred_y_vl = classifier.predict([x_valid])
    pred_y_ts = classifier.predict([x_test])

    train_pred_labels = make_prediction(pred_y_tr, threshold) #.astype(int)
    valid_pred_labels = make_prediction(pred_y_vl, threshold) #.astype(int)
    test_pred_labels = make_prediction(pred_y_ts, threshold) #.astype(int)

    training_score = jaccard_score(y_train, train_pred_labels, average='samples').round(4)
    validing_score = jaccard_score(y_valid, valid_pred_labels, average='samples').round(4)
    testing_score = jaccard_score(y_test, test_pred_labels, average='samples').round(4)

    print("training_score:", training_score)
    print("validing_score:", validing_score)
    print("testing_score:", testing_score)

    # test_y_df = pd.DataFrame(y_test, columns = class_names)
    # test_pred_labels_df = pd.DataFrame(test_pred_labels, columns = class_names)
    
    # results = evaluation(test_y_df, test_pred_labels_df)
    return training_score, validing_score, testing_score #, results