# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:04:13 2021

@author: YoonSangCho

cd Multisensor-Classification-SupervisedContrastiveLearning/code/
python main_benchmark.py
"""


#%% experiments design
import os
import logging
import datetime

yy = datetime.datetime.now().year
mm = datetime.datetime.now().month
dd = datetime.datetime.now().day
exp = "EXPERIMENTS"
MEMORY_GB = 20
server = True
# server = False
GPU_NUM = 1

# datasets
data_list = ['Heartbeat', ##### O big: GPU 1GB impossible
             'NATOPS', ##### O good 1GB OK
             'RacketSports', ##### O good 1GB OK
             'SelfRegulationSCP1', ##### O good
             'SelfRegulationSCP2', ##### O 1GB OK
             'UWaveGestureLibrary' ##### O 1GB OK
             'PEMS-SF' ##### O big: GPU 1GB impossible
             ]
# encoder list
encoder_name_list = ['resnet_18_1D', 'resnet_34_1D', 'vggnet_small_1D', 'vggnet_19_1D']
# augmentation
augmentation_list = ['None', 'horizontal', 'scaling', 'jitter', 'scaling&horizontal', 'jitter&horizontal', 'jitter&scaling'] # 'jitter&scaling&horizontal'
# learning tasks
task_list = ['Supervised', 'SupContrastive']
# hyperparameters
learning_rate = 0.001
batch_size = 256
hidden_units = 512
projection_units = 128
num_epochs = 100
dropout_rate = 0.5
temperature = 0.05
threshold = 0.5
earlystop = True
patience = 10
kfold = 5
random_state = 2017010500

#%% modules
if server:
    path = '/workspace/workspace/yscho187/Multisensor-Classification-SupervisedContrastiveLearning/'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU_NUM}" 
else:
    path = 'C:/Users/korea/Dropbox/RESEARCH/Multisensor-Classification-SupervisedContrastiveLearning/'

# path_data = path+'data/CaseStudy/'
path_data = path+'data/Benchmark/'
path_result = path+'results/'
path_model = path_result+f'model/'
path_performance = path_result+f'performance/'
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.chdir(path+'code/')
os.makedirs(path_result, exist_ok=True)
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_performance, exist_ok=True)

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import tensorflow.experimental.numpy as tnp
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus: 
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*MEMORY_GB)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
from utils.data import data_loader, normalization
from utils.augmentation import augmentation_layer
from utils.models import trainer_enc_bench, trainer_enc, trainer_clf, encoder_loader, create_encoder, create_encoder_batch, create_classifier_Multilabel, create_classifier_Singlelabel, add_projection_head, SupervisedContrastiveLoss_Multilabel, SupervisedContrastiveLoss_Singlelabel
from utils.evaluation import get_class_prob, make_prediction, evaluation, performance_evaluation, performance_evaluation_bench

#%% Experiment 
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
tf.random.set_seed(random_state)
np.random.seed(random_state)
skf = MultilabelStratifiedKFold(n_splits=kfold, shuffle=True, random_state = random_state)

for encoder_name in encoder_name_list: 
    colnames=['data_name', 'task', 'encoder', 'aug_name', 'fold', 'acc', 'f1', 'jc']
    results_df = pd.DataFrame(columns=colnames)
    for data_name in data_list:
        # datasets
        folder_name = f'BENCHMARKING_{yy}{mm}{dd}-{exp}-{encoder_name}-{data_name}/'
        os.makedirs(path_model+folder_name, exist_ok=True)
        os.makedirs(path_performance+folder_name, exist_ok=True)
        name_x = path_data+data_name+'_x'
        name_y = path_data+data_name+'_y'
        with open(name_x, 'rb') as f:
            x = pickle.load(f)
            x = np.expand_dims(x, -1)
        with open(name_y, 'rb') as f:
            y = pickle.load(f)
        print('class', len(np.unique(y)), np.unique(y))
        print('x.shape, y.shape', x.shape, y.shape)
        class_names = list(np.unique(y))
        
        # Modeling
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        y_ohe = to_categorical(y_enc)
        
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_ohe)):
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_ohe[train_index], y_ohe[test_index]
            x_train, x_test = normalization(x_train, x_test)
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=random_state, shuffle=True, stratify=y_train)
            
            x_train = tnp.swapaxes(x_train, 1, 3) # if encoder_name in ['vggnet_small_1D', 'vggnet_19_1D', 'resnet_18_1D', 'resnet_34_1D', 'resnet_50_1D', 'resnet_101_1D', 'resnet_152_1D']:
            x_valid = tnp.swapaxes(x_valid, 1, 3) # if encoder_name in ['vggnet_small_1D', 'vggnet_19_1D', 'resnet_18_1D', 'resnet_34_1D', 'resnet_50_1D', 'resnet_101_1D', 'resnet_152_1D']:
            x_test = tnp.swapaxes(x_test, 1, 3) # if encoder_name in ['vggnet_small_1D', 'vggnet_19_1D', 'resnet_18_1D', 'resnet_34_1D', 'resnet_50_1D', 'resnet_101_1D', 'resnet_152_1D']:

            # x_train = tnp.moveaxis(x_train, 1, 2) #if encoder_name=='transformer':
            # x_valid = tnp.moveaxis(x_valid, 1, 2) #if encoder_name=='transformer':
            # x_test = tnp.moveaxis(x_test, 1, 2) #if encoder_name=='transformer':
            
            # x_train = tf.squeeze(tnp.moveaxis(x_train, 1, 2), -1) # if encoder_name in ['rnn', 'lstm', 'gru', 'bilstm']:
            # x_valid = tf.squeeze(tnp.moveaxis(x_valid, 1, 2), -1) # if encoder_name in ['rnn', 'lstm', 'gru', 'bilstm']:
            # x_test = tf.squeeze(tnp.moveaxis(x_test, 1, 2), -1) # if encoder_name in ['rnn', 'lstm', 'gru', 'bilstm']:

            y_train, y_valid, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_valid), tf.convert_to_tensor(y_test)

            input_shape = x_train.shape[1:]
            n_y = len(class_names)
            print('x_train.shape', x_train.shape)
            print('x_valid.shape', x_valid.shape)
            print('x_test.shape', x_test.shape)
            batch_size = int(len(x_train)/20)
            if batch_size <= 2:
                batch_size = 3

            for task in task_list: 
                if task == 'Supervised':
                    for aug_name in augmentation_list:
                        # aug_name = 'None'
                        # Encoder and Classifier Training
                        encoder = create_encoder_batch(input_shape, encoder_name)
                        classifier_encoder_trainable = True
                        classifier = create_classifier_Singlelabel(encoder, input_shape, n_y, dropout_rate, learning_rate, hidden_units, trainable=classifier_encoder_trainable)
                        optimizer_clf = keras.optimizers.Adam(learning_rate)
                        loss_clf = keras.losses.CategoricalCrossentropy()
                        ####################################################################################################
                        path_ckpt_clf = path_model+folder_name+f'{task}_{encoder_name}_{aug_name}_{data_name}_{random_state}.h5'
                        ####################################################################################################
                        model_clf = trainer_clf(classifier, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss_clf, optimizer_clf, earlystop, patience, path_ckpt_clf)
                         
                        ################################################## performance_evaluation
                        acc_ts, f1_micro_ts, jaccard_ts = performance_evaluation_bench(model_clf, threshold, x_train, x_valid, x_test, y_train, y_valid, y_test, class_names) #, results 
                        os.makedirs(path_performance+folder_name, exist_ok=True)
                        # result_name_perf = path_performance+folder_name+f'{task}_{encoder_name}_{aug_name}_{plant}_{line}_{random_state}_tr_{training_score}_ts_{testing_score}.csv'
                        # results.to_csv(result_name_perf)
                        result_tmp = pd.DataFrame([[data_name, task, encoder_name, aug_name, fold+1, acc_ts, f1_micro_ts, jaccard_ts]], columns=colnames)
                        results_df = pd.concat((results_df, result_tmp))
                        results_df.to_csv(path_performance+folder_name+f'{encoder_name}_results.csv')

                elif task == 'SupContrastive':
                    for aug_name in augmentation_list:
                        task = 'SupContrastive'
                        ################################################## Encoder Training
                        # encoder 
                        encoder = create_encoder_batch(input_shape, encoder_name)
                        encoder_with_projection_head = add_projection_head(input_shape, projection_units, encoder)
                        optimizer_enc = keras.optimizers.Adam(learning_rate)
                        loss_enc = SupervisedContrastiveLoss_Singlelabel(temperature)
                        encoder_with_projection_head.compile(optimizer=optimizer_enc,loss=loss_enc)
                        # check point
                        path_ckpt_enc = path_model + folder_name+f'{task}_{encoder_name}_enc_{aug_name}_{data_name}_{random_state}.h5'
                        # training
                        model_enc = trainer_enc_bench(encoder_with_projection_head, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss_enc, optimizer_enc, earlystop, patience, path_ckpt_enc)
                        ##################################################
                        
                        ################################################## Classifier Training
                        # classifier 
                        classifier_encoder_trainable = False
                        classifier = create_classifier_Singlelabel(encoder, input_shape, n_y, dropout_rate, learning_rate, hidden_units, trainable=classifier_encoder_trainable)
                        optimizer_clf = keras.optimizers.Adam(learning_rate)
                        loss_clf = keras.losses.CategoricalCrossentropy()
                        # check point
                        path_ckpt_clf = path_model + folder_name+f'{task}_{encoder_name}_clf_{aug_name}_{data_name}_{random_state}.h5'
                        # training
                        model_clf = trainer_clf(classifier, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss_clf, optimizer_clf, earlystop, patience, path_ckpt_clf)

                        ################################################## performance_evaluation
                        acc_ts, f1_micro_ts, jaccard_ts = performance_evaluation_bench(model_clf, threshold, x_train, x_valid, x_test, y_train, y_valid, y_test, class_names) #, results 
                        os.makedirs(path_performance+folder_name, exist_ok=True)
                        # result_name_perf = path_performance+folder_name+f'{task}_{encoder_name}_{aug_name}_{plant}_{line}_{random_state}_tr_{training_score}_ts_{testing_score}.csv'
                        # results.to_csv(result_name_perf)
                        result_tmp = pd.DataFrame([[data_name, task, encoder_name, aug_name, fold+1, acc_ts, f1_micro_ts, jaccard_ts]], columns=colnames)
                        results_df = pd.concat((results_df, result_tmp))
                        results_df.to_csv(path_performance+folder_name+f'{encoder_name}_results.csv')

                        ################################################## classifier training (classifier_encoder_trainable = True)
                        # classifier
                        task = 'SupContrastive(trainable)'
                        classifier_encoder_trainable = True
                        classifier_tnb = create_classifier_Singlelabel(encoder, input_shape, n_y, dropout_rate, learning_rate, hidden_units, trainable=classifier_encoder_trainable)
                        optimizer_clf_tnb = keras.optimizers.Adam(learning_rate)
                        loss_clf_tnb = keras.losses.BinaryCrossentropy()
                        # checkpoint
                        path_ckpt_clf = path_model + folder_name+f'{task}_{encoder_name}_clf_{aug_name}_{data_name}_{random_state}.h5'
                        os.makedirs(path_model+folder_name, exist_ok=True)
                        # training
                        model_clf_tnb = trainer_clf(classifier_tnb, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss_clf_tnb, optimizer_clf_tnb, earlystop, patience, path_ckpt_clf)

                        ################################################## performance_evaluation
                        acc_ts, f1_micro_ts, jaccard_ts = performance_evaluation_bench(model_clf_tnb, threshold, x_train, x_valid, x_test, y_train, y_valid, y_test, class_names) #, results 
                        os.makedirs(path_performance+folder_name, exist_ok=True)
                        # result_name_perf = path_performance+folder_name+f'{task}_{encoder_name}_{aug_name}_{plant}_{line}_{random_state}_tr_{training_score}_ts_{testing_score}.csv'
                        # results.to_csv(result_name_perf)
                        result_tmp = pd.DataFrame([[data_name, task, encoder_name, aug_name, fold+1, acc_ts, f1_micro_ts, jaccard_ts]], columns=colnames)
                        results_df = pd.concat((results_df, result_tmp))
                        results_df.to_csv(path_performance+folder_name+f'{encoder_name}_results.csv')

                        