# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:59:58 2021

@author: YoonSangCho
"""

#%%  RNN: https://m.blog.naver.com/PostView.nhn?blogId=chunjein&logNo=221589624838&proxyReferer=https:%2F%2Fwww.google.com%2F
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras import Input, initializers, regularizers, layers, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Layer, RepeatVector,Permute,Multiply,Lambda,Bidirectional,Embedding,GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Flatten, Input, Dense, Activation, BatchNormalization, Dropout # NN
from tensorflow.keras.layers import Conv1D, Add, MaxPooling1D, Conv2D, MaxPooling2D, AveragePooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D # CNN
from tensorflow.keras.layers import Embedding, Bidirectional, TimeDistributed, RNN, LSTM, GRU, ConvLSTM2D, Bidirectional  #CuDNNLSTM # RNN
from tensorflow.keras.layers import SimpleRNN as SimRNN

from tensorflow import keras 


def rnn_reseq(input_shape):
    # input_shape = (60, 42)
    n_nodes = 512
    model = Sequential()
    inputs = Input(shape=input_shape)
    rnn = SimRNN(n_nodes, return_sequences = True)(inputs)
    model = Model(inputs=inputs, outputs=rnn)
    # model.summary()
    return model

def lstm_reseq(input_shape):
    n_nodes = 512
    inputs = Input(shape=input_shape)
    lstm = LSTM(n_nodes, return_sequences = True
                ,kernel_regularizer=regularizers.l2(0.01)
                ,recurrent_regularizer=regularizers.l2(0.01)
                ,bias_regularizer=regularizers.l2(0.01)
                )(inputs)
    lstm = Dropout(0.5)(lstm)
    model = Model(inputs=inputs, outputs=lstm)
    # model.summary()
    return model

def gru_reseq(input_shape):
    n_nodes = 512
    inputs = Input(shape=input_shape)
    gru = GRU(n_nodes, return_sequences = True
              ,kernel_regularizer=regularizers.l2(0.01)
              ,recurrent_regularizer=regularizers.l2(0.01)
              ,bias_regularizer=regularizers.l2(0.01)
              )(inputs)
    gru = Dropout(0.5)(gru)
    # outputs = Dense(n_class, activation='softmax')(gru)
    model = Model(inputs=inputs, outputs=gru)
    return model

def bilstm_reseq(input_shape):
    n_nodes = 512
    inputs = Input(shape=input_shape)
    bilstm = Bidirectional(LSTM(n_nodes, return_sequences = True
                                ,kernel_regularizer=regularizers.l2(0.01)
                                ,recurrent_regularizer=regularizers.l2(0.01)
                                ,bias_regularizer=regularizers.l2(0.01)
                                ))(inputs)
    bilstm = Dropout(0.5)(bilstm)
    model = Model(inputs=inputs, outputs=bilstm)
    # model.summary()
    return model

def bibilstm_reseq(input_shape):
    n_nodes = 512
    inputs = Input(shape=input_shape)
    bilstm = Bidirectional(LSTM(n_nodes, return_sequences = True
                                ,kernel_regularizer=regularizers.l2(0.01)
                                ,recurrent_regularizer=regularizers.l2(0.01)
                                ,bias_regularizer=regularizers.l2(0.01)
                                ))(inputs)
    bilstm = Bidirectional(LSTM(n_nodes, return_sequences = True
                                ,kernel_regularizer=regularizers.l2(0.01)
                                ,recurrent_regularizer=regularizers.l2(0.01)
                                ,bias_regularizer=regularizers.l2(0.01)
                                ))(bilstm)
    bilstm = Dropout(0.5)(bilstm)
    model = Model(inputs=inputs, outputs=bilstm)
    # model.summary()
    return model

def rnn(input_shape):
    # input_shape = (60, 42)
    n_nodes = 256
    model = Sequential()
    inputs = Input(shape=input_shape)
    rnn = SimRNN(n_nodes)(inputs)
    model = Model(inputs=inputs, outputs=rnn)
    # model.summary()
    return model

def lstm(input_shape):
    n_nodes = 256
    inputs = Input(shape=input_shape)
    lstm = LSTM(n_nodes
                ,kernel_regularizer=regularizers.l2(0.01)
                ,recurrent_regularizer=regularizers.l2(0.01)
                ,bias_regularizer=regularizers.l2(0.01)
                )(inputs)
    lstm = Dropout(0.5)(lstm)
    model = Model(inputs=inputs, outputs=lstm)
    # model.summary()
    return model

def gru(input_shape):
    n_nodes = 256
    inputs = Input(shape=input_shape)
    gru = GRU(n_nodes
              ,kernel_regularizer=regularizers.l2(0.01)
              ,recurrent_regularizer=regularizers.l2(0.01)
              ,bias_regularizer=regularizers.l2(0.01)
              )(inputs)
    gru = Dropout(0.5)(gru)
    # outputs = Dense(n_class, activation='softmax')(gru)
    model = Model(inputs=inputs, outputs=gru)
    return model

def bilstm(input_shape):
    n_nodes = 256
    inputs = Input(shape=input_shape)
    bilstm = Bidirectional(LSTM(n_nodes
                                ,kernel_regularizer=regularizers.l2(0.01)
                                ,recurrent_regularizer=regularizers.l2(0.01)
                                ,bias_regularizer=regularizers.l2(0.01)
                                ))(inputs)
    bilstm = Dropout(0.5)(bilstm)
    model = Model(inputs=inputs, outputs=bilstm)
    # model.summary()
    return model

def bibilstm(input_shape):
    n_nodes = 256
    inputs = Input(shape=input_shape)
    bilstm = Bidirectional(LSTM(n_nodes, return_sequences = True
                                ,kernel_regularizer=regularizers.l2(0.01)
                                ,recurrent_regularizer=regularizers.l2(0.01)
                                ,bias_regularizer=regularizers.l2(0.01)
                                ))(inputs)
    bilstm = Bidirectional(LSTM(n_nodes, return_sequences = False
                                ,kernel_regularizer=regularizers.l2(0.01)
                                ,recurrent_regularizer=regularizers.l2(0.01)
                                ,bias_regularizer=regularizers.l2(0.01)
                                ))(bilstm)
    bilstm = Dropout(0.5)(bilstm)
    model = Model(inputs=inputs, outputs=bilstm)
    # model.summary()
    return model


#%%  2D-CNN
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Cropping2D, InputSpec

import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
#%%  CNN: VGGNet

def vggnet_small_1D(input_shape):
    # , y_shape, use_bias=False, print_summary=False
    _in = Input(shape=input_shape)
    
    # (64 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # (128 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    
    # (256 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(1, 2))(x)

    # # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(1, 2))(x)

    # # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(1, 2))(x)

    # GAP
    _out = GlobalAveragePooling2D()(x)
    
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name="vggnet")
    # model.summary()
    return model

def vggnet_small(input_shape):
    # , y_shape, use_bias=False, print_summary=False
    _in = Input(shape=input_shape)
    
    # (64 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # (128 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # (256 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    # x = ReLU()(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # GAP
    _out = GlobalAveragePooling2D()(x)
    
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name="vggnet")
    # model.summary()
    return model

def vggnet_19_1D(input_shape):
    # , y_shape, use_bias=False, print_summary=False
    _in = Input(shape=input_shape)
    
    # (64 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # (128 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    
    # (256 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(1, 2))(x)

    # GAP
    _out = GlobalAveragePooling2D()(x)
    
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name="vggnet")
    # model.summary()
    return model

def vggnet_19(input_shape):
    _in = Input(shape=input_shape)
    
    # (64 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # (128 CONV => RELU => BN) * 2 => POOL(MAX)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(_in) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # (256 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # (512 CONV => RELU => BN) * 4 => POOL(MAX)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x) ##########
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # GAP
    _out = GlobalAveragePooling2D()(x)
    
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name="vggnet")
    # model.summary()
    return model



#%% 2D-CNN: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BasicBlock_1D(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck_1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck_1D, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block
def make_basic_block_layer_1D(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock_1D(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock_1D(filter_num, stride=1))

    return res_block

def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block

def make_bottleneck_layer_1D(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck_1D(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck_1D(filter_num, stride=1))

    return res_block

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        output = self.avgpool(x)
        # output = self.fc(x)

        return output

class ResNetTypeI_1D(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI_1D, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(1, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer_1D(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer_1D(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer_1D(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer_1D(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        output = self.avgpool(x)
        # output = self.fc(x)

        return output


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            # kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        output = self.avgpool(x)
        # output = self.fc(x)

        return output

class ResNetTypeII_1D(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII_1D, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(1, 7),
                                            # kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer_1D(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer_1D(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer_1D(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer_1D(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        output = self.avgpool(x)
        # output = self.fc(x)

        return output


def resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])
def resnet_34():
    return ResNetTypeI(layer_params=[3, 4, 6, 3])
def resnet_50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])
def resnet_101():
    return ResNetTypeII(layer_params=[3, 4, 23, 3])
def resnet_152():
    return ResNetTypeII(layer_params=[3, 8, 36, 3])
def resnet_18_1D():
    return ResNetTypeI_1D(layer_params=[2, 2, 2, 2])
def resnet_34_1D():
    return ResNetTypeI_1D(layer_params=[3, 4, 6, 3])
def resnet_50_1D():
    return ResNetTypeII_1D(layer_params=[3, 4, 6, 3])
def resnet_101_1D():
    return ResNetTypeII_1D(layer_params=[3, 4, 23, 3])
def resnet_152_1D():
    return ResNetTypeII_1D(layer_params=[3, 8, 36, 3])


#%% 2D-CNN: DenseNet: https://github.com/flyyufelix/DenseNet-Keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, ZeroPadding2D
from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Activation
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, InputSpec
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 
    concat_axis = -1

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    # x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(filters=int(nb_filter * compression), use_bias=False, kernel_size=(1, 1), name=conv_name_base)(x) #, padding='same', 
    # x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    # x, stage, nb_layers = nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = Concatenate(axis=-1)([concat_feat, x])

        if grow_nb_filters:
            nb_filter += growth_rate
    return concat_feat, nb_filter

def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    concat_axis=-1
    inter_channel = nb_filter * 4  
    
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x) # x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(filters=inter_channel, use_bias=False, kernel_size=(1, 1), name=conv_name_base+'_x1')(x)  #padding='same',
    # x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)


    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    # x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(filters=inter_channel, use_bias=False, kernel_size=(3, 3), name=conv_name_base+'_x2')(x)
    # x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def densenet_121(input_shape):
    nb_dense_block=4
    growth_rate=32
    nb_filter=64
    reduction=0.0
    dropout_rate=0.0
    weight_decay=1e-4
    # classes=1000
    weights_path=None
    eps = 1.1e-5
    compression = 1.0 - reduction

    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121
    concat_axis=-1
    
    img_input = Input(shape=input_shape, name='data')
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(filters=nb_filter, use_bias=False, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    # x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    # x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        # block_idx=0
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    # x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    # x = Dense(classes, name='fc6')(x)
    # x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
      model.load_weights(weights_path)

    return model

def densenet_161(input_shape):
    nb_dense_block=4
    growth_rate=48
    nb_filter=96
    reduction=0.0
    dropout_rate=0.0
    weight_decay=1e-4
    concat_axis = -1
    eps = 1.1e-5
    compression = 1.0 - reduction
    nb_filter = 96
    nb_layers = [6,12,36,24] # For DenseNet-161

    img_input = Input(shape=input_shape, name='data')
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(filters=nb_filter, use_bias=False, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    # x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    # x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    # x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    # x = Dense(classes, name='fc6')(x)
    # x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    # if weights_path is not None:
    #   model.load_weights(weights_path)

    return model

def densenet_169(input_shape):
    nb_dense_block=4
    growth_rate=32
    nb_filter=64
    reduction=0.0
    dropout_rate=0.0
    weight_decay=1e-4
    classes=1000
    weights_path=None
    concat_axis=-1

    eps = 1.1e-5
    # compute compression factor
    compression = 1.0 - reduction

    img_input = Input(shape=input_shape, name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,32,32] # For DenseNet-169

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(filters=nb_filter, use_bias=False, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    # x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    # x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    # x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    # x = Dense(classes, name='fc6')(x)
    # x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    # if weights_path is not None:
    #   model.load_weights(weights_path)

    return model

def densenet_201(input_shape):
    nb_dense_block=4
    growth_rate=32
    nb_filter=64
    reduction=0.0
    dropout_rate=0.0
    weight_decay=1e-4
    eps = 1.1e-5
    compression = 1.0 - reduction
    nb_filter = 64
    nb_layers = [6,12,48,32] # For DenseNet-169
    concat_axis = -1

    img_input = Input(shape=input_shape, name='data')
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(filters=nb_filter, use_bias=False, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    # x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    # x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    # x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    # x = Dense(classes, name='fc6')(x)
    # x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    # if weights_path is not None:
    #   model.load_weights(weights_path)

    return model


#%% 2D-CNN: DenseNet (CUSTOM)

def ConvBlock(x, filters):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters * 4, use_bias=False, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, use_bias=False, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    return x
def TransitionBlock(x, filters, compression=1):
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=int(filters * compression), use_bias=False, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x

def DenseBlock(x, num_layers, growth_rate):
    concat_feature = x
    for l in range(num_layers):
        x = ConvBlock(concat_feature, growth_rate)
        concat_feature = Concatenate(axis=-1)([concat_feature, x])
    return concat_feature


# def densenet(input_shape):
#     inputs = Input(shape=input_shape)
#     num_layers = 3
#     growth_rate=4
    
#     x = Conv2D(filters=24, kernel_size=(7, 7), strides=(1, 1), padding='same', use_bias=False)(inputs)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     # x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    
#     x = DenseBlock(x, num_layers, growth_rate)
#     x = TransitionBlock(x, x.shape[-1], 0.5)
#     x = DenseBlock(x, num_layers, growth_rate)
#     x = TransitionBlock(x, x.shape[-1], 0.5)
#     x = DenseBlock(x, num_layers, growth_rate)
#     outputs = GlobalAveragePooling2D()(x)
#     model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='DenseNet')
#     # model .summary()
#     return model
def densenet(input_shape, use_bias=False, print_summary=False):
    _in = Input(shape=input_shape)
    x = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1),padding='same', use_bias=False)(_in)
    x = DenseBlock(x, 3, 3)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 3, 3)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 3, 3)
    _out = GlobalAveragePooling2D()(x)
    # _out = Dense(units=m, use_bias=False, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name='DenseNet')
    if print_summary:
        model.summary()
    return model