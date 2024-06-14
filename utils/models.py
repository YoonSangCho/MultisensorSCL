# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 15:42:09 2021

@author: YoonSangCho
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers

#%% functions: models
import numpy as np
import os
path = '/workspace/workspace/yscho187/Multisensor-Classification-SupervisedContrastiveLearning/'
#path = 'C:/Users/korea/Dropbox/RESEARCH/Multisensor-Classification-SupervisedContrastiveLearning/'
os.chdir(path+'code/')
from utils.augmentation import augmentation_layer
# from utils.encoders_transformer import transformer
# from utils.encoders_segmentation import segnet, deconvnet, fcn_vgg, fcn_densenet, unet
from utils.encoders import vggnet_small_1D, vggnet_small, vggnet_19_1D, vggnet_19
from utils.encoders import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from utils.encoders import resnet_18_1D, resnet_34_1D, resnet_50_1D, resnet_101_1D, resnet_152_1D
from utils.encoders import densenet, densenet_121, densenet_161, densenet_169, densenet_201
from utils.encoders import rnn, lstm, gru, bilstm
from utils.encoders import rnn_reseq, lstm_reseq, gru_reseq, bilstm_reseq

# 'vggnet_small_1D', 'vggnet_small', 'vggnet_19_1D', 'vggnet_19' 
# 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152'
# 'resnet_18_1D', 'resnet_34_1D', 'resnet_50_1D', 'resnet_101_1D', 'resnet_152_1D'
# 'rnn', 'lstm', 'gru', 'bilstm' 

#%% functions: models

def encoder_loader(input_shape, encoder_name):
    # input_shape = (42, 60, 1)
    # keras.applications.__dir__()
    
    ##### CNN Feature Extraction Network: VGGNET
    # 'vggnet_small_1D', 'vggnet_small', 'vggnet_19_1D', 'vggnet_19' 
    if encoder_name == "vggnet_small_1D":
        encoder_model = vggnet_small_1D(input_shape)
    elif encoder_name == "vggnet_small":
        encoder_model = vggnet_small(input_shape)
    elif encoder_name == "vggnet_19_1D":
        encoder_model = vggnet_19_1D(input_shape)
    elif encoder_name == "vggnet_19":
        encoder_model = vggnet_19(input_shape)

    ##### CNN Feature Extraction Network: Resnet
    # 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152'
    # 'resnet_18_1D', 'resnet_34_1D', 'resnet_50_1D', 'resnet_101_1D', 'resnet_152_1D'
    elif encoder_name == "resnet_18":
        encoder_model = resnet_18()
    elif encoder_name == "resnet_34":
        encoder_model = resnet_34()
    elif encoder_name == "resnet_50":
        encoder_model = resnet_50()
    elif encoder_name == "resnet_101":
        encoder_model = resnet_101()
    elif encoder_name == "resnet_152":
        encoder_model = resnet_152()

    elif encoder_name == "resnet_18_1D":
        encoder_model = resnet_18_1D()
    elif encoder_name == "resnet_34_1D":
        encoder_model = resnet_34_1D()
    elif encoder_name == "resnet_50_1D":
        encoder_model = resnet_50_1D()
    elif encoder_name == "resnet_101_1D":
        encoder_model = resnet_101_1D()
    elif encoder_name == "resnet_152_1D":
        encoder_model = resnet_152_1D()
    
    # "ResNet50", "ResNet101", "ResNet152", "ResNet101V2", "ResNet152V2", "ResNet50V2"
    elif encoder_name == "ResNet50":
        encoder_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == "ResNet101":
        encoder_model = tf.keras.applications.resnet.ResNet101(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == "ResNet152":
        encoder_model = tf.keras.applications.resnet.ResNet152(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == "ResNet50V2":
        encoder_model = tf.keras.applications.densenet.DenseNet201(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == "ResNet101V2":
        encoder_model = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == "ResNet152V2":
        encoder_model = tf.keras.applications.densenet.DenseNet169(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    ##### CNN Feature Extraction Network: Densenet
    elif encoder_name == "densenet":
        encoder_model = densenet_121(input_shape)
    
    # 'densenet_121', 'densenet_161', 'densenet_169', 'densenet_201'
    elif encoder_name == "densenet_121":
        encoder_model = densenet_121(input_shape)
    elif encoder_name == "densenet_161":
        encoder_model = densenet_161(input_shape)
    elif encoder_name == "densenet_169":
        encoder_model = densenet_169(input_shape)
    elif encoder_name == "DenseNet121":
        encoder_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == "DenseNet169":
        encoder_model = tf.keras.applications.densenet.DenseNet169(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == "DenseNet201":
        encoder_model = tf.keras.applications.densenet.DenseNet201(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')

    ##### CNN Feature Extraction Network: VGGNET
    # 'VGG16', 'VGG19'
    elif encoder_name == 'VGG16':
        encoder_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    elif encoder_name == 'VGG19':
        encoder_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    ##### RNN Feature Extraction Network
    # 'rnn', 'lstm', 'gru', 'bilstm' 
    elif encoder_name == "rnn":
        encoder_model = rnn(input_shape)
    elif encoder_name == "lstm":
        encoder_model = lstm(input_shape)
    elif encoder_name == "gru":
        encoder_model = gru(input_shape)
    elif encoder_name == "bilstm":
        encoder_model = bilstm(input_shape)
    elif encoder_name == "rnn_reseq":
        encoder_model = rnn_reseq(input_shape)
    elif encoder_name == "lstm_reseq":
        encoder_model = lstm_reseq(input_shape)
    elif encoder_name == "gru_reseq":
        encoder_model = gru_reseq(input_shape)
    elif encoder_name == "bilstm_reseq":
        encoder_model = bilstm_reseq(input_shape)

    return encoder_model


#%%

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path= './checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간 Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력 Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화 Default: 0
            path (str): checkpoint저장 경로 Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # validation loss가 감소하면 모델을 저장한다.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        model.save_weights(self.path)
        # model.save(self.path)
        # tf.keras.models.save_model(model, self.path, overwrite=True, include_optimizer=True)

        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def trainer_clf(model, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss, optimizer, earlystop, patience, path_ckpt):
    # model = model_clf
    train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
    valid_ds=tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).batch(batch_size)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    if earlystop:
        early_stopping = EarlyStopping(patience = patience, verbose = True, path=path_ckpt)

    for epoch in range(1, num_epochs+1):
        print("\nStart of epoch %d" % (epoch,))
        # Training & Validation
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            if aug_name == 'None':
                x_batch_train_aug = x_batch_train
            else:
                x_batch_train_aug = augmentation_layer(aug_name, encoder_name)(x_batch_train)
                x_batch_train_aug = np.concatenate((x_batch_train, x_batch_train_aug))
                y_batch_train = np.concatenate((y_batch_train, y_batch_train))

            if len(x_batch_train_aug) <=2:
                pass
            else:
                with tf.GradientTape() as tape:
                    logits_train = model(x_batch_train_aug, training=True)
                    loss_tr = loss(y_batch_train, logits_train)
                grads = tape.gradient(loss_tr, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                print(f'step {step} loss training: {loss_tr.numpy()}')
                train_losses.append(loss_tr.numpy())

        for step, (x_batch_valid, y_batch_valid) in enumerate(valid_ds):
            if aug_name == 'None':
                x_batch_valid_aug = x_batch_valid
            else:
                x_batch_valid_aug = augmentation_layer(aug_name, encoder_name)(x_batch_valid)
                x_batch_valid_aug = np.concatenate((x_batch_valid, x_batch_valid_aug))
                y_batch_valid = np.concatenate((y_batch_valid, y_batch_valid))
            # x_batch_valid_aug = x_batch_valid
            if len(x_batch_valid_aug) <=2:
                pass
            else:
                with tf.GradientTape() as tape:
                    logits_valid = model(x_batch_valid_aug, training=False)
                    loss_vl = loss(y_batch_valid, logits_valid)
                print(f'step {step} loss validation: {loss_vl.numpy()}')
                valid_losses.append(loss_vl.numpy())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        if earlystop:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            model.save(path_ckpt)
    # model = tf.keras.models.load_model(path_ckpt)
    model.load_weights(path_ckpt)

    return model

def trainer_enc(model, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss, optimizer, earlystop, patience, path_ckpt):
    # model, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss, optimizer, earlystop, patience, path_ckpt = encoder_with_projection_head, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss_enc, optimizer_enc, earlystop, patience, path_ckpt
    train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
    valid_ds=tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).batch(batch_size)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    if earlystop:
        early_stopping = EarlyStopping(patience = patience, verbose = True, path=path_ckpt)
    # num_epochs=2
    for epoch in range(1, num_epochs+1):
        print("\nStart of epoch %d" % (epoch,))
        # Training & Validation
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            if aug_name == 'None':
                x_batch_train_aug = x_batch_train
            else:
                x_batch_train_aug = augmentation_layer(aug_name, encoder_name)(x_batch_train)
                x_batch_train_aug = np.concatenate((x_batch_train, x_batch_train_aug))
                y_batch_train = np.concatenate((y_batch_train, y_batch_train))

            if len(x_batch_train_aug) <=2:
                pass
            else:
                with tf.GradientTape() as tape:
                    logits_train = model(x_batch_train_aug, training=True)
                    loss_tr = loss(y_batch_train, logits_train)
                grads = tape.gradient(loss_tr, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                print(f'step {step} loss training: {loss_tr.numpy()}')
                train_losses.append(loss_tr.numpy())


        for step, (x_batch_valid, y_batch_valid) in enumerate(valid_ds):
            if aug_name == 'None':
                x_batch_valid_aug = x_batch_valid
            else:
                x_batch_valid_aug = augmentation_layer(aug_name, encoder_name)(x_batch_valid)
                x_batch_valid_aug = np.concatenate((x_batch_valid, x_batch_valid_aug))
                y_batch_valid = np.concatenate((y_batch_valid, y_batch_valid))
            if len(x_batch_valid_aug) <=2:
                pass
            else:
                with tf.GradientTape() as tape:
                    logits_valid = model(x_batch_valid_aug, training=False)
                    loss_vl = loss(y_batch_valid, logits_valid)
                print(f'step {step} loss validation: {loss_vl.numpy()}')
                valid_losses.append(loss_vl.numpy())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        if earlystop:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            model.save(path_ckpt)
    
    # model = tf.keras.models.load_model(path_ckpt, custom_objects={'SupervisedContrastiveLoss_Multilabel': SupervisedContrastiveLoss_Multilabel}, compile=False)
    model.load_weights(path_ckpt)
    return model

def trainer_enc_bench(model, x_train, x_valid, y_train, y_valid, aug_name, encoder_name, batch_size, num_epochs, loss, optimizer, earlystop, patience, path_ckpt):
    # model, x_train, x_valid, y_train, y_valid,  aug_name, batch_size, num_epochs, loss, optimizer, earlystop, patience, path_ckpt = encoder_with_projection_head, x_train, x_valid, y_train, y_valid, aug_name, batch_size, num_epochs, loss_enc, optimizer_enc, earlystop, patience, path_ckpt_enc
    # model = model_clf
    train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
    valid_ds=tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).batch(batch_size)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    if earlystop:
        early_stopping = EarlyStopping(patience = patience, verbose = True, path=path_ckpt)

    for epoch in range(1, num_epochs+1):
        print("\nStart of epoch %d" % (epoch,))
        # Training & Validation
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            if aug_name == 'None':
                x_batch_train_aug = x_batch_train
            else:
                x_batch_train_aug = augmentation_layer(aug_name, encoder_name)(x_batch_train)
                x_batch_train_aug = np.concatenate((x_batch_train, x_batch_train_aug))
                y_batch_train = np.concatenate((y_batch_train, y_batch_train))

            if len(x_batch_train_aug) <=2:
                pass
            else:
                with tf.GradientTape() as tape:
                    logits_train = model(x_batch_train_aug, training=True)
                    loss_tr = loss(y_batch_train, logits_train)
                    
                grads = tape.gradient(loss_tr, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                print(f'step {step} loss training: {loss_tr.numpy()}')
                train_losses.append(loss_tr.numpy())


        for step, (x_batch_valid, y_batch_valid) in enumerate(valid_ds):
            if aug_name == 'None':
                x_batch_valid_aug = x_batch_valid
            else:
                x_batch_valid_aug = augmentation_layer(aug_name, encoder_name)(x_batch_valid)
                x_batch_valid_aug = np.concatenate((x_batch_valid, x_batch_valid_aug))
                y_batch_valid = np.concatenate((y_batch_valid, y_batch_valid))
            if len(x_batch_valid_aug) <=2:
                pass
            else:
                with tf.GradientTape() as tape:
                    logits_valid = model(x_batch_valid_aug, training=False)
                    loss_vl = loss(y_batch_valid, logits_valid)
                print(f'step {step} loss validation: {loss_vl.numpy()}')
                valid_losses.append(loss_vl.numpy())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        if earlystop:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            model.save(path_ckpt)
    
    # model = tf.keras.models.load_model(path_ckpt, custom_objects={'SupervisedContrastiveLoss_Singlelabel': SupervisedContrastiveLoss_Singlelabel}, compile=False)
    model.load_weights(path_ckpt)
    return model

def create_encoder(input_shape, aug_name, encoder_name):
    # input_shape = (42, 60, 1)
    inputs = keras.Input(shape=input_shape)
    if aug_name == None:
        encoder_model = encoder_loader(input_shape, encoder_name)
        outputs = encoder_model(inputs)
    else:
        augmentation = layers.RandomFlip("horizontal")
        augmented = augmentation(inputs)
        encoder_model = encoder_loader(input_shape, encoder_name)
        outputs = encoder_model(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="encoder")
    return model

def create_encoder_batch(input_shape, encoder_name):
    # input_shape = (42, 60, 1)
    inputs = keras.Input(shape=input_shape)
    encoder_model = encoder_loader(input_shape, encoder_name)
    outputs = encoder_model(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="encoder")
    return model

def add_projection_head(input_shape, projection_units, encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
    )
    return model

class SupervisedContrastiveLoss_Multilabel(keras.losses.Loss):
    def __init__(self, temperature=1):
        super(SupervisedContrastiveLoss_Multilabel, self).__init__()
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        # feature_vectors = outputs
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        # tf.matmul (anchor, ) 
        similarity = tf.matmul(feature_vectors_normalized, tf.transpose(feature_vectors_normalized))
        logits = tf.divide(similarity, self.temperature)
        # SIGNLE LABEL: tfa.losses.npairs_loss(tf.squeeze(labels), logits)
        # MULTI LABEL: tfa.losses.npairs_multilabel_loss(tf.squeeze(labels), logits)
        SupConLoss = tfa.losses.npairs_multilabel_loss(tf.squeeze(labels), logits)
        return SupConLoss


class SupervisedContrastiveLoss_Singlelabel(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss_Singlelabel, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        # feature_vectors = logits_train
        # labels = y_batch_train, 
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        similarity = tf.matmul(feature_vectors_normalized, tf.transpose(feature_vectors_normalized))
        logits = tf.divide(similarity, self.temperature)
        # logits = tf.divide(similarity, 0.7)
        # SIGNLE LABEL: tfa.losses.npairs_loss(tf.squeeze(labels), logits)
        # MULTI LABEL: tfa.losses.npairs_multilabel_loss(tf.squeeze(labels), logits)
        return tfa.losses.npairs_multilabel_loss(tf.squeeze(labels), logits)

def create_classifier_Singlelabel(encoder, input_shape, n_y, dropout_rate, learning_rate, hidden_units, trainable=False):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(n_y, use_bias=False, activation="softmax")(features) # single label
    # outputs = layers.Dense(units=n_y, use_bias=False, activation='sigmoid')(features) # multi label


    model = keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(), #keras.losses.SparseCategoricalCrossentropy() # for integer 
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.FalsePositives(name='false_positives'),
                 tf.keras.metrics.FalseNegatives(name='false_negatives'),
                 #metrics=[keras.metrics.SparseCategoricalAccuracy()] 
                 ]
        )
    return model

def create_classifier_Multilabel(encoder, input_shape, n_y, dropout_rate, learning_rate, hidden_units, trainable=False):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    # outputs = layers.Dense(num_classes, activation="softmax")(features) # single label
    outputs = layers.Dense(units=n_y, use_bias=False, activation='sigmoid')(features) # multi label


    model = keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",#keras.losses.SparseCategoricalCrossentropy()
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.FalsePositives(name='false_positives'),
                 tf.keras.metrics.FalseNegatives(name='false_negatives'),
                 #metrics=[keras.metrics.SparseCategoricalAccuracy()] 
                 ]
        )
    return model
