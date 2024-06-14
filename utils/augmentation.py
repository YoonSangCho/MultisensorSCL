# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 03:51:32 2021

@author: YoonSangCho
`
https://github.com/uchidalab/time_series_augmentation

"""
#%% augmentation methods
import numpy as np
import tensorflow

import tensorflow as tf
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import tensorflow_addons as tfa
import tensorflow.experimental.numpy as tnp

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    # x = x_re
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret


#%% augemtnation KERAS LAYER
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow import keras
from tensorflow.keras import layers as layers
import numpy as np

def run_augmentation(x_batch, aug_name, encoder_name):
    if aug_name == 'horizontal':
        x_aug = tf.keras.layers.RandomFlip('horizontal')(x_batch)
    else:
        if '1D' in encoder_name: 
            # ['vggnet_small_1D', 'vggnet_19_1D', 'resnet_18_1D', 'resnet_34_1D', 'resnet_50_1D', 'resnet_101_1D', 'resnet_152_1D']:
            # x_batch = tf.cast(x_batch_train, tf.float64)
            # x_batch = tf.cast(tnp.swapaxes(x_train[:100], 1, 3), tf.float64)
            
            x_re = tf.squeeze(x_batch, axis=1) # x_re.shape
            
            ################################################################################ 1D Augmentation
            if aug_name == 'jitter':
                x_aug_re = jitter(x_re, sigma=0.03)
            elif aug_name == 'scaling':
                x_aug_re = scaling(x_re, sigma=0.1)
            elif aug_name == 'jitter&scaling':
                x_aug_re = jitter(x_re, sigma=0.03)
                x_aug_re = scaling(x_aug_re, sigma=0.1)
            elif aug_name == 'jitter&horizontal':
                x_aug_re = jitter(x_re, sigma=0.03)
                x_aug_re = tnp.expand_dims(tnp.moveaxis(x_aug_re, 1, 2), -1)
                x_aug_re = tf.keras.layers.RandomFlip('horizontal')(x_aug_re)
                x_aug_re = tnp.moveaxis(x_aug_re, 1, 2) # x_re.shape
                x_aug_re = tf.squeeze(x_aug_re, axis=-1) # x_re.shape
            elif aug_name == 'scaling&horizontal':
                x_aug_re = scaling(x_re, sigma=0.1)
                x_aug_re = tnp.expand_dims(tnp.moveaxis(x_aug_re, 1, 2), -1)
                x_aug_re = tf.keras.layers.RandomFlip('horizontal')(x_aug_re)
                x_aug_re = tnp.moveaxis(x_aug_re, 1, 2) # x_re.shape
                x_aug_re = tf.squeeze(x_aug_re, axis=-1) # x_re.shape
            elif aug_name == 'jitter&scaling&horizontal':
                x_aug_re = jitter(x_re, sigma=0.03)
                x_aug_re = scaling(x_aug_re, sigma=0.1)
                x_aug_re = tnp.expand_dims(tnp.moveaxis(x_aug_re, 1, 2), -1)
                x_aug_re = tf.keras.layers.RandomFlip('horizontal')(x_aug_re)
                x_aug_re = tnp.moveaxis(x_aug_re, 1, 2) # x_re.shape
                x_aug_re = tf.squeeze(x_aug_re, axis=-1) # x_re.shape
            elif aug_name == 'magnitude_warp':
                x_aug_re = magnitude_warp(x_re, sigma=0.2, knot=4)
            elif aug_name == 'time_warp':
                x_aug_re = time_warp(x_re, sigma=0.2, knot=4)
            elif aug_name == 'window_slice':
                x_aug_re = window_slice(x_re, reduce_ratio=0.9)
            elif aug_name == 'window_warp':
                x_aug_re = window_warp(x_re, window_ratio=0.1, scales=[0.5, 2.])
            ################################################################################ 
            x_aug = tf.expand_dims(x_aug_re, axis=1)
            
        else:
            #################### reshape: augmentation methods require the shape of (num. obs., time axis, channel axis)
            # x_batch = tf.cast(x_batch_train, tf.float64) # x_batch.shape
            x_re = tnp.moveaxis(x_batch, 1, 2) # x_re.shape
            x_re = tf.squeeze(x_re, axis=-1) # x_re.shape
            
            ################################################################################ Augmentation 
            if aug_name == 'jitter':
                x_aug_re = jitter(x_re, sigma=0.03)
            elif aug_name == 'scaling':
                x_aug_re = scaling(x_re, sigma=0.1)
            elif aug_name == 'jitter&scaling':
                x_aug_re = jitter(x_re, sigma=0.03)
                x_aug_re = scaling(x_aug_re, sigma=0.1)
            elif aug_name == 'jitter&horizontal':
                x_aug_re = jitter(x_re, sigma=0.03)
                x_aug_re = tnp.expand_dims(tnp.moveaxis(x_aug_re, 1, 2), -1)
                x_aug_re = tf.keras.layers.RandomFlip('horizontal')(x_aug_re)
            elif aug_name == 'scaling&horizontal':
                x_aug_re = scaling(x_re, sigma=0.1)
                x_aug_re = tnp.expand_dims(tnp.moveaxis(x_aug_re, 1, 2), -1)
                x_aug_re = tf.keras.layers.RandomFlip('horizontal')(x_aug_re)
            elif aug_name == 'jitter&scaling&horizontal':
                x_aug_re = jitter(x_re, sigma=0.03)
                x_aug_re = scaling(x_aug_re, sigma=0.1)
                x_aug_re = tnp.expand_dims(tnp.moveaxis(x_aug_re, 1, 2), -1)
                x_aug_re = tf.keras.layers.RandomFlip('horizontal')(x_aug_re)
            elif aug_name == 'magnitude_warp':
                x_aug_re = magnitude_warp(x_re, sigma=0.2, knot=4)
            elif aug_name == 'time_warp':
                x_aug_re = time_warp(x_re, sigma=0.2, knot=4)
            elif aug_name == 'window_slice':
                x_aug_re = window_slice(x_re, reduce_ratio=0.9)
            elif aug_name == 'window_warp':
                x_aug_re = window_warp(x_re, window_ratio=0.1, scales=[0.5, 2.])
            ################################################################################ 
            if aug_name == 'jitter&horizontal' or aug_name == 'scaling&horizontal' or aug_name == 'jitter&scaling&horizontal':
                x_aug = x_aug_re
            else:
                x_aug = tnp.expand_dims(tnp.moveaxis(x_aug_re, 1, 2), -1)
    return x_aug

class augmentation_layer(keras.layers.Layer):
    def __init__(self, aug_name = 'jitter', encoder_name = 'vggnet', **kwargs):
        super(augmentation_layer, self).__init__()
        self.aug_name = aug_name
        self.encoder_name = encoder_name
    def call(self, x):
        x_aug = run_augmentation(tf.cast(x, tf.float64), self.aug_name, self.encoder_name)
        return x_aug

#%% examples: augemtnation NUMPY

# def run_augmentation(x, aug_name):
#     # list = ['horizontal', 'jitter', 'scaling', 'permutation', 'magnitude_warp', 'time_warp', 'window_slice', 'window_warp']
#     # keras layers: 'horizontal'
#     # numpy: 'jitter', 'scaling', 'rotation', 'permutation', 'magnitude_warp', 'time_warp', 'window_slice', 'window_warp'
    
#     # jitter(x, sigma=0.03)
#     # scaling(x, sigma=0.1)
#     # rotation(x)
#     # permutation(x, max_segments=5, seg_mode="equal")
#     # magnitude_warp(x, sigma=0.2, knot=4)
#     # time_warp(x, sigma=0.2, knot=4)
#     # window_slice(x, reduce_ratio=0.9)
#     # window_warp(x, window_ratio=0.1, scales=[0.5, 2.])
    
#     # x =  np.array([[[1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0],
#     #                         [4.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
#     #                         [7.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0]],
#     #                         [[10.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0],
#     #                         [13.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0],
#     #                         [16.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0]]])
#     # x = np.expand_dims(x, -1)
#     # print(x.shape)
#     import tensorflow
#     if aug_name == 'horizontal':
#         x_aug = tensorflow.keras.layers.RandomFlip('horizontal')(x)
#     else:
#         x_re = np.moveaxis(x, 1, 2)
#         x_re = x_re.reshape(x_re.shape[0], x_re.shape[1], x_re.shape[2])
#         if aug_name == 'jitter':
#             x_aug_re = jitter(x_re, sigma=0.03)
#         elif aug_name == 'scaling':
#             x_aug_re = scaling(x_re, sigma=0.1)
#         elif aug_name == 'rotation':
#             x_aug_re = rotation(x_re)
#         elif aug_name == 'permutation':
#             x_aug_re = permutation(x_re, max_segments=5, seg_mode="equal")
#         elif aug_name == 'magnitude_warp':
#             x_aug_re = magnitude_warp(x_re, sigma=0.2, knot=4)
#         elif aug_name == 'time_warp':
#             x_aug_re = time_warp(x_re, sigma=0.2, knot=4)
#         elif aug_name == 'window_slice':
#             x_aug_re = window_slice(x_re, reduce_ratio=0.9)
#         elif aug_name == 'window_warp':
#             x_aug_re = window_warp(x_re, window_ratio=0.1, scales=[0.5, 2.])
#         x_aug = np.expand_dims(np.moveaxis(x_aug_re, 1, 2), -1)
#         # print(x_aug, x_aug.shape)
#     return x_aug
'''
def run_augmentation_tnp(x_batch, aug_name, encoder_name):
    if aug_name == 'horizontal':
        x_aug = tf.keras.layers.RandomFlip('horizontal')(x_batch)
    else:
        if encoder_name in ['rnn', 'lstm', 'gru', 'bilstm']:
            x_re = x_batch
        
            if aug_name == 'jitter':
                x_aug_re = jitter(x_re, sigma=0.03)
            elif aug_name == 'scaling':
                x_aug_re = scaling(x_re, sigma=0.1)
            elif aug_name == 'rotation':
                x_aug_re = rotation(x_re)
            elif aug_name == 'permutation':
                x_aug_re = permutation(x_re, max_segments=5, seg_mode="equal")
            elif aug_name == 'magnitude_warp':
                x_aug_re = magnitude_warp(x_re, sigma=0.2, knot=4)
            elif aug_name == 'magnitude_warp_tnp':
                x_aug_re = magnitude_warp_tnp(x_re, sigma=0.2, knot=4)
            elif aug_name == 'time_warp':
                x_aug_re = time_warp(x_re, sigma=0.2, knot=4)
            elif aug_name == 'window_slice':
                x_aug_re = window_slice(x_re, reduce_ratio=0.9)
            elif aug_name == 'window_warp':
                x_aug_re = window_warp(x_re, window_ratio=0.1, scales=[0.5, 2.])
            x_aug = x_aug_re
'''

'''
x =  np.array([[[1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0, 8.0, 9.0]],
                        [[10.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0]]])
x = np.expand_dims(x, -1)
print(x.shape)
x_re = x.copy()
x_re = np.moveaxis(x, 1, 2)
x_re = x_re.reshape(x_re.shape[0], x_re.shape[1], x_re.shape[2])
print(x_re.shape)
# augmentation inputshape (#batch, #timesteps, #dimensions)
x_aug_re = jitter(x_re, sigma=0.03)
print(x_aug_re, x_aug_re.shape)
x_aug_re = scaling(x_aug_re, sigma=0.1)
print(x_aug_re, x_aug_re.shape)
# x_aug_re = rotation(x_re)
# print(x_aug_re, x_aug_re.shape)
x_aug_re = permutation(x_re, max_segments=5, seg_mode="equal")
print(x_aug_re, x_aug_re.shape)

x_aug_re = magnitude_warp(x_re, sigma=0.2, knot=4)
print(x_aug_re, x_aug_re.shape)
x_aug_re = magnitude_warp_tnp(x_re, sigma=0.2, knot=4)
print(x_aug_re, x_aug_re.shape)
x_aug_re = time_warp(x_re, sigma=0.2, knot=4)
print(x_aug_re, x_aug_re.shape)
x_aug_re = window_slice(x_re, reduce_ratio=0.9)
print(x_aug_re, x_aug_re.shape)
x_aug_re = window_warp(x_re, window_ratio=0.1, scales=[0.5, 2.])
print(x_aug_re, x_aug_re.shape)

x_aug = np.expand_dims(np.moveaxis(x_aug_re, 1, 2), -1)
print(x_aug, x_aug.shape)
'''