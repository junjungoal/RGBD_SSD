# coding: utf-8

import scipy.io
import numpy as np
import argparse
from keras.preprocessing import image
import pickle

rgb = pickle.load(open('/home/jun/projects/ssd_keras/pkls/nyu_RGB.pkl', 'rb'))
depth = pickle.load(open('/home/jun/projects/ssd_keras/pkls/nyu_depth.pkl', 'rb'))

split = scipy.io.loadmat('./splits.mat')
train_index = split['trainNdxs']
test_index = split['testNdxs']

rgb_keys = sorted(rgb.keys())
depth_keys = sorted(depth.keys())

train_rgb = {}
train_depth = {}
test_rgb = {}
test_depth = {}
for rgb_key, depth_key in zip(rgb_keys, depth_keys):
    index = int(rgb_key[-8:-4])
    if index in train_index:
        train_rgb[rgb_key] = rgb[rgb_key]
        train_depth[depth_key] = depth[depth_key]
    if index in test_index:
        test_rgb[rgb_key] = rgb[rgb_key]
        test_depth[depth_key] = depth[depth_key]


pickle.dump(train_rgb, open('/home/jun/projects/ssd_keras/pkls/train_nyu_RGB.pkl', 'wb'))
pickle.dump(train_depth, open('/home/jun/projects/ssd_keras/pkls/train_nyu_depth.pkl', 'wb'))
pickle.dump(test_rgb, open('/home/jun/projects/ssd_keras/pkls/test_nyu_RGB.pkl', 'wb'))
pickle.dump(test_depth, open('/home/jun/projects/ssd_keras/pkls/test_nyu_depth.pkl', 'wb'))
