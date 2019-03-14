# coding: utf-8
import os, sys
import scipy.io
import numpy as np
import argparse
from keras.preprocessing import image

class MatPreprocessor(object):
    def __init__(self, mat_path, dataset_path):
        self.mat_path = mat_path
        self.dataset_path = dataset_path
        self.num_classes = 19
        self.rgb_data = dict()
        self.depth_data = dict()
        self.name_list = []
        self.nyu_rgb_data = dict()
        self.nyu_depth_data = dict()
        self._preprocess_mat()

    def _preprocess_mat(self):
        matdata = scipy.io.loadmat(self.mat_path)
        bboxes_data = matdata['SUNRGBDMeta2DBB'][0]
        path_prefix = self.dataset_path
        for data in bboxes_data:
            bounding_boxes = []
            one_hot_classes = []
            path = self.dataset_path + data[0][0]
            if len(data[1]) != 0:
                rgb_image_name = data[3][0].replace('/n/fs/sun3d/data/', '')
                depth_image_name = data[2][0].replace('/n/fs/sun3d/data/', '')
                img_path = path_prefix + rgb_image_name
                img = image.load_img(img_path)
                img = image.img_to_array(img)
                height = float(img.shape[0])
                width = float(img.shape[1])
                for klass in data[1][0]:
                    for bbox in klass[1]:
                        #xmin = int(bbox[0]) / width
                        #ymin = int(bbox[1])/ height
                        #xmax = (int(bbox[2]) +int(bbox[0])) / width
                        #ymax = (int(bbox[3]) + int(bbox[1])) / height
                        xmin = int(bbox[0])
                        ymin = int(bbox[1])
                        xmax = (int(bbox[2]) +int(bbox[0]))
                        ymax = (int(bbox[3]) + int(bbox[1]))
                    class_name = klass[2]
                    bounding_box = [xmin, ymin, xmax, ymax]
                    if 'NYU0011' in rgb_image_name:
                        print(class_name)
                    one_hot_class = self._to_one_hot(class_name)
                    if one_hot_class is None:
                        continue
                    bounding_boxes.append(bounding_box)
                    one_hot_classes.append(one_hot_class)
                if len(bounding_boxes) == 0:
                    continue
                image_data = np.hstack((bounding_boxes, one_hot_classes))
                if 'NYU' in rgb_image_name:
                    self.nyu_rgb_data[os.path.join(path_prefix, rgb_image_name)] = image_data
                    self.nyu_depth_data[os.path.join(path_prefix, depth_image_name)] = image_data
                self.rgb_data[os.path.join(path_prefix, rgb_image_name)] = image_data
                self.depth_data[os.path.join(path_prefix, depth_image_name)] = image_data


    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        if name == 'bathtub':
            one_hot_vector[0] = 1
        elif name == 'bed':
            one_hot_vector[1] = 1
        elif name == 'bookshelf':
            one_hot_vector[2] = 1
        elif name == 'box':
            one_hot_vector[3] = 1
        elif name == 'chair':
            one_hot_vector[4] = 1
        elif name == 'counter':
            one_hot_vector[5] = 1
        elif name == 'desk':
            one_hot_vector[6] = 1
        elif name == 'door':
            one_hot_vector[7] = 1
        elif name == 'dresser':
            one_hot_vector[8] = 1
        elif name == 'garbage_bin':
            one_hot_vector[9] = 1
        elif name == 'lamp':
            one_hot_vector[10] = 1
        elif name == 'monitor':
            one_hot_vector[11] = 1
        elif name == 'night_stand':
            one_hot_vector[12] = 1
        elif name == 'pillow':
            one_hot_vector[13] = 1
        elif name == 'sink':
            one_hot_vector[14] = 1
        elif name == 'sofa':
            one_hot_vector[15] = 1
        elif name == 'table':
            one_hot_vector[16] = 1
        elif name == 'tv':
            one_hot_vector[17] = 1
        elif name == 'toilet':
            one_hot_vector[18] = 1
        else:
            return None
        return one_hot_vector

parser = argparse.ArgumentParser(description='indicate specific image file path and mat file')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--mat_path', type=str)
args = parser.parse_args()

import pickle

preprocessed_data = MatPreprocessor(args.mat_path, args.dataset_path)
rgb_data = preprocessed_data.rgb_data
depth_data = preprocessed_data.depth_data

nyu_rgb_data = preprocessed_data.nyu_rgb_data
nyu_depth_data = preprocessed_data.nyu_depth_data

pickle.dump(nyu_rgb_data, open('/home/jun/projects/ssd_keras/pkls/nyu_RGB.pkl', 'wb'))
pickle.dump(nyu_depth_data, open('/home/jun/projects/ssd_keras/pkls/nyu_depth.pkl', 'wb'))

#pickle.dump(rgb_data, open('../pkls/RGB.pkl', 'wb'))
#pickle.dump(depth_data, open('../pkls/depth.pkl', 'wb'))

