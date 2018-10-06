#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Modified from: https://github.com/ry/tensorflow-resnet
utils/preprocessor.py
"""


from __future__ import absolute_import, division, print_function

import imgaug.augmenters as iaa
import numpy as np

import cv2


class BatchPreprocessor(object):

  def __init__(self, csv_file, label_map, output_size, horizontal_flip=False,
               aug=False, crop=False, shuffle=False):
    self.label_map = label_map
    self.label_map_inv = {v: k for k, v in label_map.items()}
    self.num_classes = len(self.label_map)
    self.output_size = output_size
    self.horizontal_flip = horizontal_flip
    self.aug = aug
    self.crop = crop
    self.shuffle = shuffle

    self.pointer = 0
    self.images = []
    self.labels = []

    # Read the dataset file
    with open(csv_file) as dataset_file:
      lines = dataset_file.readlines()
    for line in lines:
      items = line.strip().split(',')
      self.images.append(items[0])
      self.labels.append(self.label_map_inv[items[1]])

    if self.shuffle:
      self.shuffle_data()

  def shuffle_data(self):
    images = self.images[:]
    labels = self.labels[:]
    self.images = []
    self.labels = []

    idx = np.random.permutation(len(labels))
    for i in idx:
      self.images.append(images[i])
      self.labels.append(labels[i])

  def reset_pointer(self):
    self.pointer = 0

    if self.shuffle:
      self.shuffle_data()

  def next_batch(self, batch_size):
    if self.pointer + batch_size <= len(self.labels):
      paths = self.images[self.pointer:(self.pointer + batch_size)]
      labels = self.labels[self.pointer:(self.pointer + batch_size)]

      self.pointer += batch_size

    else:
      paths = self.images[self.pointer:]
      labels = self.labels[self.pointer:]
      self.reset_pointer()

    images = np.ndarray([len(paths), self.output_size[0], self.output_size[1], 3])
    for i, p in enumerate(paths):
      img = cv2.imread(p)
      large = np.min(img.shape[:-1]) > 800
      if large:
        img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

      # Flip image at random if flag is selected
      if self.horizontal_flip and np.random.random() < 0.3:
        img = cv2.flip(img, 1)

      if self.aug and np.random.random() < 0.9:
        seq = iaa.SomeOf((2, 5), [
            iaa.AdditiveGaussianNoise(loc=(0.8, 1.2), scale=(0, 3)),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            # iaa.GaussianBlur((0.9, 1.1)),
            iaa.ContrastNormalization((0.7, 1.3), per_channel=0.5)
        ])
        img = seq.augment_image(img)

      if self.crop and np.random.random() < 0.6 and large:
        seq = iaa.Sequential([
            iaa.Crop(percent=((0.05, 0.1), (0.05, 0.1), (0.05, 0.1), (0.05, 0.1)), keep_size=False)
        ])
        img = seq.augment_image(img)

      try:
        image = np.zeros((224, 224, 3), dtype=float)
        image[:, :, :] = [104, 117, 124]
        height = img.shape[0]
        width = img.shape[1]
        if max(height, width) > 224:
          if height > width:
            ratio = 224 / height
            width = int(ratio * width)
            img = cv2.resize(img, (width, 224))
          else:
            ratio = 224 / width
            height = int(ratio * height)
            img = cv2.resize(img, (224, height))
        height = img.shape[0]
        width = img.shape[1]
        if height >= width:
          ratio = 224 / height
          width = int(ratio * width)
          img = cv2.resize(img, (width, 224))
          image[:, int((224 - width) / 2):int((224 - width) / 2) + width, :] = img
        else:
          ratio = 224 / width
          height = int(ratio * height)
          img = cv2.resize(img, (224, height))
          image[int((224 - height) / 2):int((224 - height) / 2) + height, :, :] = img
      except IOError:
        print('Read image `{}` error.'.format(p))
        return -1, -1, -1

      image = 2 * (image / 255. - 0.5)
      images[i] = image

    # Expand labels to one hot encoding
    one_hot_labels = np.zeros((len(paths), self.num_classes))
    for i, l in enumerate(labels):
      one_hot_labels[i][l] = 1

    return images, one_hot_labels, paths
