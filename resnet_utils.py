#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.metrics import confusion_matrix


def write_flags(flags, filename):
  with open(filename, 'w') as flags_file:
    flags_file.write('resnet_depth={}\n'.format(flags.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(flags.num_epochs))
    flags_file.write('batch_size={}\n'.format(flags.batch_size))
    flags_file.write('train_layers={}\n'.format(flags.train_layers))
    flags_file.write('tensorboard_root_dir={}\n'.format(flags.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(flags.log_step))
    flags_file.write('load weights from: {}'.format(flags.weights_path))


def gen_label_map(csv_file):
  with open(csv_file) as f:
    lines = [line.strip().split(',')[-1] for line in f.readlines()]

  lines = list(set(lines))
  label_map = {}
  for i, line in enumerate(lines):
    label_map[i] = line

  return label_map


def cal_print_cm(labels, predictions, label_map):
  cm = confusion_matrix(labels, predictions)
  precision = {}
  recall = {}
  pr = 0
  ap = ar = 0
  for i, c in enumerate(cm):
    pr += c[i]
    ap += sum(cm[:, i])
    ar += sum(c)
    precision[label_map[i]] = c[i]/sum(cm[:, i])
    recall[label_map[i]] = c[i]/sum(c)

  precision['all'] = pr/ap
  recall['all'] = pr/ar

  for i, c in enumerate(cm):
    print('{} {}'.format(label_map[i], c))

  return precision, recall


def check_accuracy(sess, prediction, x, is_training, label_map, preprocesser, epochs, batch_size):
  preprocesser.reset_pointer()
  error_cls = {}
  apredictions = []
  alabels = []
  step = 1
  while step <= epochs:
    batch_xs, lbls, paths = preprocesser.next_batch(batch_size)
    preds = sess.run(prediction, feed_dict={x: batch_xs, is_training: False})
    step += 1
    lbls = np.argmax(lbls, axis=1)
    apredictions.extend(preds)
    alabels.extend(lbls)
    for l, p, f in zip(lbls, preds, paths):
      if p != l:
        error_cls.setdefault(label_map[l], []).append((f, label_map[p]))

  precision, recall = cal_print_cm(alabels, apredictions, label_map)

  return precision, recall, error_cls
