#!/usr/bin/env python
# -*- coding: utf-8 -*-




from __future__ import absolute_import, division, print_function

import csv
import os

import numpy as np

np.random.seed(0)
ratio = 0.9


def gen_lines(path):
  lines = []
  for f in os.listdir(path):
    if f.endswith('.jpg'):
      lines.append(os.path.join(path, f))
  return lines


path = '/home/bonnie/companydata/mars_classifier'
dirs = os.listdir(path)
classes = {}
with open('./data/classes.txt', 'w') as f:
  for i, d in enumerate(dirs):
    classes[i] = d
    f.write('{}\n'.format(d))


total_lines = []
for d in dirs:
  total_lines.append(gen_lines('{}/{}'.format(path, d), d))

print(len(total_lines))  

for lines in total_lines:
  print(lines[0][1]+': '+str(len(lines)))

train_lines = []
valid_lines = []

for lines in total_lines:
  np.random.shuffle(lines)
  cut = int(len(lines)*ratio)
  train_lines += lines[:cut]
  valid_lines += lines[cut:]

np.random.shuffle(train_lines)
np.random.shuffle(valid_lines)


with open('train.csv', 'wb') as f:
  cw = csv.writer(f)
  for line in train_lines:
    cw.writerow(line)

with open('valid.csv', 'wb') as f:
  cw = csv.writer(f)
  for line in valid_lines:
    cw.writerow(line)
