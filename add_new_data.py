# !/usr/bin/env python
# -*- coding: utf-8 -*-



from __future__ import absolute_import, division, print_function

import os
import csv

train_csv = 'train.csv'
valid_csv = 'valid.csv'

# with open(train_csv) as f:
#   train_lines = [line.strip() for line in f.readlines()]
with open(valid_csv) as f:
  valid_lines = [line.strip() for line in f.readlines()]

add_folder = '/home/bonnie/companydata/mars_new'
folders = os.listdir(add_folder)

valid_set = set(valid_lines)

train_lines = []
for folder in folders:
  _folder = os.path.join(add_folder, folder)
  images = [os.path.join(_folder, image) for image in os.listdir(_folder)
            if image.endswith('.jpg') and not image.startswith('.')]
  for image in images:
    if image+','+_folder.split('/')[-1] not in valid_set:
      train_lines.append('{},{}'.format(os.path.join(_folder, image), folder))

for folder in folders:
  count = 0
  for line in train_lines:
    if line.endswith(folder):
      count += 1
  print('{}->{}'.format(folder, count))

with open('train.csv', 'wb') as f:
  cw = csv.writer(f)
  for line in train_lines:
    cw.writerow(line.split(','))
