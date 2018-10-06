#!/usr/bin/python
# -*- coding: utf-8 -*-



from __future__ import absolute_import, division, print_function

import datetime
import os
import shutil
import time

import tensorflow as tf

from resnet_model import ResNetModel
from resnet_preprocessor import BatchPreprocessor
from resnet_utils import check_accuracy, gen_label_map, write_flags

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 101, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
# tf.app.flags.DEFINE_string('weights_path', '/home/bonnie/companydata/models/ResNet-L101.npy', 'Model path')
tf.app.flags.DEFINE_string('weights_path', '/home/bonnie/companydata/mars/models/resnet_20180419_134827/checkpoint/model_epoch30.ckpt',
                           'Model path')
tf.app.flags.DEFINE_string('train_layers', 'fc,scale5,scale4,scale3',
                           'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('skip_layers', 'fc',
                           'Load pretrained weights for layers skipping these, seperated by commas')
tf.app.flags.DEFINE_string('training_file', './data/train.csv', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', './data/valid.csv', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '/home/bonnie/companydata/mars/models',
                           'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 50, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS


def main(_):
  # Create training directories
  now = datetime.datetime.now()
  train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
  train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
  checkpoint_dir = os.path.join(train_dir, 'checkpoint')
  tensorboard_dir = os.path.join(train_dir, 'tensorboard')
  tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
  tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

  if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
  if not os.path.isdir(train_dir): os.mkdir(train_dir)
  if not os.path.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)
  if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
  if not os.path.isdir(tensorboard_train_dir): os.makedirs(tensorboard_train_dir)
  if not os.path.isdir(tensorboard_val_dir): os.makedirs(tensorboard_val_dir)

  # Write flags to txt
  write_flags(FLAGS, os.path.join(train_dir, 'flags.txt'))

  label_map = gen_label_map(FLAGS.training_file)
  for k, v in label_map.items():
    print('{} -> {}'.format(k, v))
  num_classes = len(label_map)

  # Batch preprocessors
  train_preprocessor = BatchPreprocessor(csv_file=FLAGS.training_file,
                                         label_map=label_map,
                                         output_size=[224, 224],
                                         horizontal_flip=True,
                                         aug=True,
                                         shuffle=True)
  val_preprocessor = BatchPreprocessor(csv_file=FLAGS.val_file,
                                       label_map=label_map,
                                       output_size=[224, 224])

  # Placeholders
  # x = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
  x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
  y = tf.placeholder(tf.float32, [None, num_classes], name='label')
  is_training = tf.placeholder('bool', [], name='is_training')

  # Model
  train_layers = FLAGS.train_layers.split(',')
  model = ResNetModel(is_training, depth=FLAGS.resnet_depth, num_classes=num_classes)
  loss = model.cal_loss(x, y)
  learning_rate = tf.placeholder(tf.float32)
  train_op = model.optimize(learning_rate, train_layers)

  # Training accuracy of the model
  prob = tf.identity(model.prob, name='prob')
  prediction = tf.to_int32(tf.argmax(prob, 1), name='prediction')

  # Summaries
  train_writer = tf.summary.FileWriter(tensorboard_train_dir)
  saver = tf.train.Saver(max_to_keep=FLAGS.num_epochs)  # save all models

  # Get the number of training/validation steps per epoch
  no = len(train_preprocessor.labels) // FLAGS.batch_size
  train_batches_per_epoch = no + 1 if len(train_preprocessor.labels) % FLAGS.batch_size else no
  no = len(val_preprocessor.labels) // FLAGS.batch_size
  val_batches_per_epoch = no + 1 if len(val_preprocessor.labels) % FLAGS.batch_size else no

  gpu_options = tf.GPUOptions()
  config = tf.ConfigProto(gpu_options=gpu_options)
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)



    # Load the pretrained weights
    # skip_layers = FLAGS.skip_layers.split(',')
    # model.load_original_weights(sess, FLAGS.weights_path, skip_layers=skip_layers)
    # checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
    # saver.save(sess, checkpoint_path)

    # load pretrained models
    saver.restore(sess, FLAGS.weights_path)

    print("{} Start training...".format(datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')))

    tf.train.write_graph(sess.graph_def, checkpoint_dir, 'mars_resnet_101.pb', as_text=True)

    for epoch in range(FLAGS.num_epochs):
      print("{} Epoch number: {}".format(datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'), epoch + 1))
      step = 1
      if epoch <= 10:
        lr = 0.01
      else:
        lr = 0.001
      print('Learning rate: {}.'.format(lr))
      tt = time.time()

      # Start training
      while step <= train_batches_per_epoch + int(train_batches_per_epoch/2):
        batch_xs, batch_ys, _ = train_preprocessor.next_batch(FLAGS.batch_size)
        _, l = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys,
                                                     is_training: True, learning_rate: lr})

        # Logging
        if step % FLAGS.log_step == 0:
          t = time.time() - tt
          print('Loss of minibatch: {:.3f}.'.format(l))
          print('Processing speed: {:.2f} images/second.'.format(
              FLAGS.batch_size * FLAGS.log_step / t))
          tt = time.time()

        step += 1

      # Epoch completed, start validation
      print("{} Start validation".format(datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')))

      train_precision, train_recall, _ = check_accuracy(sess, prediction, x, is_training, label_map,
                                                        preprocesser=train_preprocessor,
                                                        epochs=train_batches_per_epoch,
                                                        batch_size=FLAGS.batch_size)
      val_precision, val_recall, _ = check_accuracy(sess, prediction, x, is_training, label_map,
                                                    preprocesser=val_preprocessor,
                                                    epochs=val_batches_per_epoch,
                                                    batch_size=FLAGS.batch_size)
      lm = sorted(label_map.values())
      lm.append('all')
      print('=' * 10 + 'precision of train' + '=' * 10)
      for k in lm:
        if k not in train_precision:
          print('{} -> {:.3f}'.format(k, 0))
          continue
        print('{} -> {:.3f}'.format(k, train_precision[k]))
      print('=' * 10 + 'precision of valid' + '=' * 10)
      for k in lm:
        if k not in val_precision:
          print('{} -> {:.3f}'.format(k, 0))
          continue
        print('{} -> {:.3f}'.format(k, val_precision[k]))
      print('=' * 10 + 'recall of train' + '=' * 10)
      for k in lm:
        if k not in train_recall:
          print('{} -> {:.3f}'.format(k, 0))
          continue
        print('{} -> {:.3f}'.format(k, train_recall[k]))
      print('=' * 10 + 'recall of valid' + '=' * 10)
      for k in lm:
        if k not in val_recall:
          print('{} -> {:.3f}'.format(k, 0))
          continue
        print('{} -> {:.3f}'.format(k, val_recall[k]))

      print("{} Train Precision = {:.4f}".format(
          datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'), train_precision['all']))
      print("{} Validation Precision = {:.4f}".format(
          datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'), val_precision['all']))
      print("{} Train Recall = {:.4f}".format(
          datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'), train_recall['all']))
      print("{} Validation Recall = {:.4f}".format(
          datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'), val_recall['all']))

      # Reset the dataset pointers
      val_preprocessor.reset_pointer()
      train_preprocessor.reset_pointer()

      if epoch % 5 == 0:
        print("{} Saving checkpoint of model...".format(
            datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')))

      # save checkpoint of the model
      checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch' + str(epoch + 1) + '.ckpt')
      saver.save(sess, checkpoint_path)

      print("{} Model checkpoint saved at {}".format(
          datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'), checkpoint_path))



    train_precision, train_recall, error_cls = check_accuracy(sess, prediction, x, is_training, label_map,
                                                              preprocesser=train_preprocessor,
                                                              epochs=train_batches_per_epoch,
                                                              batch_size=FLAGS.batch_size)
    path = checkpoint_path
    if not os.path.isdir('{}/error'.format(path)):
      os.makedirs('{}/error'.format(path))
    print('=' * 10 + 'precision of train' + '=' * 10)
    for k in lm:
      if k not in train_precision:
        print('{} -> {:.3f}'.format(k, 0))
        continue
      print('{} -> {:.3f}'.format(k, train_precision[k]))
    print('=' * 10 + 'recall of train' + '=' * 10)
    for k in lm:
      if k not in train_recall:
        print('{} -> {:.3f}'.format(k, 0))
        continue
      print('{} -> {:.3f}'.format(k, train_recall[k]))
    for k in lm:
      if k not in error_cls:
        continue
      print('Train Error in {}'.format(k))
      for f, p in error_cls[k]:
        print('{} -> {}'.format(f, p))
        shutil.copy(
            f, '{}/error/train_{}_{}_{}.jpg'.format(path, f.rpartition('/')[-1][:-4], k, p))

    val_precision, val_recall, error_cls = check_accuracy(sess, prediction, x, is_training, label_map,
                                                          preprocesser=val_preprocessor,
                                                          epochs=val_batches_per_epoch,
                                                          batch_size=FLAGS.batch_size)
    print('=' * 10 + 'precision of valid' + '=' * 10)
    for k in lm:
      if k not in val_precision:
        print('{} -> {:.3f}'.format(k, 0))
        continue
      print('{} -> {:.3f}'.format(k, val_precision[k]))
    print('=' * 10 + 'recall of valid' + '=' * 10)
    for k in lm:
      if k not in val_recall:
        print('{} -> {:.3f}'.format(k, 0))
        continue
      print('{} -> {:.3f}'.format(k, val_recall[k]))
    for k in lm:
      if k not in error_cls:
        continue
      print('Valid Error in {}'.format(k))
      for f, p in error_cls[k]:
        print('{} -> {}'.format(f, p))
        shutil.copy(
            f, '{}/error/val_{}_{}_{}.jpg'.format(path, f.rpartition('/')[-1][:-4], k, p))


if __name__ == '__main__':
  tf.app.run()
