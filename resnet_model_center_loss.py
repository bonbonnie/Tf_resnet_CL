#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
Modified from: https://github.com/ry/tensorflow-resnet
resnet/model.py
"""


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
CONV_WEIGHT_DECAY = 0.0005      #卷积层权重衰减率
CONV_WEIGHT_STDDEV = 0.1        #卷积层权重标准差
MOVING_AVERAGE_DECAY = 0.9997    #滑动平均模型衰减率
BN_DECAY = MOVING_AVERAGE_DECAY   #BN层衰减率
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'   # must be grouped with training op
FC_WEIGHT_STDDEV = 0.01       #全连接层权重标准差


def get_center_loss(features, labels, alpha, num_classes):
  """获取center loss及center的更新op

  Arguments:
      features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
      labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
      alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
      num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

  Return：
      loss: Tensor,可与softmax loss相加作为总的loss进行优化.
      centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
      centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
  """
  # 获取特征的维数，例如256维
  len_features = features.get_shape()[1]
  # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
  # 设置trainable=False是因为样本中心不是由梯度进行更新的
  centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                            initializer=tf.constant_initializer(0), trainable=False)
  # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
  labels = tf.argmax(labels, 1)  # 返回lables最大值索引号

  # 根据样本label,获取mini-batch中每一个样本对应的中心值
  centers_batch = tf.gather(centers, labels)
  # 计算loss
  loss = tf.reduce_mean(tf.nn.l2_loss(features - centers_batch))

  # 当前mini-batch的特征值与它们对应的中心值之间的差
  diff = centers_batch - features

  # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
  _, unique_idx, unique_count = tf.unique_with_counts(labels)
  appear_times = tf.gather(unique_count, unique_idx)
  appear_times = tf.reshape(appear_times, [-1, 1])

  diff = diff / tf.cast((1 + appear_times), tf.float32)
  diff = alpha * diff

  centers_update_op = tf.scatter_sub(centers, labels, diff)

  return loss, centers_update_op


class ResNetModel(object):

  def __init__(self, is_training, depth=50, num_classes=1000):
    self.is_training = is_training
    self.num_classes = num_classes
    self.depth = depth
    self.loss = np.inf
    self.gap = 0
    self.prob = 0
    self.center_loss = 0
    self.centers_update_op = None

    if depth in NUM_BLOCKS:
      self.num_blocks = NUM_BLOCKS[depth]
    else:
      raise ValueError('Depth is not supported; it must be 50, 101 or 152')

  def inference(self, x):
    # Scale 1
    with tf.variable_scope('scale1'):
      s1_conv = conv(x, ksize=7, stride=2, filters_out=64)
      s1_bn = bn(s1_conv, is_training=self.is_training)
      s1 = tf.nn.relu(s1_bn)

    # Scale 2
    with tf.variable_scope('scale2'):
      s2_mp = tf.nn.max_pool(s1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
      s2 = stack(s2_mp, is_training=self.is_training,
                 num_blocks=self.num_blocks[0], stack_stride=1, block_filters_internal=64)

    # Scale 3
    with tf.variable_scope('scale3'):
      s3 = stack(s2, is_training=self.is_training,
                 num_blocks=self.num_blocks[1], stack_stride=2, block_filters_internal=128)

    # Scale 4
    with tf.variable_scope('scale4'):
      s4 = stack(s3, is_training=self.is_training,
                 num_blocks=self.num_blocks[2], stack_stride=2, block_filters_internal=256)

    # Scale 5
    with tf.variable_scope('scale5'):
      s5 = stack(s4, is_training=self.is_training,
                 num_blocks=self.num_blocks[3], stack_stride=2, block_filters_internal=512)

    # global average pooling
    avg_pool = tf.reduce_mean(s5, reduction_indices=[1, 2], name='avg_pool')

    self.gap = avg_pool

    # flatten
    # shape = int(np.prod(s5.get_shape()[1:]))
    # avg_pool = tf.reshape(s5, shape=(-1, shape))

    with tf.variable_scope('fc'):
      self.prob = fc(avg_pool, num_units_out=self.num_classes)

    return self.prob

  def cal_loss(self, batch_x, batch_y=None):
    self.inference(batch_x)
    # add center loss
    self.center_loss, self.centers_update_op = get_center_loss(
        self.gap, batch_y, 0.5, self.num_classes)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prob, labels=batch_y)   #定义损失函数，计算logits和labels之间的softmax交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)     #计算张量维度上的平均值
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)    #获取losses
    self.loss = tf.add_n([cross_entropy_mean] + regularization_losses)   #总loss为交叉熵的平均值加上正则化的loss
    return self.loss

  def optimize(self, learning_rate, train_layers=None):
    trainable_var_names = ['weights', 'biases', 'beta', 'gamma']
    if train_layers:
      var_list = [v for v in tf.trainable_variables() if
                  v.name.split(':')[0].split('/')[-1] in trainable_var_names and
                  contains(v.name, train_layers)]
    else:
      var_list = None

    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=var_list)
    with tf.control_dependencies([self.centers_update_op]):
      train_op = tf.train.MomentumOptimizer(
          learning_rate, 0.9).minimize(self.loss, var_list=var_list)
    # train_op = tf.train.GradientDescentOptimizer(
    #     learning_rate).minimize(self.loss, var_list=var_list)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([self.loss]))

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    return tf.group(train_op, batchnorm_updates_op)


  def load_original_weights(self, session, path, skip_layers=None):
    weights_dict = np.load(path, encoding='bytes').item()

    for op_name in weights_dict:
      parts = op_name.split('/')

      if skip_layers:
        if contains(op_name, skip_layers):
          continue

      if parts[0] == 'fc' and self.num_classes != 1000:
        continue

      full_name = "{}:0".format(op_name)
      var = [v for v in tf.global_variables() if v.name == full_name][0]
      session.run(var.assign(weights_dict[op_name]))


# 做一个l2regularizer
def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
  "A little wrapper around tf.get_variable to do weight decay"
  if weight_decay > 0:
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    regularizer = None

  return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                         trainable=trainable)

# 定义卷积层
def conv(x, ksize, stride, filters_out):
  filters_in = x.get_shape()[-1]
  shape = [ksize, ksize, filters_in, filters_out]
  initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
  weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer,
                          weight_decay=CONV_WEIGHT_DECAY)
  return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


# 定义batch normalization层
def bn(x, is_training):
  x_shape = x.get_shape()   # 获取训练数据的shape
  params_shape = x_shape[-1:]

  axis = list(range(len(x_shape) - 1))

  beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
  gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

  moving_mean = _get_variable('moving_mean', params_shape,
                              initializer=tf.zeros_initializer(), trainable=False)
  moving_variance = _get_variable('moving_variance', params_shape,
                                  initializer=tf.ones_initializer(), trainable=False)

  # These ops will only be preformed when training.
  mean, variance = tf.nn.moments(x, axis)
  update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
  update_moving_variance = moving_averages.assign_moving_average(
      moving_variance, variance, BN_DECAY)
  tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
  tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

  mean, variance = control_flow_ops.cond(
      is_training, lambda: (mean, variance),
      lambda: (moving_mean, moving_variance))

  return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

# 定义堆叠block的函数
def stack(x, is_training, num_blocks, stack_stride, block_filters_internal):
  for n in range(num_blocks):
    block_stride = stack_stride if n == 0 else 1
    with tf.variable_scope('block%d' % (n + 1)):
      x = block(x, is_training, block_filters_internal=block_filters_internal,
                block_stride=block_stride)
  return x

# 定义block
def block(x, is_training, block_filters_internal, block_stride):
  filters_in = x.get_shape()[-1]   # 获取输入通道数

  m = 4
  filters_out = m * block_filters_internal   # 该block输出卷积核的个数
  shortcut = x


  # 一个残差单元的结构：
  with tf.variable_scope('a'):
    a_conv = conv(x, ksize=1, stride=block_stride, filters_out=block_filters_internal)
    a_bn = bn(a_conv, is_training)
    a = tf.nn.relu(a_bn)

  with tf.variable_scope('b'):
    b_conv = conv(a, ksize=3, stride=1, filters_out=block_filters_internal)
    b_bn = bn(b_conv, is_training)
    b = tf.nn.relu(b_bn)

  with tf.variable_scope('c'):
    c_conv = conv(b, ksize=1, stride=1, filters_out=filters_out)
    c = bn(c_conv, is_training)

  # 若输入输出通道不一致则再通过一个卷积层将其变一致
  with tf.variable_scope('shortcut'):
    if filters_out != filters_in or block_stride != 1:
      shortcut_conv = conv(x, ksize=1, stride=block_stride, filters_out=filters_out)
      shortcut = bn(shortcut_conv, is_training)

  return tf.nn.relu(c + shortcut)   # 返回经过残差单元的数据


# 定义全连接层
def fc(x, num_units_out):
  num_units_in = x.get_shape()[1]
  weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
  weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer,
                          weight_decay=FC_WEIGHT_STDDEV)
  biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
  return tf.nn.xw_plus_b(x, weights, biases)


def contains(target_str, search_arr):
  rv = False
  for search_str in search_arr:
    if search_str in target_str:
      rv = True
      break

  return rv
