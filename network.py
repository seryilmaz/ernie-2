# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import sparsegen3
from six.moves import urllib
import tensorflow as tf
import inputdata as cinput
import numpy as np

IMAGE_SIZE = cinput.IMAGE_SIZE
NUM_CLASSES = cinput.NUM_CLASSES 
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cinput.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_VAL = cinput.NUM_EXAMPLES_PER_EPOCH_FOR_VAL
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = cinput.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
IMAGE_SIZE = cinput.IMAGE_SIZE

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

use_fp16 = False
data_dir = '/tmp/data'
batch_size = 128


TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  #if not data_dir:
  #  raise ValueError('Please supply a data_dir')
  data_dir = '/tmp/data'
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
  images, labels = cinput.distorted_inputs(data_dir=data_dir,
                                                  batch_size=batch_size, phase=0)
  if use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(phase=1):
  data_dir = '/tmp/data'
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  #if not FLAGS.data_dir:
  #  raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
  images, labels = cinput.inputs(phase=phase,
                                        data_dir=data_dir,
                                        batch_size=batch_size)
  if use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels



def _variable_on_cpu(name, shape, init):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=init, dtype = dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if use_fp16 else tf.float32
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype = dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  
  return var
  
  
  
def _sparse_weight(name, indices, shape, values_length, stddev, wd):
  with tf.device('/cpu:0'):
    
    dtype = tf.float16 if use_fp16 else tf.float32
    
    values = tf.get_variable('sparse_values',( values_length),initializer=tf.truncated_normal_initializer(stddev=stddev, dtype = dtype),dtype=dtype)
    sp_input=tf.SparseTensor(indices=indices, values=values, shape=shape)
    #var=tf.sparse_tensor_to_dense(sp_input, default_value=0, validate_indices=True, name=name)
    if wd is not None:

      weight_decay = tf.mul(tf.nn.l2_loss(values), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
  
  return sp_input

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
    

def inference(images,wd, stddev1,stddev2,batch_size_inf=batch_size,localize=0,randomize=0):
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3,3,3,64], stddev = 5e-2, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding = 'SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_act = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_act, name = scope.name)
    _activation_summary(conv1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_act = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_act, name=scope.name)
    _activation_summary(conv2)

  # pool1
  pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool1')
                         
  #conv3                       
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3,3,64,128], stddev = 5e-2, wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding = 'SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_act = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_act, name = scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_act = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_act, name=scope.name)
    _activation_summary(conv4)

  # pool2
  pool2 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  #conv5                       
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3,3,128,256], stddev = 5e-2, wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding = 'SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_act = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(pre_act, name = scope.name)
    _activation_summary(conv5)

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_act = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(pre_act, name=scope.name)
    _activation_summary(conv6)

  # pool3
  pool3 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')
                         
  # local3
  #print(pool2.get_shape())
  with tf.variable_scope('local3') as scope:
    if (localize==1 and randomize ==1):
      print('random localization')
      indices = sparsegen3.indices_layer3
      reshape = tf.reshape(pool3,[batch_size_inf,-1])
      sparse_weights = _sparse_weight('sparse_weights', indices,(2304,504),129024, stddev=stddev1, wd=wd)  
      biases = _variable_on_cpu('biases', [504], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.transpose(tf.sparse_tensor_dense_matmul(sparse_weights, reshape, adjoint_a=True, adjoint_b=True, name=None)) + biases, name=scope.name)
    if (localize==1 and randomize ==0):
      print('patterned localization')
      tensor_group1 = tf.split(1,3,pool3,name='split1')
      
      tensor0 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split1')[0],name='split11')[0] , [batch_size_inf, -1])
      tensor1 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split2')[0],name='split22')[1] , [batch_size_inf, -1])
      tensor2 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split3')[0],name='split33')[2] , [batch_size_inf, -1])
      tensor3 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split4')[1],name='split44')[0] , [batch_size_inf, -1])
      tensor4 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split5')[1],name='split55')[1] , [batch_size_inf, -1])
      tensor5 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split6')[1],name='split66')[2] , [batch_size_inf, -1])
      tensor6 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split7')[2],name='split77')[0] , [batch_size_inf, -1])
      tensor7 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split8')[2],name='split88')[1] , [batch_size_inf, -1])
      tensor8 = tf.reshape(  tf.split(2,3,tf.split(1,3,pool3,name='split9')[2],name='split99')[2] , [batch_size_inf, -1])
      
      dim = tensor0.get_shape()[1].value
      weights0 = _variable_with_weight_decay('weights0', shape=[dim, 56], stddev=stddev1, wd=wd) 
      weights1 = _variable_with_weight_decay('weights1', shape=[dim, 56], stddev=stddev1, wd=wd)   
      weights2 = _variable_with_weight_decay('weights2', shape=[dim, 56], stddev=stddev1, wd=wd) 
      weights3 = _variable_with_weight_decay('weights3', shape=[dim, 56], stddev=stddev1, wd=wd)   
      weights4 = _variable_with_weight_decay('weights4', shape=[dim, 56], stddev=stddev1, wd=wd) 
      weights5 = _variable_with_weight_decay('weights5', shape=[dim, 56], stddev=stddev1, wd=wd)   
      weights6 = _variable_with_weight_decay('weights6', shape=[dim, 56], stddev=stddev1, wd=wd) 
      weights7 = _variable_with_weight_decay('weights7', shape=[dim, 56], stddev=stddev1, wd=wd)   
      weights8 = _variable_with_weight_decay('weights8', shape=[dim, 56], stddev=stddev1, wd=wd) 
                                                                                                 
      out0 = tf.matmul(tensor0,weights0)
      out1 = tf.matmul(tensor1,weights1)
      out2 = tf.matmul(tensor2,weights2)
      out3 = tf.matmul(tensor3,weights3)
      out4 = tf.matmul(tensor4,weights4)
      out5 = tf.matmul(tensor5,weights5)
      out6 = tf.matmul(tensor6,weights6)
      out7 = tf.matmul(tensor7,weights7)
      out8 = tf.matmul(tensor8,weights8)
          
      out_combined = tf.concat(1,[out0, out1,out2,out3,out4,out5,out6,out7,out8])                                    
      biases = _variable_on_cpu('biases', [504], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(out_combined+biases)
      
    if (localize ==0):
      print('no localization')
      reshape = tf.reshape(pool3, [batch_size_inf, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 504],
                                            stddev=stddev1, wd=wd)
      biases = _variable_on_cpu('biases', [504], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      
    
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    if (localize==1 and randomize==0):
      tensor_group_l4 = tf.split(1,2,local3,name='split')
      print('layer 4 patterned localization')
      weights1 = _variable_with_weight_decay('weights1', shape=[252, 128],
                                            stddev=stddev2, wd=wd)
      weights2 = _variable_with_weight_decay('weights2', shape=[252, 128],
                                            stddev=stddev2, wd=wd)
      out1 = tf.matmul(tensor_group_l4[0],weights1)
      out2 = tf.matmul(tensor_group_l4[1],weights2)
      out_comb = tf.concat(1,[out1, out2])
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(out_comb + biases, name=scope.name)

    if (localize==1 and randomize==1):
      print('layer 4 random localization')
      reshape = tf.reshape(local3,[batch_size_inf,-1])
      indices2 = sparsegen3.indices_layer4
      sparse_weights = _sparse_weight('sparse_weights', indices2,(504,256),64512, stddev=stddev2, wd=wd) 
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.transpose(tf.sparse_tensor_dense_matmul(sparse_weights, reshape, adjoint_a=True, adjoint_b=True, name=None)) + biases, name=scope.name)

    if (localize==0):
      print('layer 4 no localization')
      weights = _variable_with_weight_decay('weights', shape=[504, 256],
                                            stddev=stddev2, wd=wd)
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)    
      
    _activation_summary(local4)

  # linear layer(WX + b),
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [256, NUM_CLASSES],
                                          stddev=1/256.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')  
  
  
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op



def train(total_loss, global_step, learning_rate, lr_decay, momentum):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(learning_rate,
                                  global_step,
                                  decay_steps,
                                  lr_decay,
                                  staircase=False)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_nesterov=True)
    #opt = tf.train.AdamOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  
  
