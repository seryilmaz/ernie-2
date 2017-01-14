from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os
import glob
import os.path
import time
import math
import numpy as np
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import network
import inputdata

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/tmp/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_examples_train', 40000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
eval_batch_size=1000
eval_train_batch_size=1000
log_dir = '/tmp/train'
log_dir2 = '/tmp/test'
batch_size=128

def eval_acc(saver, top_k_op):
  """Run Eval once.
  Args:
    saver: Saver.
    top_k_op: Top K op.
  """
  with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        num_iter = int(math.ceil(FLAGS.num_examples / eval_batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * eval_batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
          predictions = sess.run([top_k_op])
          true_count += np.sum(predictions)
          step += 1
        precision = true_count / total_sample_count
        print(total_sample_count)
        print(true_count)
        print('time: %s, val precision: %.3f' % (datetime.now(), precision))
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  return precision

def eval_acc_train(saver, top_k_op):
  """Run Eval once.
  Args:
    saver: Saver.
    top_k_op: Top K op.
  """
  with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        num_iter = int(math.ceil(FLAGS.num_examples_train / eval_batch_size))
        true_count = 0  
        total_sample_count = num_iter * eval_batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
          predictions = sess.run([top_k_op])
          true_count += np.sum(predictions)
          step += 1

        precision = true_count / total_sample_count
        print(total_sample_count)
        print(true_count)
        print('time: %s, train precision: %.3f' % (datetime.now(), precision))
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

def eval_init(wd,stddev1,stddev2, localize,randomize, sweep):
  """Eval CIFAR-10 for a number of steps."""
  with tf.device('/cpu:0'):
    with tf.Graph().as_default() as g:
      # Get images and labels for CIFAR-10.
      with tf.device('/cpu:0'):
        if sweep ==1:
          images, labels = inputdata.inputs(phase=1, batch_size=1000)
        if sweep ==0:
          images, labels = inputdata.inputs(phase=2, batch_size=1000)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = network.inference(images,wd, stddev1,stddev2, batch_size_inf=1000,localize=localize,randomize=randomize)
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.device('/cpu:0'):
          precision = eval_acc(saver, top_k_op)
  return precision

def eval_init_train(wd,stddev1,stddev2, localize,randomize, sweep):
  """Eval CIFAR-10 for a number of steps."""
  with tf.device('/cpu:0'):
    with tf.Graph().as_default() as g:
      # Get images and labels for CIFAR-10.
      with tf.device('/cpu:0'):
        if sweep ==1:
          images, labels = inputdata.inputs(phase=0, batch_size=1000)
        if sweep ==0:
          images, labels = inputdata.inputs(phase=0, batch_size=1000)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = network.inference(images,wd, stddev1,stddev2, batch_size_inf=1000,localize=localize,randomize=randomize)
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.device('/cpu:0'):
          eval_acc_train(saver, top_k_op)

def eval_acc_test(saver, top_k_op):
  """Run Eval once.

  Args:
    saver: Saver.
    top_k_op: Top K op.
  """
  with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      ckpt = tf.train.get_checkpoint_state(log_dir2)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        num_iter = int(math.ceil(FLAGS.num_examples / eval_batch_size))
        true_count = 0  
        total_sample_count = num_iter * eval_batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
          predictions = sess.run([top_k_op])
          
          true_count += np.sum(predictions)
          step += 1
        precision = true_count / total_sample_count
        print(total_sample_count)
        print(true_count)
        print('time: %s, test precision: %.3f' % (datetime.now(), precision))
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  return precision

def eval_init_test(wd,stddev1,stddev2, localize,randomize, sweep):
  """Eval CIFAR-10 for a number of steps."""
  with tf.device('/cpu:0'):
    with tf.Graph().as_default() as g:
      # Get images and labels for CIFAR-10.
      with tf.device('/cpu:0'): 
        images, labels = inputdata.inputs(phase=2, batch_size=1000)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = network.inference(images,wd, stddev1,stddev2, batch_size_inf=1000,localize=localize,randomize=randomize)
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.device('/cpu:0'):
          eval_acc_test(saver, top_k_op)


def train(lr,lr_decay,momentum, wd, stddev1,stddev2,max_steps,localize, randomize, sweep):
  with tf.Graph().as_default():
    global_step = tf.Variable(0,trainable = False)
    images, labels= inputdata.distorted_inputs(sweep=sweep)
    logits = network.inference(images,wd=wd,stddev1=stddev1,stddev2=stddev2, localize=localize,randomize=randomize)
    loss = network.loss(logits, labels)
    train_op = network.train(loss,global_step, learning_rate=lr, lr_decay=lr_decay, momentum=momentum)
    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep = 2)
    saver2 = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep = 2)
    summary_op =   tf.summary.merge_all()
    init = tf.initialize_all_variables()
    
    sess = tf.Session(config = tf.ConfigProto(log_device_placement = False))
    sess.run(init)
    
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    best_val_acc = 0
    for step in xrange(max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 1000 == 0:
        num_examples_per_step = batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      # Save the model checkpoint periodically.
      if step % 3000 == 0 or (step + 1) == max_steps:
        checkpoint_path = os.path.join(log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
      if step % 3000 == 0 or (step + 1) == max_steps:
        val_precision = eval_init(wd,stddev1,stddev2,localize, randomize, sweep)    
        wildcard = 'model.ckpt*'
        path = os.path.join(log_dir,wildcard)

        wildcard2 = 'events.out*'
        path = os.path.join(log_dir,wildcard2)
        for f in glob.glob(path):
          os.remove(f)  
        if val_precision > best_val_acc :
          best_val_acc = val_precision
          path = os.path.join(log_dir2,wildcard)
          checkpoint_path2 = os.path.join(log_dir2, 'model.ckpt')
          saver2.save(sess,checkpoint_path2, global_step=step)
    print("best val acc:")
    print(best_val_acc)
    eval_init_test(wd,stddev1,stddev2,localize, randomize, sweep)   

def main(argv=None):  # pylint: disable=unused-argument
  import sys
  network.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  lr = float(sys.argv[1])
  lr_decay = float(sys.argv[2])
  momentum = float(sys.argv[3])
  wd = float(sys.argv[4])
  stddev1 = float(sys.argv[5])
  stddev2 = float(sys.argv[6])
  max_steps = int(sys.argv[7])
  localize=int(sys.argv[8])
  randomize=int(sys.argv[9])
  sweep=int(sys.argv[10])
  print('learning rate:%.6f, lr_decay: %.3f, momentum: %.3f, weight decay coef: %.5f, init stddev: %.5f, init stddev2: %.5f,max_steps: %d, localize: %d, randomize: %d, sweep: %d' %(lr,lr_decay,momentum,wd,stddev1,stddev2,max_steps,localize, randomize, sweep))
  train(lr,lr_decay,momentum,wd,stddev1,stddev2, max_steps,localize=localize,randomize=randomize, sweep=sweep)
  
if __name__ == '__main__':
  tf.app.run()
