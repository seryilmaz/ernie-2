#!/home/burc/anaconda2/lib python
import os
import tensorflow as tf
import numpy as np
from six.moves import urllib
import gzip
import os
import re
import sys
import tarfile
#import matplotlib
#import matplotlib.pyplot as plt
##FLAGS = tf.app.flags.FLAGS
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=40000
NUM_EXAMPLES_PER_EPOCH_FOR_VAL=10000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST=10000
IMAGE_SIZE=24
NUM_CLASSES = 10

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_dir = '/tmp/data'
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_dir, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  
  tarfile.open(filepath, 'r:gz').extractall(dest_dir)

def read_cifar10(file_queue):

  class CIFAR10dummy(object):
    pass
  result = CIFAR10dummy();  


  data_dir = '/tmp/data/cifar-10-batches-bin'
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i) for i in range(1,6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Cannot find file: ' + f)

  #file_queue = tf.train.string_input_producer(filenames);    
  label_bytes = 1;
  image_bytes = 32*32*3;
  rec_bytes = label_bytes + image_bytes;

  print(filenames[0])
  reader = tf.FixedLengthRecordReader(record_bytes = rec_bytes);    
  result.key, value = reader.read(file_queue);     
  data_bytes = tf.decode_raw(value, tf.uint8);
  result.label = tf.cast(tf.slice(data_bytes, [0], [label_bytes]), tf.int32)
  depth_major = tf.reshape(tf.slice(data_bytes, [label_bytes],[image_bytes]), [3,32,32]);
  result.uint8image = tf.transpose(depth_major,[1,2,0]);
  return result;

                                         
def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch([image,label], batch_size = batch_size, num_threads = num_preprocess_threads, 
                                         capacity = min_queue_examples + 3*batch_size, 
                                         min_after_dequeue= min_queue_examples) 
  else:
    images, label_batch = tf.train.batch([image,label], batch_size = batch_size, num_threads = 1, 
                                         capacity = min_queue_examples + 3*batch_size)
                                         
  tf.summary.image('images', images)
  return images, tf.reshape(label_batch, [batch_size])                                                                                                           

  
def inputs(data_dir = '/tmp/data/cifar-10-batches-bin', phase = 0, batch_size=128, sweep =1 ):
  #maybe_download_and_extract()

  if phase ==0:
    files = [os.path.join(data_dir, 'data_batch_%d.bin' %i) for i in range(1,5)]
    num_examples_per_epoch= NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  if phase ==1:
    files = [os.path.join(data_dir, 'data_batch_5.bin')]
    num_examples_per_epoch= NUM_EXAMPLES_PER_EPOCH_FOR_VAL
  if phase ==2:
    files = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch= NUM_EXAMPLES_PER_EPOCH_FOR_TEST
  
  for f in files:
    if not tf.gfile.Exists(f):
      raise ValueError("Failed to find file:" +f)
  # Create a queue that produces the filenames to read.
  file_queue = tf.train.string_input_producer(files)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(file_queue)

  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE;
  width = IMAGE_SIZE;
  depth = 3;
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)
  # Image processing for training
  # 

  
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)
 
  return _generate_image_and_label_batch(float_image, read_input.label,min_queue_examples, batch_size,shuffle=False)
  
##phase 0: training; phase 1: validation; phase 2: testing
def distorted_inputs(data_dir = '/tmp/data/cifar-10-batches-bin', phase = 0, batch_size=128 , sweep = 1):
  #maybe_download_and_extract()
  
  
  if phase ==0:
    files = [os.path.join(data_dir, 'data_batch_%d.bin' %i) for i in range(1,6-sweep)]
    num_examples_per_epoch= NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  if phase ==1:
    files = [os.path.join(data_dir, 'data_batch_5.bin')]
    num_examples_per_epoch= NUM_EXAMPLES_PER_EPOCH_FOR_VAL
  if phase ==2:
    files = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch= NUM_EXAMPLES_PER_EPOCH_FOR_TEST
  
  for f in files:
    if not tf.gfile.Exists(f):
      raise ValueError("Failed to find file:" +f)
  # Create a queue that produces the filenames to read.
  file_queue = tf.train.string_input_producer(files)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(file_queue)
  print('size of read input image: %f', read_input.uint8image.get_shape())
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
 
  print('size of reshaped image: %f', reshaped_image.get_shape())
  height = IMAGE_SIZE;
  width = IMAGE_SIZE;
  depth = 3;
  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63/255.0)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)  
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)



