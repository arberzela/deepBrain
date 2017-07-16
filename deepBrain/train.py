from datetime import datetime
import os.path
import re
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
#import deepBrain
#import utils

FLAGS = tf.app.flags.FLAGS

# Optional arguments for the model hyperparameters.

tf.app.flags.DEFINE_integer('patient', default_value=1,
                            docstring='''Patient number for which the model is built.''')
tf.app.flags.DEFINE_string('train_dir', default_value='..\\models\\train',
                           docstring='''Directory to write event logs and checkpoints.''')
tf.app.flags.DEFINE_string('data_dir', default_value='.\\data\\TFRecords',
                           docstring='''Path to the TFRecords.''')
tf.app.flags.DEFINE_integer('max_steps', default_value=20,
                           docstring='''Number of batches to run.''')
tf.app.flags.DEFINE_boolean('log_device_placement', default_value=False,
                           docstring='''Whether to log device placement.''')
tf.app.flags.DEFINE_integer('batch_size', default_value=32,
                           docstring='''Number of inputs to process in a batch.''')
tf.app.flags.DEFINE_integer('temporal_stride', default_value=2,
                           docstring='''Stride along time.''')
tf.app.flags.DEFINE_boolean('shuffle', default_value=True,
                           docstring='''Whether to shuffle or not the train data.''')
tf.app.flags.DEFINE_boolean('use_fp16', default_value=False,
                           docstring='''Type of data.''')
tf.app.flags.DEFINE_float('keep_prob', default_value=0.5,
                           docstring='''Keep probability for dropout.''')
tf.app.flags.DEFINE_integer('num_hidden', default_value=1024,
                           docstring='''Number of hidden nodes.''')
tf.app.flags.DEFINE_integer('num_rnn_layers', default_value=2,
                           docstring='''Number of recurrent layers.''')
tf.app.flags.DEFINE_string('checkpoint', default_value=None,
                           docstring='''Continue training from checkpoint file.''')
tf.app.flags.DEFINE_string('rnn_type', default_value='uni-dir',
                           docstring='''uni-dir or bi-dir.''')
tf.app.flags.DEFINE_float('initial_lr', default_value=0.00001,
                           docstring='''Initial learning rate for training.''')
tf.app.flags.DEFINE_integer('num_filters', default_value=0.9999,
                           docstring='''Number of convolutional filters.''')
tf.app.flags.DEFINE_float('moving_avg_decay', default_value=0.5,
                           docstring='''Decay to use for the moving average of weights.''')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', default_value=5,
                           docstring='''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('lr_decay_factor', default_value=0.9,
                           docstring='''Learning rate decay factor.''')

# Read architecture hyper-parameters from checkpoint file if one is provided.
if FLAGS.checkpoint is not None:
    param_file = FLAGS.checkpoint + '\\deepBrain_parameters.json'
    with open(param_file, 'r') as file:
        params = json.load(file)
        # Read network architecture parameters from previously saved
        # parameter file.
        FLAGS.num_hidden = params['num_hidden']
        FLAGS.num_rnn_layers = params['num_rnn_layers']
        FLAGS.rnn_type = params['rnn_type']
        FLAGS.num_filters = params['num_filters']
        FLAGS.use_fp16 = params['use_fp16']
        FLAGS.temporal_stride = params['temporal_stride']
        FLAGS.initial_lr = params['initial_lr']


def loss(scope, feats, labels, seq_lens):
    """Calculate the total loss of the deepBrain model.
    This function builds the graph for computing the loss.
    FLAGS:
      feats: Tensor of shape BxCHxT representing the
             brain features.
      labels: sparse tensor holding labels of each utterance.
      seq_lens: tensor of shape [batch_size] holding
              the sequence length per input utterance.
    Returns:
       Tensor of shape [batch_size] containing
       the total loss for a batch of data
    """

    # Build inference Graph.
    logits = deepBrain.inference(feats, seq_lens, FLAGS)

    # Build the portion of the Graph calculating the losses.
    strided_seq_lens = tf.div(seq_lens, FLAGS.temporal_stride)
    _ = deepBrain.loss(logits, labels, strided_seq_lens)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for loss in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a
        # multi-GPU training session. This helps the clarity
        # of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % helper_routines.TOWER_NAME, '',
                           loss.op.name)
        # Name each loss as '(raw)' and name the moving average
        # version of the loss as the original loss name.
        tf.scalar_summary(loss_name + '(raw)', loss)
        tf.scalar_summary(loss_name, loss_averages.average(loss))

    # Without this loss_averages_op would never run
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def set_learning_rate():
    """ Learning rate reguralization. """

    # Create a variable to count the number of train() calls.
    # This equals the number of batches processed.
    global_step = tf.get_variable(name='global_step', shape=[],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (deepBrain.NUM_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        FLAGS.initial_lr,
        global_step,
        decay_steps,
        FLAGS.lr_decay_factor,
        staircase=True)

    return learning_rate, global_step


def fetch_data():
    """ Fetch features, labels and sequence_lengths from a common queue."""

    feats, labels, seq_lens = deepBrain.inputs(eval_data='train',
                                                data_dir=FLAGS.data_dir,
                                                batch_size=FLAGS.batch_size,
                                                use_fp16=FLAGS.use_fp16,
                                                shuffle=FLAGS.shuffle)

    return feats, labels, seq_lens


def main(argv=None):
    #TODO
    
if __name__ == '__main__':
    tf.app.run()
