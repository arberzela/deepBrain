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
