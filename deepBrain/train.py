from datetime import datetime
import os.path
import re
import time
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
#import deepBrain
#import utils


def parser():
    'Parse command line args.'
    parser = argparse.ArgumentParser()
    parser.add_argument('patient', type=int,
                        help='Patient number for which the model is built')
    parser.add_argument('--train_dir', type=str,
                        default='..\\models\\train',
                        help='Directory to write event logs and checkpoints')
    parser.add_argument('--data_dir', type=str,
                        default='.\\data\\TFRecords',
                        help='Path to the TFRecords')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Number of batches to run')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of inputs to process in a batch')
    parser.add_argument('--temporal_stride', type=int, default=2,
                        help='Stride along time')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--shuffle', dest='shuffle',
                                action='store_true')
    feature_parser.add_argument('--no-shuffle', dest='shuffle',
                                action='store_false')
    parser.set_defaults(shuffle=True)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--use_fp16', dest='use_fp16',
                                action='store_true')
    feature_parser.add_argument('--use_fp32', dest='use_fp16',
                                action='store_false')
    parser.set_defaults(use_fp16=False)

    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probability for dropout')
    parser.add_argument('--num_hidden', type=int, default=1024,
                        help='Number of hidden nodes')
    parser.add_argument('--num_rnn_layers', type=int, default=2,
                        help='Number of recurrent layers')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Continue training from checkpoint file')
    parser.add_argument('--rnn_type', type=str, default='uni-dir',
                        help='uni-dir or bi-dir')
    parser.add_argument('--initial_lr', type=float, default=0.00001,
                        help='Initial learning rate for training')
    parser.add_argument('--num_filters', type=int, default=64,
                        help='Number of convolutional filters')
    parser.add_argument('--moving_avg_decay', type=float, default=0.9999,
                        help='Decay to use for the moving average of weights')
    parser.add_argument('--num_epochs_per_decay', type=int, default=5,
                        help='Epochs after which learning rate decays')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9,
                        help='Learning rate decay factor')
    
    args = parser.parse_args()

    # Read architecture hyper-parameters from checkpoint file
    # if one is provided.
    if args.checkpoint is not None:
        param_file = args.checkpoint + '/deepBrain_parameters.json'
        with open(param_file, 'r') as file:
            params = json.load(file)
            # Read network architecture parameters from previously saved
            # parameter file.
            args.num_hidden = params['num_hidden']
            args.num_rnn_layers = params['num_rnn_layers']
            args.rnn_type = params['rnn_type']
            args.num_filters = params['num_filters']
            args.use_fp16 = params['use_fp16']
            args.temporal_stride = params['temporal_stride']
            args.initial_lr = params['initial_lr']
    return args


def loss(scope, feats, labels, seq_lens):
    """Calculate the total loss of the deepBrain model.
    This function builds the graph for computing the loss.
    ARGS:
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
    logits = deepBrain.inference(feats, seq_lens, ARGS)

    # Build the portion of the Graph calculating the losses.
    strided_seq_lens = tf.div(seq_lens, ARGS.temporal_stride)
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
                             ARGS.batch_size)
    decay_steps = int(num_batches_per_epoch * ARGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        ARGS.initial_lr,
        global_step,
        decay_steps,
        ARGS.lr_decay_factor,
        staircase=True)

    return learning_rate, global_step


def fetch_data():
    """ Fetch features, labels and sequence_lengths from a common queue."""

    feats, labels, seq_lens = deepBrain.inputs(eval_data='train',
                                                data_dir=ARGS.data_dir,
                                                batch_size=ARGS.batch_size,
                                                use_fp16=ARGS.use_fp16,
                                                shuffle=ARGS.shuffle)

    return feats, labels, seq_lens


def main():
    #TODO
    

if __name__ == '__main__':
    ARGS = parser()
    main()
