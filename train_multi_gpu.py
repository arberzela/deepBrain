from datetime import datetime
import os.path
import re
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import deepBrain
from deepBrain import FLAGS
import util

TOWER_NAME = 'tower'

def tower_loss(scope, feats, labels, seq_lens):
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
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '',
                           loss.op.name)
        # Name each loss as '(raw)' and name the moving average
        # version of the loss as the original loss name.
        tf.summary.scalar(loss_name + '(raw)', loss)
        tf.summary.scalar(loss_name, loss_averages.average(loss))

    # Without this loss_averages_op would never run
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the
       gradient has been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for each_grad, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(each_grad, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # The variables are redundant because they are shared
        # across towers. So we will just return the first tower's pointer to
        # the Variable.
        weights = grad_and_vars[0][1]
        grad_and_var = (grad, weights)
        average_grads.append(grad_and_var)
    return average_grads


def set_learning_rate():
    """ Learning rate reguralization. """

    # Create a variable to count the number of train() calls.
    # This equals the number of batches processed.
    global_step = tf.get_variable(name='global_step', shape=[],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (deepBrain.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
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


def initialize_from_checkpoint(sess, saver):
    """ Initialize variables on the graph"""
    # Initialise variables from a checkpoint file, if provided.
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/train/model.ckpt-0,
        # extract global_step from it.
        checkpoint_path = ckpt.model_checkpoint_path
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        return global_step
    else:
        print('No checkpoint file found')
        return


def train():
    """
    Train deepBrain for a number of steps.
    This function build a set of ops required to build the model and optimize
    weights.
    """
    with tf.Graph().as_default(), tf.device('/cpu'):
        
        # Learning rate set up
        learning_rate, global_step = set_learning_rate()

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Fetch a batch worth of data for each tower
        feats, labels, seq_lens = deepBrain.inputs(eval_data='train',
                                               shuffle=FLAGS.shuffle)

        tower_grads = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                name_scope = '%s_%d' % (TOWER_NAME, i)
                with tf.name_scope(name_scope) as scope:
                    # Calculate the loss for one tower of the deepBrain model.
                    # This function constructs the entire deepBrain model
                    # but shares the variables across all towers.
                    loss = tower_loss(scope, feats[i], labels[i], seq_lens[i])

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of
                    # data on this tower.
                    grads_and_vars = optimizer.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads_and_vars)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name
                                                      + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_avg_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Start running operations on the Graph. allow_soft_placement
        # must be set to True to build towers on GPU, as some of the
        # ops do not have GPU implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        # Initialize vars.
        if FLAGS.checkpoint is not None:
            global_step = initialize_from_checkpoint(sess, saver)
        else:
            sess.run(tf.global_variable_initializer(),
                     tf.local_variable_initializer())

        # Start the queue runners.
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss_op])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Print progress periodically.
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus
                
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # Run the summary ops periodically.
            if step % 50 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
       

def main(argv=None):
    """
    Creates checkpoint directory to save training progress and records
    training parameters in a json file before initiating the training session.
    """
    if FLAGS.train_dir != FLAGS.checkpoint:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    # Dump command line arguments to a parameter file,
    # in-case the network training resumes at a later time.
    with open(os.path.join(FLAGS.train_dir,
                           'deepBrain_parameters.json'), 'w') as outfile:
        json.dump(vars(FLAGS), outfile, sort_keys=True, indent=4)
    train()

if __name__ == '__main__':
    tf.app.run()

