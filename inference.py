import tensorflow as tf
#import custom_RNN
import time
import logging
import numpy as np
from dataGenerator import DataGenerator

# Accounting the 0th indice +  space + blank label = 28 characters
NUM_CLASSES =  ord('z') - ord('a') + 1 + 1 + 1

# create logger
logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# create console and file handler and set level to debug
ch = logging.StreamHandler()
fh = logging.FileHandler('inference.log')
ch.setLevel(logging.DEBUG)
fh.setLevel(logging.DEBUG)
# add formatter to ch and fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

FLAGS = tf.app.flags.FLAGS

# suppress the warnings log
tf.logging.set_verbosity(tf.logging.ERROR)

# Optional arguments for the model hyperparameters.

tf.app.flags.DEFINE_integer('patient', default_value=1,
                            docstring='''Patient number for which the model is built.''')
tf.app.flags.DEFINE_string('train_dir', default_value='.\\tensorboard\\',
                           docstring='''Directory to write event logs and checkpoints.''')
tf.app.flags.DEFINE_string('data_dir', default_value='C:\\Users\\user\\Desktop\\tfVirtEnv\\deepBrain\\data\\json\\Patient1\\train.json',
                           docstring='''Path to the TFRecords.''')
tf.app.flags.DEFINE_integer('max_epochs', default_value=1,
                           docstring='''Number of batches to run.''')
tf.app.flags.DEFINE_boolean('log_device_placement', default_value=False,
                           docstring='''Whether to log device placement.''')
tf.app.flags.DEFINE_integer('batch_size', default_value=6,
                           docstring='''Number of inputs to process in a batch.''')
tf.app.flags.DEFINE_integer('temporal_stride', default_value=2,
                           docstring='''Stride along time.''')
tf.app.flags.DEFINE_boolean('sortagrad', default_value=True,
                           docstring='''Whether to sort the batches in the first epoch of train.''')
tf.app.flags.DEFINE_boolean('shuffle', default_value=True,
                           docstring='''Whether to shuffle or not the train data.''')
tf.app.flags.DEFINE_float('keep_prob', default_value=0.5,
                           docstring='''Keep probability for dropout.''')
tf.app.flags.DEFINE_integer('num_hidden', default_value=512,
                           docstring='''Number of hidden nodes.''')
tf.app.flags.DEFINE_integer('num_conv_layers', default_value=1,
                           docstring='''Number of convolutional layers.''')
tf.app.flags.DEFINE_integer('num_rnn_layers', default_value=1,
                           docstring='''Number of recurrent layers.''')
tf.app.flags.DEFINE_string('cell_type', default_value='LSTM',
                           docstring='''Type of cell to use for the recurrent layers.''')
tf.app.flags.DEFINE_string('rnn_type', default_value='uni-dir',
                           docstring='''uni-dir or bi-dir.''')
tf.app.flags.DEFINE_float('initial_lr', default_value=0.00001,
                           docstring='''Initial learning rate for training.''')
tf.app.flags.DEFINE_integer('num_filters', default_value=64,
                           docstring='''Number of convolutional filters.''')
tf.app.flags.DEFINE_float('moving_avg_decay', default_value=0.9999,
                           docstring='''Decay to use for the moving average of weights.''')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', default_value=5,
                           docstring='''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('lr_decay_factor', default_value=0.9,
                           docstring='''Learning rate decay factor.''')

logger.info('Running model for the following paramethers:')

train_data = DataGenerator(FLAGS.data_dir, FLAGS.batch_size)
valid_data = DataGenerator('C:\\Users\\user\\Desktop\\tfVirtEnv\\deepBrain\\data\\json\\Patient1\\valid.json', FLAGS.batch_size)

num_channels = 32 # compute from data

def inference_graph(inputs, seq_len, batch_size, prob, graph):
    '''
    Function to create the model graph for training and evaluation.
    :inputs: ECoG ndarray of shape (batch_size, T, CH)
    :seq_len: list (shape = (batch_size)) of ints which hold the lengh of each sequence
    :train: bool that indicates if the graph is used for train or evaluation
            If false deactivate the dropout layer.
    :returns: tf.graph, logits
    '''
    
    num_channels = 32 # compute from data
    with graph.as_default():
        # expand the dimension of feats from [batch_size, T, CH] to [batch_size, T, CH, 1]
        inputs = tf.expand_dims(inputs, dim=-1)

        # convolutional layers
        with tf.variable_scope('conv'):
            conv_weights = tf.get_variable(name='conv_weights',
                                           initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                           shape=(11, num_channels, 1, FLAGS.num_filters))

            conv = tf.nn.conv2d(inputs, conv_weights,
                                [1, FLAGS.temporal_stride, 1, 1],
                                 padding='SAME')

            biases = tf.get_variable(name='conv_biases',
                                     initializer=tf.constant_initializer(-0.05),
                                     shape=[FLAGS.num_filters])
        
            conv_out = tf.nn.relu(tf.nn.bias_add(conv, biases))

            conv_out = tf.nn.dropout(conv_out, prob)

        # recurrent layer
        with tf.variable_scope('rnn_layers'):
            # Reshape conv output to fit rnn input
            rnn_input = tf.reshape(conv_out, [FLAGS.batch_size, -1, num_channels*FLAGS.num_filters])
        
            if FLAGS.cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden,
                                               activation=tf.nn.relu6,
                                               state_is_tuple=True)
            #elif FLAGS.cell_type == 'CustomRNN':
             #   cell = custom_RNN.LayerNormalizedLSTMCell(FLAGS.num_hidden,
              #                                            activation=tf.nn.relu6,
               #                                           state_is_tuple=True)

            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=prob)

            multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_rnn_layers,
                                                     state_is_tuple=True)

            seq_len_conv = tf.div(seq_len, FLAGS.temporal_stride)
            if FLAGS.rnn_type == 'uni-dir':
                rnn_outputs, _ = tf.nn.dynamic_rnn(multi_cell, rnn_input,
                                                   sequence_length=seq_len_conv,
                                                   dtype=tf.float32)
            else:
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    multi_cell, multi_cell, rnn_input,
                    sequence_length=seq_len_conv, dtype=tf.float32)
                outputs_fw, outputs_bw = outputs
                rnn_outputs = outputs_fw + outputs_bw

        with tf.name_scope('fc_layer'):
            batch_s = tf.shape(inputs)[0]

            # Reshaping to apply the same weights over the timesteps
            fc_inputs = tf.reshape(rnn_outputs, [-1, cell.output_size])

            fc_weights = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 NUM_CLASSES],
                                                 stddev=0.1))
            # Zero initialization
            # Tip: Is tf.zeros_initializer the same?
            fc_biases = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]))

            # Doing the affine projection
            logits = tf.add(tf.matmul(fc_inputs, fc_weights), fc_biases)

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, NUM_CLASSES])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))

    return logits, seq_len_conv


def main(argv=None):
    # log the user paramethers used for the model
    for flag, value in tf.flags.FLAGS.__flags.items():
        logger.info('{}: {}'.format(flag, value))
        
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('inputs'):
            # Has size [batch_size, max_duration, num_channels], but the
            # batch_size and max_duration can vary along each step
    
            inputs = tf.placeholder(tf.float32, name='inputs', shape=[None, None, num_channels])

            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
            targets = tf.sparse_placeholder(tf.int32, name='targets')

            # 1d array of size [batch_size] that holds the the lengths for each
            # sequence in the batch
            seq_len = tf.placeholder(tf.int32, name='seq_len', shape=[None])

            prob = tf.placeholder_with_default(input=FLAGS.keep_prob, name='prob', shape=())
            batch_size = tf.placeholder_with_default(input=FLAGS.batch_size, name='batch_size', shape=())

        logits, seq_len_conv = inference_graph(inputs, seq_len, batch_size, prob, graph)
        
        loss = tf.nn.ctc_loss(targets, logits, seq_len_conv)
        cost = tf.reduce_mean(loss)

        # summary writer for loss in order to visualize progress in TensorBoard
        loss_summary = tf.summary.scalar('loss', cost)
        
        optimizer = tf.train.MomentumOptimizer(FLAGS.initial_lr,
                                               0.9).minimize(cost)

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len_conv)

        # Inaccuracy: character error rate
        cer = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))

        logger.info('Successfully created computational graph!!')
        
    with tf.Session(graph=graph) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        summary_train = tf.summary.FileWriter(FLAGS.train_dir+'train', sess.graph)
        summary_valid = tf.summary.FileWriter(FLAGS.train_dir+'valid', sess.graph)

        for epoch in range(FLAGS.max_epochs):
            train_cost = train_cer = 0
            valid_cost = valid_cer = 0
            start = time.time()
            
            logger.info('Calculating loss for epoch: '+str(epoch+1))
            for iteration, batch in enumerate(train_data.iterate_train(epoch, FLAGS.sortagrad)):

                batch_train_inputs = batch['x']
                batch_train_seq_len = batch['input_lengths']
                batch_train_targets = batch['sparse_y']

                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len,
                        batch_size: len(batch_train_inputs)}

                summary_str, batch_cost, _ = sess.run([merged, cost, optimizer], feed_dict=feed)
                summary_train.add_summary(summary_str, epoch * train_data.batches_per_epoch + iteration)
                logger.info("Epoch: {}, Iteration: {}, Loss: {}".format(epoch+1, iteration+1, batch_cost))
                train_cost += batch_cost
                train_cer += sess.run(cer, feed_dict=feed)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_cer = {:.3f}, time = {:.3f} \n"
            train_cost /=  train_data.batches_per_epoch # train cost for the entire epoch
            train_cer /= train_data.batches_per_epoch # train cer for the entire epoch
            logger.info(log.format(epoch+1, FLAGS.max_epochs, train_cost, train_cer, time.time() - start))

            # evaluate epoch over the validation set
            for iteration, batch in enumerate(valid_data.iterate_valid()):

                batch_valid_inputs = batch['x']
                batch_valid_seq_len = batch['input_lengths']
                batch_valid_targets = batch['sparse_y']

                valid_feed = {inputs: batch_valid_inputs,
                              targets: batch_valid_targets,
                              seq_len: batch_valid_seq_len,
                              prob: 1.0, # turn off dropout during validation
                              batch_size: len(batch_valid_inputs)} 

                summary_str, batch_cost = sess.run([merged, cost], feed_dict=valid_feed)
                summary.add_summary(summary_str, epoch * valid_data.batches_per_epoch + iteration)
                logger.info("Epoch: {}, Iteration: {}, Loss: {}".format(epoch+1, iteration+1, batch_cost))
                valid_cost += batch_cost
                valid_cer += sess.run(cer, feed_dict=valid_feed)

            log = "Epoch {}/{}, validation_cost = {:.3f}, validation_cer = {:.3f}, time = {:.3f} \n"
            valid_cost /=  valid_data.batches_per_epoch # validation cost for the entire epoch
            valid_cer /= valid_data.batches_per_epoch # validation cer for the entire epoch
            logger.info(log.format(epoch+1, FLAGS.max_epochs, valid_cost, valid_cer, time.time() - start))


if __name__ == '__main__':
    tf.app.run()
