import tensorflow as tf
import read_TFRec
#import rnn_cell

NUM_CLASSES = read_TFRec.NUM_CLASSES

FLAGS = tf.app.flags.FLAGS

# Optional arguments for the model hyperparameters.

tf.app.flags.DEFINE_integer('patient', default_value=1,
                            docstring='''Patient number for which the model is built.''')
tf.app.flags.DEFINE_string('train_dir', default_value='..\\models\\train',
                           docstring='''Directory to write event logs and checkpoints.''')
tf.app.flags.DEFINE_string('data_dir', default_value='.\\data\\TFRecords\\',
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
tf.app.flags.DEFINE_integer('num_conv_layers', default_value=1,
                           docstring='''Number of convolutional layers.''')
tf.app.flags.DEFINE_integer('num_rnn_layers', default_value=2,
                           docstring='''Number of recurrent layers.''')
tf.app.flags.DEFINE_string('checkpoint', default_value=None,
                           docstring='''Continue training from checkpoint file.''')
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

from util import _variable_on_cpu
from util import _variable_with_weight_decay
from util import _activation_summary

def inputs(eval_data, shuffle=False):
    """
    Construct input for the model evaluation using the Reader ops.

    :eval_data: 'train', 'test' or 'eval'
    :shuffle: bool, to shuffle the tfrecords or not. 
    
    :returns:
      feats: 3D tensor of [batch_size, T, CH] size.
      labels: Labels. 1D tensor of [batch_size] size.
      seq_lens: SeqLens. 1D tensor of [batch_size] size.

    :raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    feats, labels, seq_lens = read_TFRec.inputs(patientNr=FLAGS.patient,
                                                eval_data=eval_data,
                                                data_dir=FLAGS.data_dir,
                                                batch_size=FLAGS.batch_size,
                                                shuffle=shuffle)
    if FLAGS.use_fp16:
        feats = tf.cast(feats, tf.float16)
    return feats, labels, seq_lens


def conv_layer(l_input, kernel_shape):
    '''
    Convolutional layers wrapper function.

    :feats: input of conv layer
    :kernel_shape: shape of filter

    :returns:
       :conv_drop: tensor variable
       :kernel: tensor variable
    '''
    
    kernel = _variable_with_weight_decay(
        'weights',
        shape=kernel_shape,
        wd_value=None,
        use_fp16=FLAGS.use_fp16)

    conv = tf.nn.conv2d(l_input, kernel,
                        [1, FLAGS.temporal_stride, 1, 1],
                         padding='SAME')

    biases = _variable_on_cpu('biases', [FLAGS.num_filters],
                                tf.constant_initializer(-0.05),
                                FLAGS.use_fp16)
        
    bias = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv)

    # dropout
    conv_drop = tf.nn.dropout(conv, FLAGS.keep_prob)
    return conv_drop, kernel


def inference(feats, seq_lens):
    '''
    Build the deepBrain model.

    :feats: ECoG features returned from inputs().
    :seq_lens: Input sequence length for each utterance.

    :returns: logits.
    '''
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

    feat_len = feats.get_shape().as_list()[-1]

    # expand the dimension of feats from [batch_size, T, CH] to [batch_size, T, CH, 1]
    feats = tf.expand_dims(feats, dim=-1)
        
    # convolutional layers
    with tf.variable_scope('conv1') as scope:
        conv_drop, kernel = conv_layer(l_input=feats,
                                       kernel_shape=shape=[11, feat_len, 1, FLAGS.num_filters])

    if FLAGS.num_conv_layers > 1:
        for layer in range(2, FLAGS.num_conv_layers + 1):
            with tf.variable_scope('conv' + str(layer)) as scope:
                conv_drop, _ = conv_layer(l_input=conv_drop,
                                               kernel_shape=[11, feat_len, FLAGS.num_filters, FLAGS.num_filters])

    # recurrent layer
    with tf.variable_scope('rnn') as scope:

        # Reshape conv output to fit rnn input
        rnn_input = tf.reshape(conv_drop, [FLAGS.batch_size, -1,
                                            feat_len*FLAGS.num_filters])
        # Permute into time major order for rnn
        rnn_input = tf.transpose(rnn_input, perm=[1, 0, 2])
        # Make one instance of cell on a fixed device,
        # and use copies of the weights on other devices.
        cell = rnn_cell.CustomRNNCell(
            FLAGS.num_hidden, activation=tf.nn.relu6,
            use_fp16=FLAGS.use_fp16)
        drop_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, output_keep_prob=FLAGS.keep_prob)
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(
            [drop_cell] * FLAGS.num_rnn_layers)

        seq_lens = tf.div(seq_lens, FLAGS.temporal_stride)
        if FLAGS.rnn_type == 'uni-dir':
            rnn_outputs, _ = tf.nn.dynamic_rnn(multi_cell, rnn_input,
                                               sequence_length=seq_lens,
                                               dtype=dtype, time_major=True,
                                               scope='rnn',
                                               swap_memory=True)
        else:
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                multi_cell, multi_cell, rnn_input,
                sequence_length=seq_lens, dtype=dtype,
                time_major=True, scope='rnn',
                swap_memory=True)
            outputs_fw, outputs_bw = outputs
            rnn_outputs = outputs_fw + outputs_bw
        _activation_summary(rnn_outputs)

    # Linear layer(WX + b) - softmax is applied by CTC cost function.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', [FLAGS.num_hidden, NUM_CLASSES],
            wd_value=None,
            use_fp16=FLAGS.use_fp16)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0),
                                  FLAGS.use_fp16)
        logit_inputs = tf.reshape(rnn_outputs, [-1, cell.output_size])
        logits = tf.add(tf.matmul(logit_inputs, weights),
                        biases, name=scope.name)
        logits = tf.reshape(logits, [-1, FLAGS.batch_size, NUM_CLASSES])
        _activation_summary(logits)

    return logits

def loss(logits, labels, seq_lens):
    """Compute mean CTC Loss.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from inputs(). 1-D tensor
              of shape [batch_size]
      seq_lens: Length of each utterance for ctc cost computation.
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average ctc loss across the batch.
    ctc_loss = tf.nn.ctc_loss(inputs=tf.cast(logits, tf.float32),
                              labels=labels, sequence_length=seq_lens)
    ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss')
    tf.add_to_collection('losses', ctc_loss_mean)

    # The total loss is defined as the cross entropy loss plus all
    # of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
                            

def _add_loss_summaries(total_loss):
    """Add summaries for losses in deepBrain model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    
    :total_loss: Total loss from loss().
    
    :returns:
      :loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for each_loss in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average
        # version of the loss as the original loss name.
        tf.scalar_summary(each_loss.op.name + ' (raw)', each_loss)
        tf.scalar_summary(each_loss.op.name, loss_averages.average(each_loss))

    return loss_averages_op
