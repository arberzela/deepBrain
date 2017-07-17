import tensorflow as tf
import read_TFRec
#import rnn_cell
#from helper_routines import _variable_on_cpu
#from helper_routines import _variable_with_weight_decay
#from helper_routines import _activation_summary

NUM_CLASSES = read_TFRec.NUM_CLASSES

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

        
def inputs(eval_data, shuffle=False):
    """
    Construct input for the model evaluation using the Reader ops.

    :patientNr: int, patient number
    :eval_data: 'train', 'test' or 'eval'
    :data_dir: folder containing the pre-processed data
    :batch_size: int, size of mini-batch
    :use_fp16: bool, if True use fp16 else fp32
    :shuffle: bool, to shuffle the tfrecords or not. 
    
    :returns:
      feats: 4D tensor of [batch_size, T, CH, 1] size.
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


def inference(feats, seq_lens, params):
    '''
    Build the deepBrain model.

    :feats: ECoG features returned from inputs().
    :seq_lens: Input sequence length for each utterance.
    :params: parameters of the model.

    :returns: logits.
    '''
    if params.use_fp16:
        dtype = tf.float16
    else:
        dtype = tf.float32

    feat_len = feats.get_shape().as_list()[-1]

    # convolutional layers
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'wights',
            shape=[11, feat_len, 1, params.num_filters],
            wd_value=None,
            use_fp16=params.use_fp16)

        feats = tf.expand_dims(feats, dim=-1)
        conv = tf.nn.conv2d(feats, kernel,
                            [1, params.temoral_stride, 1, 1],
                             padding='SAME')

        biases = _variable_on_cpu('biases', [params.num_filters],
                                  tf.constant_initializer(-0.05),
                                  params.use_fp16)
        
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

        # dropout
        conv1_drop = tf.nn.dropout(conv1, params.keep_prob)

    # recurrent layer
