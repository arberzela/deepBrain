import tensorflow as tf
import read_TFRec
#import rnn_cell
#from helper_routines import _variable_on_cpu
#from helper_routines import _variable_with_weight_decay
#from helper_routines import _activation_summary

NUM_CLASSES = read_TFRec.NUM_CLASSES

def inputs(patientNr, eval_data, data_dir, batch_size, use_fp16=False, shuffle=False):
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
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    feats, labels, seq_lens = read_TFRec.inputs(patientNr=patientNr,
                                                eval_data=eval_data,
                                                data_dir=data_dir,
                                                batch_size=batch_size,
                                                shuffle=shuffle)
    if use_fp16:
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
