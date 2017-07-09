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

