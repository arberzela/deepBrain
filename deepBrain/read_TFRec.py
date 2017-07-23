import os.path
import glob
import tensorflow as tf
from create_TFRec import ALPHABET
from DataExtraction.speech_related import speech_channels
os.chdir('..')


NUM_CLASSES = len(ALPHABET) + 1  # Additional class for blank character


def read_and_decode(filename_queue, nr_channels, batch_size):
    """
    Construct a queued batch of ECoG features and transcripts.
   
    :filename_queue: queue of filenames to read data from.
    :nr_channels: number of valid ECoG channels for the specified patient
    :batch_size: Number of utterances per batch.
    
    :returns:
      feats: 4D tensor of [batch_size, T, CH, 1] size.
      labels: transcripts. List of length batch_size.
      seq_lens: Sequence Lengths. List of length batch_size.
    """

    # Define how to parse the example
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    context_features = {
        "seq_len": tf.FixedLenFeature([], dtype=tf.int64),
        "labels": tf.VarLenFeature(dtype=tf.int64)
    }
    sequence_features = {
        # ECoG features are nr_channels-dimensional
        "feats": tf.FixedLenSequenceFeature([nr_channels, ], dtype=tf.float32) 
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # Generate a batch worth of examples after bucketing
    seq_len, (feats, labels) = tf.contrib.training.bucket_by_sequence_length(
        input_length=tf.cast(context_parsed['seq_len'], tf.int32),
        tensors=[sequence_parsed['feats'], context_parsed['labels']],
        batch_size=batch_size,
        bucket_boundaries=list(range(100, 1900, 100)),
        allow_smaller_final_batch=True,
        num_threads=16,
        dynamic_pad=True)

    return feats, tf.cast(labels, tf.int32), seq_len


def getKey(filename):
    # get the file's text name without extension
    file_text_name = os.path.splitext(os.path.basename(filename))
    # get two elements, the last one is the number. Sort based on this number
    file_last_num = os.path.basename(file_text_name[0]).split('_')
    
    return int(file_last_num[1])


def inputs(patientNr, eval_data, data_dir, batch_size, shuffle=False):
    """
    This function constructs the input for the neural network after reading the
    TFRecord files for the secified patient.

    :patientNr: int, indicating the patient.
    :eval_data: bool, indicating if one should use the train, validation or test set.
    :data_dir: Path to the data directory.
    :batch_size: Number of utterances per batch.
    :shuffle: bool, indicating if the data should be shuffled or not.
    
    :returns:
      feats: 4D tensor of [batch_size, T, CH, 1] size.
      labels: transcripts. List of length batch_size.
      seq_lens: Sequence Lengths. List of length batch_size.
    """
    nr_channels = len(speech_channels['p'+str(patientNr)])
    if eval_data == 'train':
        filelist = glob.glob(os.path.join(data_dir,
                                               'Patient' + str(patientNr)
                                               + '\\train\\*.tfrecords'))
        filenames = sorted(filelist, key=getKey)
        
    elif eval_data == 'val':
        filenames = glob.glob(os.path.join(data_dir,
                                           'Patient' + str(patientNr)
                                               + '\\valid\\*.tfrecords'))

    elif eval_data == 'test':
        filenames = glob.glob(os.path.join(data_dir,
                                           'Patient' + str(patientNr)
                                               + '\\test\\*.tfrecords'))

    for file in filenames:
        if not tf.gfile.Exists(file):
            raise ValueError('Failed to find file: ' + file)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)

    # Generate a batch of ECoG data and labels by building up a queue of examples.
    return filename_queue#read_and_decode(filename_queue, nr_channels, batch_size)
