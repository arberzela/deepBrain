import os
import glob
import shutil
from six.moves import cPickle
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

PATH = 'C:\\Users\\user\\Desktop\\tfVirtEnv\\deepBrain\\data\\'
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ' "
CHAR_TO_ID = {ch: i for (i, ch) in enumerate(ALPHABET)}


def make_example(seq_len, spec_feat, labels):
    '''
    Creates a SequenceExample for a single utterance.
    This function makes a SequenceExample given the sequence length,
    brain data features and corresponding transcript.
    These sequence examples are read using tf.parse_single_sequence_example
    during training.
    
    See: https://github.com/tensorflow/tensorflow/
    blob/246a3724f5406b357aefcad561407720f5ccb5dc/
    tensorflow/python/kernel_tests/parsing_ops_test.py

    :seq_len: integer represents the sequence length in time frames.
    :spec_feat: [TxCH] matrix of brain data features.
    :labels: list of ints representing the encoded transcript.
    
    :return: Serialized sequence example.
    '''

    # Feature lists for the sequential features of the example
    feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                  for frame in spec_feat]
    feat_dict = {"feats": tf.train.FeatureList(feature=feats_list)}
    sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

    # Context features for the entire sequence
    len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
    label_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))

    context_feats = tf.train.Features(feature={"seq_len": len_feat,
                                               "labels": label_feat})

    ex = tf.train.SequenceExample(context=context_feats,
                                  feature_lists=sequence_feats)

    return ex.SerializeToString()


def read_data(filename):
    '''
    Reads the pickled data file.

    :filename: the file to read the data from.
               the filename is in the format: partition + patientNr + .pickle
    :return: features: dict containing the brain data per utterance.
             transcripts: dict of lists representing the transcripts.
             utt_len: dict of ints that hold the sequence length of each
                      utternce in time frames.
    '''

    features = {}
    transcripts = {}
    utt_len = {} # Required for sorting the utterances based on length

    with open(filename, 'rb') as file:
        data = cPickle.load(file)
        #data contains a list of tuples where each tuple has the [TxCH] ndarray as the
        #element and the sentence string as second element.
        for i, utterance in enumerate(data):
            features[i+1] = utterance[0]
            utt_len[i+1] = utterance[0].shape[0]
            transcripts[i+1] = [CHAR_TO_ID[j] for j in utterance[1]]

    return features, transcripts, utt_len
    

def create_TFRec(patientNr):
    '''
    This function generates sequence examples for each utterance given the ECoG
    data and the transcripts, and saves these records into .TFRecord files.

    :patientNr: int representing the patient number.
    '''
    
    os.chdir(PATH)
    print('Creating TFRecords for patient ' + str(patientNr) + '!')
    
    for partition in ['train', 'valid', 'test']:
        print('Processing ' + partition + 'set!')
        pickled_file = os.path.join(PATH + partition + '\\' + partition
                                    + str(patientNr) + '.pickle')
        features, transcripts, utt_len = read_data(pickled_file)
        sorted_utts = sorted(utt_len, key=utt_len.get)
        # bin into groups of 100 frames.
        max_t = int(utt_len[sorted_utts[-1]]/100)
        min_t = int(utt_len[sorted_utts[0]]/100)

        write_dir = '.\\TFRecords\\' + 'Patient'+ str(patientNr) + '\\' + partition
        if os.path.isdir(write_dir):
            shutil.rmtree(write_dir)
        path = Path(write_dir)
        path.mkdir(parents=True, exist_ok=True)

        if partition == 'train':
            # Create multiple TFRecords based on utterance length for training
            writer = {}
            count = {}
            print('Processing training files...')
            for i in range(min_t, max_t+1):
                filename = os.path.join(write_dir, 'train' + '_' + str(i)
                                        + '.tfrecords')
                writer[i] = tf.python_io.TFRecordWriter(filename)
                count[i] = 0

            for utt in tqdm(sorted_utts):
                example = make_example(utt_len[utt], features[utt].tolist(),
                                       transcripts[utt])
                index = int(utt_len[utt]/100)
                writer[index].write(example)
                count[index] += 1

            for i in range(min_t, max_t+1):
                writer[i].close()
            print('Processed '+str(len(sorted_utts))+' utterances')
            print(count, '\n')

        else:
            # Create single TFRecord for valid and test partition
            filename = os.path.join(write_dir, os.path.basename(write_dir) +
                                    '.tfrecords')
            print('Creating', filename)
            record_writer = tf.python_io.TFRecordWriter(filename)
            for utt in tqdm(sorted_utts):
                example = make_example(utt_len[utt], features[utt].tolist(),
                                       transcripts[utt])
                record_writer.write(example)
            record_writer.close()
            print('Processed '+str(len(sorted_utts))+' utterances', '\n')
            
    print('\n')

if __name__ == '__main__':
    for patientNr in range(1, 5):
        create_TFRec(patientNr)

            
