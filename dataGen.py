import numpy as np
import json
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor, wait

from utils import text_to_int_sequence, sparse_tensor_feed


class TrainDataGenerator(object):
    def __init__(self, json_file, batch_size):
        '''
        Read the json file that holds the data and separate the data into batches
        sorted by the duration of each utterance.
        :json_file: str, path to the .json file containing the data
        :batch_size: int, number of utterances contained in a batch
        '''
        unsorted_feats, unsorted_durations, unsorted_texts = [], [], []
        with open(json_file) as json_line_file:
            for json_line in json_line_file:
                example = json.loads(json_line)
                unsorted_feats.append(np.array(example['feature'], dtype=np.float32))
                unsorted_durations.append(float(example['duration']))
                unsorted_texts.append(example['text'])
        
        unsorted_feats = np.array(unsorted_feats)
        unsorted_texts = np.array(unsorted_texts)
        indices_sorted, sorted_durations = zip(*sorted(enumerate(unsorted_durations), key=itemgetter(1)))
        
        self.sorted_durations = list(sorted_durations)
        self.sorted_feats = unsorted_feats[np.array(indices_sorted)]
        self.sorted_texts = unsorted_texts[np.array(indices_sorted)]
        self.batch_size = batch_size
        self.batches_per_epoch = int(np.ceil(len(unsorted_feats) / batch_size))

        
    def prepare_batch(self, features, texts):
        """ Featurize a minibatch of data, zero pad them and return a dictionary
        Params:
            features (list(np.array)): List of ECoG data
            texts (list(str)): List of texts corresponding to the features
        Returns:
            dict: See below for contents
        """
        assert len(features) == len(texts),\
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        nr_channels = features[0].shape[1]
        
        # This may differ for the last batch (may be smaller)
        batch_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((batch_size, max_length, nr_channels))
        for i in range(batch_size):
            feat = features[i]
            #feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat
        
        y = text_to_int_sequence(texts)
        sparse_y = sparse_tensor_feed(y)
            
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'y': y,  # list(int) Labels (integer sequences)
            'sparse_y': sparse_y, # A tuple with (indices, values, shape)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
        }    
        
    
    def iterate(self, features, texts):
        pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
        future = pool.submit(self.prepare_batch,
                             features[:self.batch_size],
                             texts[:self.batch_size])
        start = self.batch_size
        for i in range(self.batches_per_epoch - 1):
            wait([future])
            batch = future.result()
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.prepare_batch,
                                 features[start: start + self.batch_size],
                                 texts[start: start + self.batch_size])
            yield batch
            start += self.batch_size
        # Wait on the last minibatch
        wait([future])
        batch = future.result()
        yield batch
        
    
    def iterate_train(self):
        durations, features, texts = (self.sorted_durations,
                                      self.sorted_feats,
                                      self.sorted_texts)
        
        return self.iterate(features, texts)

                
