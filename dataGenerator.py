import numpy as np
import json
import random
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor, wait

from utils import text_to_int_sequence, sparse_tensor_feed


class DataGenerator(object):
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

        self.batch_size = batch_size
        self.batches_per_epoch = int(np.ceil(len(unsorted_feats) / batch_size))
        
        self.sorted_durations = list(sorted_durations)
        self.sorted_feats = unsorted_feats[np.array(indices_sorted)]
        self.sorted_texts = unsorted_texts[np.array(indices_sorted)]

        # firstly split the sorted features and transcripts in self.batches_per_epoch
        # arrays. So for instance if the durations of the sorted (increasing order) features are:
        # [83, 105, 108, 254, 300] and self.batches_per_epoch = 3, the result after the split
        # would be [[83, 105], [108, 254], [300]]
        self.sorted_feats_batches = np.array_split(self.sorted_feats, self.batches_per_epoch)
        self.sorted_texts_batches = np.array_split(self.sorted_texts, self.batches_per_epoch)

        
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

    def iterate(self, features, texts, shuffle = True):
        
        if shuffle:
            feats_texts_batches = list(zip(features, texts))
            random.shuffle(feats_texts_batches)
            features[:], texts[:] = zip(*feats_texts_batches)

        pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
        
        for i in range(self.batches_per_epoch):
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.prepare_batch,
                                 features[i],
                                 texts[i])
            wait([future])
            batch = future.result()
            yield batch
            
    def iterate_train(self, epoch, sortagrad=True):
        # in order to maintain the order of the sorted arrays we
        # just give to the generator a shallow copy of the class attributes
        durations, features, texts = (self.sorted_durations.copy(),
                                      self.sorted_feats_batches.copy(),
                                      self.sorted_texts_batches.copy())
        if epoch == 0 and sortagrad:
            shuffle = False
        else:
            shuffle = True
        
        return self.iterate(features, texts, shuffle)

    def iterate_valid(self):
        # in order to maintain the order of the sorted arrays we
        # just give to the generator a shallow copy of the class attributes
        durations, features, texts = (self.sorted_durations.copy(),
                                      self.sorted_feats_batches.copy(),
                                      self.sorted_texts_batches.copy())
        
        return self.iterate(features, texts)
                
