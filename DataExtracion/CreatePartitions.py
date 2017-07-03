import os
import numpy as np
from six.moves import cPickle
from Alignment import loadAlignedData

PATH = 'C:\\Users\\user\\Desktop\\Master Project\\ongoing\\'

def split_data(patientNr, train_percent=0.8, shuffle=False):
    '''
    Split the data for each patient into training, validation and test sets
    and save them in separate directories.

    :patientNr: int
                Patient number.
    :train_percent: float
                    Set how much of the dataset to use for train.
                    The rest is split in half between the validation and test set.
    :shuffle: bool
              Shuffle or not the data before partitioning 
    '''
    #TODO: shuffle

    os.chdir(PATH)
    valid_percent = (1 - train_percent) / 2
    test_percent = valid_percent
    all_data = loadAlignedData(patientNr)
    
    def write_data(partition):
        if not os.path.isdir(partition):
            os.mkdir(partition)
        os.chdir(PATH + partition)
        if not os.path.isfile(partition + str(patientNr) + '.pickle'):
            if partition == 'train':
                data = all_data[:int(train_percent * len(all_data))]
            elif partition == 'valid':
                data = all_data[int(train_percent * len(all_data)):int((train_percent + valid_percent) * len(all_data))]
            elif partition == 'test':
                data = all_data[int((train_percent + valid_percent) * len(all_data)):]
            with open(partition + str(patientNr) + '.pickle', 'wb') as f:
                cPickle.dump(data, f)
        os.chdir(PATH)

    for partition in ['train', 'valid', 'test']:
        write_data(partition)

if __name__ == '__main__':
    for patientNr in range(1,5):
        split_data(patientNr)
