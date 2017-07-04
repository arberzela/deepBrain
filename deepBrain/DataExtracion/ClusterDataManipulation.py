import os
import numpy as np
import h5py
from six.moves import cPickle

class Patient(object):

    def __init__(self, patient):

        dataDict = dict()
        self.name = patient

        path = '/home/zelaa/arber/ongoing/' + self.name
        channels = get_channels(path)
        os.chdir(path)
        
        for file in os.listdir(path):
            if file.startswith('ps'):
                dataList = list()
                f = h5py.File(path + '/' + file, 'r')
                data = np.array(f.get('data'))
                diff = np.array(f.get('diff')).flatten()
                assert(len(channels) == data.shape[2])
                #index for ps markers
                for ps in range(data.shape[0]):
                    dataList.append(data[ps, 2048:2048 + np.int(diff[ps]) + 1, :]) # add 1 to diff
                dataDict[f.filename] = dataList
                
        for i in dataDict.keys():
            for j in range(1,len(dataDict) + 1):
                if 'ps'+str(j) in i:
                    dataDict['ps'+str(j)] = dataDict.pop(i)
        self.allData = dataDict

def save_all_data():
    for i in range(1,5):
        patient = Patient('P'+str(i))
        with open('patient'+str(i)+'.pickle', 'wb') as f:
            cPickle.dump(patient.allData, f)
        
