import os
import numpy as np
import h5py
import scipy.io as sio
from six.moves import cPickle


def get_channels(patient_path):
    for file in os.listdir(patient_path):
        if file.endswith('.hdr.mat'):
            header = sio.loadmat(patient_path + '\\' + file)
        
    header = header['H'][0][0][5][0]
    channels = list()
    for ch in header:
        channels.append(str(ch[0]))
        
    return channels

class Patient(object):

    def __init__(self, patient):

        dataDict = dict()
        self.name = patient

        path = 'C:\\Users\\user\\Desktop\\Master Project\\Bisherige Daten\\' + self.name + '\\corrected_header'
        channels = get_channels(path)
        os.chdir(path)
        
        for file in os.listdir(path):
            if file.startswith('ps'):
                dataList = list()
                f = h5py.File(path + '\\' + file, 'r')
                data = np.array(f.get('data'))
                diff = np.array(f.get('diff')).flatten()
                assert(len(channels) == data.shape[2])
                #index for ps markers
                for ps in range(data.shape[0]):
                    dataList.append(data[ps, 2048:2048 + np.int(diff[ps]) + 1, :]) # add 1 to diff
                dataDict[f.filename] = dataList
                
        for i in list(dataDict.keys()):
            for j in range(1, len(dataDict) + 1):
                if 'ps' + str(j) + '.mat' in i:
                    dataDict[j] = dataDict.pop(i)
        self.allData = dataDict
        self.days = len(self.allData)
        self.channels = channels

    def get_day(self, day = None):
        if day == None:
            return self.allData
        elif day > self.days:
            print('\nThis patient has less number of days recorded!!')
        else:
            return self.allData[day]
