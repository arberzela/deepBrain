import numpy as np
import os
from six.moves import cPickle
import matplotlib.pyplot as plt
from wyrm.types import Data
from wyrm import plot
from ModDataManipulation import get_channels

os.chdir('C:\\Users\\user\\Desktop\\Master Project\\ongoing')

def plot_wyrm(sentence):
    dat = Data(sentence, [np.arange(1, sentence.shape[0] + 1), np.arange(1, sentence.shape[1] + 1)], ['time','channels'], ['ms', 'mV'])
    plot.plot_channels(dat)
    plt.show()

def common_average_reference(patient, patientNr):
    channels = get_channels('C:\\Users\\user\\Desktop\\Master Project\\Bisherige Daten\\P' + str(patientNr) + '\\corrected_header')
    if patientNr == 1:
        index_grid = channels.index('CAR_grid')
    elif patientNr == 4:
        index_grid = channels.index('CARgrid')
    if 'CAR_IH' in channels:
        index_IH = channels.index('CAR_IH')
    try:
        for i in patient:
            day = patient[i]
            for j, sen in enumerate(day):
                patient[i][j][:,:68] = (patient[i][j][:,:68].T - patient[i][j][:,index_grid]).T
                patient[i][j][:,channels.index('IHA1'):channels.index('IHD4')+1] = (patient[i][j][:,channels.index('IHA1'):channels.index('IHD4')+1].T - patient[i][j][:,index_IH]).T
    except NameError:
        pass
    
    return patient

            
# speech related bipolar electrodes (faulty channels removed)
speech_channels = {'p1': np.array([5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42]) - 1,
          'p2': np.array([3, 4, 11, 12, 17, 19, 20, 25, 26, 27, 28, 33, 34, 36, 41, 42, 43, 44, 45, 46, 47, 48, 55, 56, 57, 58]) - 1,
          'p3': np.array([21, 22, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 48, 49]) - 1, #for patient 3 not sure which channels are faulty
          'p4': np.array([19, 27, 28, 29, 30, 35, 37, 38, 44, 45, 46, 51, 52, 53, 54, 81, 82]) - 1}

def keep_only_speech_related(patient, speech_ch):
    _patient = patient.copy()
    for i in patient:
        day = patient[i]
        for j, sen in enumerate(day):
            _patient[i][j] = patient[i][j][:,speech_ch]
    return _patient

def save_corrected(CAR = True, only_speech = True):
    for p in range(1,5):
        with open('patient' + str(p) + '.pickle', 'rb') as f:
            patient = cPickle.load(f)
            
        if CAR:
            patient = common_average_reference(patient, p)
        if only_speech:
            patient = keep_only_speech_related(patient, speech_channels['p'+str(p)])
        
        with open('patient_' + str(p) + '.pickle', 'wb') as g:
            cPickle.dump(patient, g)
    
