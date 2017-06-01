import numpy as np
import os
from six.moves import cPickle
import matplotlib.pyplot as plt
from wyrm.types import Data
from wyrm import plot

os.chdir('C:\\Users\\user\\Desktop\\Master Project\\ongoing')

def plot_wyrm(sentence):
    dat = Data(sentence, [np.arange(1, sentence.shape[0] + 1), np.arange(1, sentence.shape[1] + 1)], ['time','channels'], ['ms', 'mV'])
    plot.plot_channels(dat)
    plt.show()

# faulty electrodes for each patient
faulty = {'p1': [0, 1, 46, 47, 54, 55, 62, 63, 68, 70, 71, 80, 81, 85],
          'p2': [17, 29, 34, 57, 62, 63, 64, 68],
          'p3': [],
          'p4': [3, 4, 11, 12, 16, 19, 20, 35, 42, 64, 65, 71, 111]}

def rem_faulty(pat, faulty):
    patient = pat.copy()
    for i in pat:
        day = pat[i]
        for j, sen in enumerate(day):
            patient[i][j] = np.delete(sen, faulty, axis=1)
    return patient

def save_corrected():
    for p in [1, 2, 4]:
        with open('patient' + str(p) + '.pickle', 'rb') as f:
            _patient = cPickle.load(f)
        patient = rem_faulty(_patient, faulty['p' + str(p)])
        with open('patient_' + str(p) + '.pickle', 'wb') as g:
            cPickle.dump(patient, g)
    
