import os
import numpy as np
import h5py
from SentencesWords import SentenceSlots as SS

channels = ['G_A1', 'G_A2', 'G_A3', 'G_A4', 'G_A5', 'G_A6', 'G_A7', 'G_A8', 'G_B1', 'G_B2', 'G_B3', 'G_B4',
                'G_B5',
                'G_B6', 'G_B7', 'G_B8', 'G_C1', 'G_C2', 'G_C3', 'G_C4', 'G_C5', 'G_C6', 'G_C7', 'G_C8', 'G_D1',
                'G_D2',
                'G_D3', 'G_D4', 'G_D5', 'G_D6', 'G_D7', 'G_D8', 'G_E1', 'G_E2', 'G_E3', 'G_E4', 'G_E5', 'G_E6',
                'G_E7',
                'G_E8', 'G_F1', 'G_F2', 'G_F3', 'G_F4', 'G_F5', 'G_F6', 'G_F7', 'G_F8', 'G_G1', 'G_G2', 'G_G3',
                'G_G4',
                'G_G5', 'G_G6', 'G_G7', 'G_G8', 'G_H1', 'G_H2', 'G_H3', 'G_H4', 'G_H5', 'G_H6', 'G_H7', 'G_H8',
                'FL1',
                'FL2', 'FL3', 'FL4', 'FL5', 'FL6', 'IHA1', 'IHA2', 'IHA3', 'IHA4', 'IHB1', 'IHB2', 'IHB3', 'IHB4',
                'IHC1', 'IHC2', 'IHC3', 'IHC4', 'IHD1', 'IHD2', 'IHD3', 'IHD4', 'FLA1', 'FLA2', 'FLA3', 'FLA4',
                'FLB1',
                'FLB2', 'FLB3', 'FLB4', 'FLB5', 'FLB6', 'ECG', 'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                'O1',
                'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'T1', 'T2', 'EOG', 'STCR', 'STCL',
                'CAR_grid']

class Patient:




    def get_neuronal_sentences(self,channel=None):
        if channel is None:
            return self.allData
        if type(channel) is int:
            return self.allData[channels[channel]]
        else:

            return self.allData[channel]

    def word_voltage(self):
        slots = SS()
        word_perc = slots.word_percentages()




    def __init__(self,patient):


        dataDict = dict()
        self.name = patient
        self.allData = dataDict



        path = '/media/ralvi/0A527FA0527F8F67/Project/CAR_EEG/P4'
        allData = []
        differences = []

        for file in os.listdir(path):
            if file.startswith('ps'):
                f = h5py.File(path + '/' + file, 'r')
                data = np.array(f.get('data'))
                allData.append(data)
                diff = np.array(f.get('diff'))
                differences.append(diff)


        for data,diff in zip(allData,differences):

            diff = np.array(diff).flatten()
            data = np.array(data)
            print(diff[0])
            #index for channels
            for i in range(data.shape[2]):
                #index for ps markers
                for j in range(data.shape[0]):

                    if channels[i] in dataDict:

                        dataDict[channels[i]].append([list(data[j,2047:2047+np.int(diff[j]),i])])
                    else:
                        dataDict[channels[i]] = [list(data[j,2047:2047+np.int(diff[j]),i])]












