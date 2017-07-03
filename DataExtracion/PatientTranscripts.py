from SentencesWords import *
import os
from six.moves import cPickle

class PatientTranscripts(object):
    def __init__(self, patient):
        dataDict = dict()
        self.name = patient

        path = 'C:\\Users\\user\\Desktop\\Master Project\\Bisherige Daten\\' + self.name + '\\TextGrid_Segmentierung'
        os.chdir(path)

        for day, file in enumerate(os.listdir(path)):
            if file.endswith('.TextGrid'):
                dataDict[day + 1] = Sentences(file)

        self.transcripts = dataDict

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
    def __len__(self):
        return(len(self.transcripts))
                
def get_all_transcripts():
    
    all_transcripts = list()  # list containing all PatientTranscripts objects
    for i in range(1, 5):
        all_transcripts.append(PatientTranscripts('P' + str(i)))
        
    return all_transcripts


