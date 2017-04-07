import operator
import os
import numpy as np
import h5py
import math
from SentencesWords import SentenceSlots as SS
from energyAndWords import WordsAndVoltages
from Sentence import Sentence
import _pickle as Cpickle
channels = ['G_A1', 'G_A2', 'G_A3', 'G_A4', 'G_A5', 'G_A6', 'G_A7', 'G_A8', 'G_B1', 'G_B2', 'G_B3', 'G_B4',
                'G_B5',
                'G_B6', 'G_B7', 'G_B8', 'G_C1', 'G_C2', 'G_C3', 'G_C4', 'G_C5', 'G_C6', 'G_C7', 'G_C8', 'G_D1',
                'G_D2',
                'G_D3', 'G_D4', 'G_D5', 'G_D6', 'G_D7', 'G_D8', 'G_E1', 'G_E2', 'G_E3', 'G_E4', 'G_E5', 'G_E6',
                'G_E7',
                'G_E8', 'G_F1', 'G_F2', 'G_F3', 'G_F4', 'G_F5', 'G_F6', 'G_F7', 'G_F8', 'G_G1', 'G_G2', 'G_G3',
                'G_G4',
                'G_G5', 'G_G6', 'G_G7', 'G_G8', 'G_H1', 'G_H2', 'G_H3', 'G_H4', 'G_H5', 'G_H6', 'G_H7', 'G_H8']

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



        path = 'C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4'
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
            #index for channels
            #for i in range(data.shape[2]):
            for j in range(data.shape[0]):
                #index for ps markers
                #for j in range(data.shape[0]):
                for i in range(len(channels)):
                    if channels[i] in dataDict:

                        dataDict[channels[i]].append(list(data[j,2047:2047+np.int(diff[j]),i]))
                    else:
                        dataDict[channels[i]] = [list(data[j,2047:2047+np.int(diff[j]),i])]




# Dictionary where the sentences start and end (tuple) is used as a key and the values are the words in that sentence.
sentenceDict = {}
# Dictionary where the sentences start and end (tuple) is used as a key and the values are the word percentages in that sentence.
wordPercentagesDict = {}
sentenceDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0049.TextGrid").sentences)
sentenceDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0070_ganz.TextGrid").sentences)
sentenceDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0142.TextGrid").sentences)
sentenceDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0167.TextGrid").sentences)
wordPercentagesDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0049.TextGrid").word_percentages())
wordPercentagesDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0070_ganz.TextGrid").word_percentages())
wordPercentagesDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0142.TextGrid").word_percentages())
wordPercentagesDict.update(SS("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\081104da_0167.TextGrid").word_percentages())

# calculate the energy
def calculateEnergy(wordVoltages):
    energy = 0
    # for every sample point
    for samplePoint in wordVoltages:
        energy = energy + math.pow(samplePoint, 2)
    energy = energy / len(wordVoltages)
    energy = np.log2(energy)
    return energy

# check if we have an EnergyAndWords object for the word.
def checkWordPresent(wordList, wordName, percentage, sentenceStart):
    # if the list is empty, just return -1 as an index, otherwise if list is not empty and we find our word, return an index
    if (len(wordList) != 0):
        counter = 0
        for wordEnergy in wordList:
            if(wordEnergy.getName() == wordName):
                if(wordEnergy.getPercentage() == percentage):
                    if(wordEnergy.getStartSentence() == sentenceStart):
                        return counter
            counter += 1

        return -1
    else:
        return -1

# List of sentences. Each sentence has the words and their corresponding percentages.
sentenceData = []
# for every key in the sorted key list, get the word ands percentages for the sentences
for keyPair in sorted(wordPercentagesDict.keys()):

    # list of words in the sentence
    nameWordsInSentence = sentenceDict.get(keyPair)
    # list of percentages of words in the sentence
    percentagesWordsInSentence = wordPercentagesDict.get(keyPair)
    # create a sentence
    singleSentence = Sentence()
    # get each word and the corresponding percentage
    for i in range(0,len(nameWordsInSentence)):
        wordName = nameWordsInSentence[i]
        wordPercentage = percentagesWordsInSentence[i]
        singleSentence.put(wordName,wordPercentage,keyPair[0])
    sentenceData.append(singleSentence)

print(len(sentenceData))
# creating the Patient object
patientP4 = Patient("P4")
# getting the neuronal data
patientData = patientP4.get_neuronal_sentences(None)
# create a list for the EnergyAndWords objects
wordsAndVoltages = []
#iterate over the channels
for key in patientData.keys():
    # get the data for one channel
    data = patientData[key]
    #print(len(sentenceData))
    # get each sentence for the channel
    for j in range(0, len(data)):
        # initial start index
        startIndex = 0
        for word, percentage, sentenceStart in sentenceData[j].get():
            # calculate the end index for the word
            endIndex = int(startIndex + len(data[j]) * percentage)
            #print(isinstance((startIndex + len(data[j]) * percentage), (int)))
            # get the neuronal data for the word
            wordVoltages = data[j][startIndex:endIndex]
            # the next word starts from the end index of the previous one
            startIndex = endIndex + 1
            # calculate the energy for the word
            #calculatedEnergy = calculateEnergy(wordVoltages)
            # check if we have already a  word in our dictionary which has the same sentence and percentage. If so get the index
            indexFinalWords = checkWordPresent(wordsAndVoltages,word,percentage, sentenceStart)
            '''
            if the index is -1 then the word is not found. Need to create the object.
            We should add the energies for the same word but for different channels to the same object.
            '''
            if(indexFinalWords != -1):
                wordsAndVoltages[indexFinalWords].put(wordVoltages)
            else:
                # create the object for the word
                singleWord = WordsAndVoltages(word,percentage,sentenceStart)
                singleWord.put(wordVoltages)
                wordsAndVoltages.append(singleWord)
print(len(wordsAndVoltages))

'''
# 3 of the most used words, ich, das, ist
tokens = ["ich","das","ist"]
for token in tokens:
    for word in wordsAndEnergies:
        if(word.getName() == token):
            print(token)
            print(word.getEnergyList())

'''

'''
# Save the distributions for each word into a dictionary, using the word as the key and a list of the distributions as a value

dictWordVoltages = {}
for word in wordsAndVoltages:
    wordName = (word.getName()).lower()
    wordsVoltagesVector = (word.getVoltageList())
    listOfMatriceVoltages = dictWordVoltages.get(wordName)
    if (listOfMatriceVoltages != None):
        listOfMatriceVoltages.append(wordsVoltagesVector)
    else:
        listOfMatriceVoltages = []
        listOfMatriceVoltages.append(wordsVoltagesVector)
    dictWordVoltages[wordName] = listOfMatriceVoltages

#print(len(dictWordEnergyVectors))
# order the dictionary from before by the highest word frequency
'''
'''
counterSamplePoints = 0
counterNegativeSamplePoints = 0
for key in sorted(dictWordVoltages, key=lambda k: len(dictWordVoltages[k]), reverse = True):
    print(key)
    for i in range(len(dictWordVoltages.get(key))):
        print(len(dictWordVoltages.get(key)[i]))
        for j in range(len(dictWordVoltages.get(key)[i])):
            for k in range(len(dictWordVoltages.get(key)[i][j])):
                counterSamplePoints +=1
                if((dictWordVoltages.get(key)[i][j])[k] < 0):
                    counterNegativeSamplePoints += 1

print(counterSamplePoints)
print(counterNegativeSamplePoints)

'''
with open("wordObjects.pickle","wb") as file:
    Cpickle.dump(wordsAndVoltages, file)







