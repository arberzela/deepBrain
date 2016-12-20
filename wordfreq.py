import pandas as panda
import string
import operator

# the dictionairy which will keep the word frequencies, the word is the key and the value is the frequency

wordFrequenciesDict = dict()


def separateSentencesIntoWords(sentence):
    # get all the words from the sentence split with any whitespace character
    words = sentence.split()
    # remove punctuation
    for word in words:
        translation = str.maketrans("","",string.punctuation)
        refinedWord = (word.translate(translation)).lower()
        wordFrequency = wordFrequenciesDict.get(refinedWord)
        if(wordFrequency == None):
            wordFrequenciesDict[refinedWord] = 1
        else:
            wordFrequenciesDict.update({refinedWord:wordFrequency+1})



def wordsToFrequence(fileName):

    # get the excel file for the fileName string
    file = panda.ExcelFile(fileName)
    # get the sheet Tabelle1 from the excel file
    sheet = file.parse("Tabelle1")
    # get the column as a list we can loop through
    columns = list(sheet.columns.values)
    # get the column index for zugezo, zugezo is the column we want to get the text from
    indexColumn = columns.index("Hochdeutsch ohne (Vor-)Vorfeld/Nachfeld")
    # get the list of rows for the column zugezo
    rowsData = file.parse("Tabelle1", parse_cols=[indexColumn])
    # iterate through the sentences from the column
    Data = rowsData.get("Hochdeutsch ohne (Vor-)Vorfeld/Nachfeld")
    for x in range(1, len(Data)):
        separateSentencesIntoWords(Data[x])



# get the words and update the frequencies for all patient
wordsToFrequence("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P1\\linguistic_information\\P1_07_09_2016_anonymisiert.xls")
print(len(wordFrequenciesDict))
wordsToFrequence("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P2\\linguistic_information\\P2_07_09_2016_anonymisiert.xls")
print(len(wordFrequenciesDict))
wordsToFrequence("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P3\\linguistic_information\\P3_07_09_2016_anonymisiert.xls")
print(len(wordFrequenciesDict))
wordsToFrequence("C:\\Users\\Lindarx\\Desktop\\CAR_EEG\\P4\\linguistic_information\\P4_11_10_2016_anonymisiert.xls")
print(len(wordFrequenciesDict))
print(wordFrequenciesDict)

# getting a sorted version of the dictionary from the highest word frequency to the lowest
sortedWordsFrequency = sorted(wordFrequenciesDict.items(), key=operator.itemgetter(1), reverse = True)
print(sortedWordsFrequency)