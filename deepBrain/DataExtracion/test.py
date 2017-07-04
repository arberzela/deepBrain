from SentencesWords import *
import os

def equality(patient):
    '''
    Function for checking if the words and sentences attributes of Sentences
    objects contain the same string objects. If not, it exposes where the this
    inequality occurs.
    '''
    
    os.chdir('C:\\Users\\user\\Desktop\\Master Project\\Bisherige Daten\\' + patient + '\\TextGrid_Segmentierung')
    for file in os.listdir():
        if file.endswith('.TextGrid'):
            try:
                day = Sentences(file)
                print(file + ': Good!')
                words = []
                sentences = []
                for i in day.words:
                    if '' in i:
                        print('     Empty string here!')
                    words += i
                for j in day.sentences:
                    sentences += j

                words = [i.lower() for i in words]
                sentences = [i.lower() for i in sentences]
                if len(words) == len(sentences):
                    print('     Equal lengths!!')
                if words == sentences:
                    print('     Equal!')
                else:
                    print('     Unequal!')
                    n = 0
                    for i,j in zip(words, sentences):
                        if i != j:
                            print('****')
                            print(i)
                            print(j)
                            print(n)
                            print('****')
                        n += 1
                        
            except AssertionError:
                print(file + ': Faulty!')
        
