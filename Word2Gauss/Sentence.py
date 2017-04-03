# contains the words and the percentages in a sentence
class Sentence:

    # words and their percentages as a tuple
    '''for each sentence we have a list, in the list we append a tuple.
    The word, the percentage of the word and the start index for the sentence to which the word belongs to.'''
    def __init__(self):
        self.dataInSentence = []

    def put(self, wordName, wordPercentage, sentenceStart):
        self.dataInSentence.append((wordName, wordPercentage, sentenceStart))

    def get(self):
        return self.dataInSentence


