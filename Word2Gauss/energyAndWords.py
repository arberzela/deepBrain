class WordsAndVoltages:

    def __init__(self, wordName, percentageValue, sentenceStart):

        # the word
        self.wordName = wordName
        # the percentage
        self.percentage = percentageValue
        # the sentence start
        self.sentenceStart = sentenceStart
        # energies for all channels.
        self.voltageChannels = []

    # put the energy for one channel, it will be in order so no other information is added for the channel.
    def put(self,energy):
        self.voltageChannels.append(energy)

    # get the word
    def getName(self):
        return self.wordName
    # get the percentage
    def getPercentage(self):
        return self.percentage
    # get the sentence index start
    def getStartSentence(self):
        return self.sentenceStart
    # get the energy list for all channels
    def getVoltageList(self):
        return self.voltageChannels
