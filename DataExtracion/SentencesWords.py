import re
from collections import OrderedDict
import numpy as np

def scale(x):
    m = np.asarray(x)
    return m / m.sum()

def as_strings(listObj):
    out = list()
    for i in listObj:
        out += i
    out = [i.lower() for i in out]
    return out

class TextGrid(object):
    """A class whose methods access the sentences in a .TextGrid file and \
their corresponding durations."""

    def __init__(self, filename):
        file = open(filename, 'r')
        f_str = file.read()
        file.close()

        # remove \x00 from the string
        f_str = re.sub('\x00','',f_str)
        
        # TextGrid files include three items. Here we take just the third one
        # as it is the one that contains the sentence's durations
        f_str = re.sub('intervals \[.*?\]:|\n','',f_str)
        f_str = re.sub(' +',' ',f_str)
        f_str = re.sub('\?','',f_str)
        f_str = re.sub('!','',f_str)
        f_str = re.sub(',','',f_str)
        f_str = re.sub('\t','',f_str)
        item_2_index = f_str.find('item [2]')
        item_3_index = f_str.find('item [3]')
        str_words = f_str[item_2_index:item_3_index]
        str_sentc = f_str[item_3_index:]

        # remove the irrelevant elements from the string
        def removeIrrelevant(str_of):
            str_of = re.sub('item \[\d\]: .+ size = \d{1,10} ', '', str_of)
            list_of = str_of.split(' ')

            for i in list_of:
                index = list_of.index(i)
                if i == 'text' or i == 'xmin' or i == 'xmax' or i == '=':
                    list_of[index] = -1
                elif i == '""':
                    list_of[index-6] = -1
                    list_of[index-3] = -1
                    list_of[index] = -1
                else:
                    list_of[index] = re.sub('"','',list_of[index])

            list_of = [x for x in list_of if x != -1]
            for j in list_of:
                if j == '':
                    list_of.remove(j)
            return list_of

        words_list = removeIrrelevant(str_words)
        sentc_list = removeIrrelevant(str_sentc)

        def getTimeSlots(list_of):
            dictn = {}
            xminmax = []
            w_or_s = []

            for i in range(len(list_of)):
                try:
                    float(list_of[i])
                    xminmax.append(i)
                except ValueError:
                    if '.' in list_of[i]:
                        list_of[i] = re.sub('\.','',list_of[i])
                    else:
                        continue

            xmin = []
            for j in range(0, len(xminmax)):
                if j % 2 == 0:
                    xmin.append(xminmax[j])

            xmax = [b + 1 for b in xmin]

            for x in xmin:
                try:
                    a = list_of[x+2: xmin[xmin.index(x) + 1]]
                    w_or_s.append(a)
                except IndexError:
                    w_or_s.append(list_of[x+2:])

            for k in range(len(xmin)):
                dictn[(float(list_of[xmin[k]]), float(list_of[xmax[k]]))] = w_or_s[k]

            return OrderedDict(sorted(dictn.items(), key=lambda t: t[0][0]))

        words = getTimeSlots(words_list)
        for i in words:
            if words[i] == []:
                del words[i]
            elif '' in words[i]:
                words[i].remove('')
                
        self._words = words
        self._sentences = getTimeSlots(sentc_list)
        self.file = filename

class Sentences(TextGrid):
    
    def __init__(self, filename):
        super().__init__(filename)
        
        self.sentence_slots = [slot for slot in self._sentences]
        self.word_slots = [slot for slot in self._words]
        self.sentences = [self._sentences[sen] for sen in self._sentences]
        self.words = [self._words[word] for word in self._words]
        self.nr_sentences = len(self.sentences)
        self.words_as_str = as_strings(self.words)
        assert(len(self.words) == sum([len(s) for s in self.sentences]))

    def __repr__(self):
        return self.file

    def get_sentence(self, sen_nr):
        if sen_nr > self.nr_sentences:
            raise IndexError
        else:
            return self.sentences[sen_nr - 1]

    def get_sentence_time(self, sen_nr):
        if sen_nr > self.nr_sentences:
            raise IndexError
        else:
            return self.sentence_slots[sen_nr - 1]

    def sentence_len(self, sen_nr):
        if sen_nr > self.nr_sentences:
            raise IndexError
        else:
            return len(self.sentences[sen_nr - 1])

    def all_sentence_lens(self):
        all_sentence_lens = list()
        for i in range(1, self.nr_sentences + 1):
            all_sentence_lens.append(self.sentence_len(i))
        return all_sentence_lens

    def sentence_duration(self, sen_nr):
        if sen_nr > self.nr_sentences:
            raise IndexError
        else:
            xmin = self.get_sentence_time(sen_nr)[0]
            xmax = self.get_sentence_time(sen_nr)[1]
            return xmax - xmin

    def all_sentence_durations(self):
        all_sentence_durations = list()
        for i in range(1, self.nr_sentences + 1):
            all_sentence_durations.append(self.sentence_duration(i))
        return all_sentence_durations

    def all_word_durations(self):
        _all_word_durations = list()
        for i in self.word_slots:
            _all_word_durations.append(i[1] - i[0])

        acc = 0
        all_word_durations = list()
        for i, j in enumerate(self.all_sentence_lens()):
            all_word_durations.append(_all_word_durations[acc:acc + j])
            acc += j
        assert(len(all_word_durations) == len(self.all_sentence_lens()))
        return all_word_durations

    def word_percentages(self):
        perc_list = list()
        for i, j in zip(self.all_sentence_durations(), self.scaled_word_durations()):
            perc_list.append([k/i for k in j])

        return perc_list

    # not correct
    def scaled_word_durations(self):
        scaled_durations = list()
        for i, j in zip(self.all_sentence_durations(), self.all_word_durations()):
            scaled_durations.append(i * scale(j))

        return scaled_durations
         
