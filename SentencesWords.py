import re

class SentenceSlots():
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

            return dictn
        

        self.words = getTimeSlots(words_list)
        self.sentences = getTimeSlots(sentc_list)

    def word_percentages(self):
        word_perc = self.sentences
        perc = dict()
        for k, v in word_perc.items():
            perc[k] = []
            s_time = k[1] - k[0]
            for word in v:
                for r, s in self.words.items():
                    if s[0] == word and r[0] >= k[0] and r[1] <= k[1]:
                        perc[k].append((r[1] - r[0])/s_time)

        return perc
