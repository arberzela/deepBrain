from SentencesWords import SentenceSlots as SS

inst = SS('081104da_0049.TextGrid')
words = inst.words
sentc = inst.sentences

word_perc = sentc
for k, v in word_perc.items():
    s_time = k[1] - k[0]
    for word in v:
        for r, s in words.items():
            if s[0] == word and r[0] >= k[0] and r[1] <= k[1]:
                word_perc[k][v.index(word)] = (r[1] - r[0])/s_time

print(word_perc)
