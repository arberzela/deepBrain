import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import lasagne
from itertools import islice


LARGEST_UINT32 = 4294967295

def tokenizer(s):
    return s.strip().split()

def text_to_pairs(text,random_gen,half_window_size=2,nsample_per_word=1):

    npairs = sum([2* len(doc)*half_window_size *nsample_per_word for doc in text])
    pairs = np.empty((npairs,5),dtype=np.uint32)
    randids = random_gen(npairs)
    next_pair = 0
    for doc in text:
        cdoc = doc
        doc_len = cdoc.shape[0]
        for i in range(doc_len):
            if cdoc[i] == LARGEST_UINT32:
                continue
            for j in range(i+1,min(i+half_window_size+1,doc_len)):
                if cdoc[j] == LARGEST_UINT32:
                    continue
            for k in range(nsample_per_word):
                pairs[next_pair,0] = cdoc[i]
                pairs[next_pair,1] = cdoc[j]
                pairs[next_pair,2] = cdoc[i]
                pairs[next_pair,3] = randids[next_pair]
                pairs[next_pair,4] = 0
                next_pair += 1

                pairs[next_pair, 0] = cdoc[i]
                pairs[next_pair, 1] = cdoc[j]
                pairs[next_pair, 2] = randids[next_pair]
                pairs[next_pair, 3] = cdoc[j]
                pairs[next_pair, 4] = 1
                next_pair += 1
    return np.ascontiguousarray(pairs[:next_pair, :])
class Vocabulary(object):
    def __init__(self, tokens, tokenizer=tokenizer):
        '''
        tokens: a {'token1': 0, 'token2': 1, ...} map of token -> id
            the ids must run from 0 to n_tokens - 1 inclusive
        tokenizer: accepts a string and returns a list of strings
        TODO: Think how to transform a corpus into tokens, unique words perhaps?
        '''
        self._tokens = tokens
        self._ids = {i: token for token, i in tokens.items()}
        self._ntokens = len(tokens)
        self._tokenizer = tokenizer

    def word2id(self,word):
        if word in self._tokens:
            return self._tokens[word]
        else:
            return None

    def id2word(self,id):
        if id in self._ids:
            return self._ids[id]
        else:
            return None

    def tokenize(self, s):
        '''
        Removes OOV tokens using built
        '''
        tokens = self._tokenizer(s)
        return [token for token in tokens if token in self._tokens]

    def tokenize_ids(self, s, remove_oov=True):
        tokens = self._tokenizer(s)
        if remove_oov:
            return np.array([self.word2id(token)
                             for token in tokens if token in self._tokens],
                            dtype=np.uint32)

        else:
            ret = np.zeros(len(tokens), dtype=np.uint32)
            for k, token in enumerate(tokens):
                try:
                    ret[k] = self.word2id(token)
                except KeyError:
                    ret[k] = LARGEST_UINT32
            return ret

    def random_ids(self, num):
        return np.random.randint(0, self._ntokens, size=num).astype(np.uint32)


    def iter_pairs(self,fin,vocab,batch_size=10,nsamples=2,window=5):

        documents = iter(fin)
        batch = list(islice(documents, batch_size))
        while len(batch) > 0:
            text = [
                vocab.tokenize_ids(doc, remove_oov=False)
                for doc in batch
                ]
            pairs = text_to_pairs(text, vocab.random_ids,
                                  nsamples_per_word=nsamples,
                                  half_window_size=window)
            yield pairs
            batch = list(islice(documents, batch_size))

            


