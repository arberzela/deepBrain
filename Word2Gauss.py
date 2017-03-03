import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import lasagne
from itertools import islice
import re
import collections

LARGEST_UINT32 = 4294967295
DTYPE = np.float32


def tokenizer(s):
    return s.strip().split()

def create_vocabulary(corpus):

    with open(corpus, encoding='utf-8') as f:
        words = set()
        lines_passed = 0
        id = 0
        vocabulary = {}
        for line in f:
            if lines_passed >= 1000:
                break
            #get only letters
            line = re.sub('[^A-Za-zÄäÖöÜüß]+', ' ', line)
            line = line.lower()
            words.update(line.split())
            lines_passed += 1

        for word in words:
            vocabulary[word] = id
            id += 1

        return vocabulary


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
            for j in range(i+1,min(i,half_window_size+1,doc_len)):
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
    def __init__(self, corpus, tokenizer=tokenizer):
        '''
        tokens: a {'token1': 0, 'token2': 1, ...} map of token -> id
            the ids must run from 0 to n_tokens - 1 inclusive
        tokenizer: accepts a string and returns a list of strings
        TODO: Think how to transform a corpus into tokens, unique words perhaps?

    '''


        self._tokens = create_vocabulary(corpus)
        self._ids = {i: token for token, i in self._tokens.items()}
        self._ntokens = len(self._tokens)
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


class GaussianDistribution(object):

    def __init__(self,N,size=100,mu0=0.1,sigma_mean0=10,sigma_std0=1.0,sigma_min=0.1,sigma_max=10):

        self.N = N
        self.K = size

        #Parameter initialization

        #mu = random normal with std mu0,mean 0
        self.mu = mu0 * np.random.randn(self.N,self.K).astype(DTYPE)

        #Sigma = random normal with mean sigma_mean0, std sigma_std0, and min/max of sigma_min, sigma_max
        self.Sigma = np.random.randn(self.N,1).astype(DTYPE)
        self.Sigma *= sigma_std0
        self.Sigma += sigma_mean0
        self.Sigma = np.maximum(sigma_min,np.minimum(self.Sigma,sigma_max))

class GaussianEmbedding(object):

    def __init__(self, N, size=100, covariance_type='spherical',
                 energy='KL', C=1.0, m=0.1, M=10.0, Closs=1.0, eta=1.0):

        self.dist = GaussianDistribution(N,size,0.1,M,1.0,m,M)
        self.eta = eta

        self._acc_grad_mu = theano.shared(np.zeros(N))
        self._acc_grad_sigma = theano.shared(np.zeros(N))
        self.C = C
        self.m = m
        self.M = M
        self.Closs = Closs
        self.grad_mu = theano.shared(np.zeros_like(self.dist.mu))
        self.grad_sigma = theano.shared(np.zeros_like(self.dist.Sigma))



    def energy(self,i,j):
        mu_i = self.dist.mu[i]
        mu_j = self.dist.mu[j]
        Sigma_i = self.dist.Sigma[i]
        Sigma_j = self.dist.Sigma[j]

        det_fac = self.dist.K * np.log(Sigma_j/Sigma_i)
        trace_fac = self.dist.K * Sigma_j / Sigma_i

        return -0.5 * float(
            trace_fac * np.sum((mu_i - mu_j)**2 /Sigma_i) - self.dist.K - det_fac
        )

    def loss(self,pos,neg):
        return max(0.0,
                   self.Closs - self.energy.energy(*pos) + self.energy.energy(*neg)
                   )

     def update(self,gradients,params,eta,fac,k):
        
        #accumulate mu
        val = self.acc_grad_mu[k].get_value()
        val += np.sum(gradients[:-1]**2)/len(gradients[:-1])
        self.acc_mu_grad_mu[k].set_value(val)

        #accumulate sigma
        val = self._acc_grad_sigma.get_value()
        val += gradients[-1]**2
        self._acc_grad_sigma.set_value(val)
        

        #updates
        #update mu
        val = self.grad_mu[k].get_value()
        eta_mu = eta/np.sqrt(val+1.0)
        val -= fac*eta_mu*gradients[:-1]
        self.grad_mu[k].set_value(val)
        l2_mu = np.sqrt(np.sum(self.grad_mu[k].get_value()**2))
        if l2_mu > self.C:
            val = self.grad_mu[k].get_value()
            val *= (self.C/l2_mu)
            self.grad_mu[k].set_value(val)

        #update sigma
        val = self.grad_sigma[k].get_value()
        eta_sigma = eta / np.sqrt(val + 1.0)
        val -= fac*eta*gradients[-1]
        self.grad_sigma[k].set_value(val)
        self.grad_sigma[k].set_value(np.maximum(self.m,np.minimum(self.M,val)))
        

class KLdiv(object):
    '''
    Negative KL-div
    pair: 2 x k+1 matrix
    We only consider spherical covariance matrices
    '''
    
    def __init__(self, pair):
        
        self.pair = np.float32(pair)
        # dimensions of the multivariate Gaussian distribution
        self.k = len(self.pair[0]) - 1
        
        self.mi = self.pair[0][:self.k]
        self.mj = self.pair[1][:self.k]
        self.si = self.pair[0][self.k]
        self.sj = self.pair[1][self.k]

        # TensorVariables for mi, mj, si, sj respectivelly.
        self.a = T.fvector('a')
        self.b = T.fvector('b')
        self.c = T.fscalar('c')
        self.d = T.fscalar('d')

        # Energy as a TensorVariable
        self.E = -0.5 * (self.k * self.d / self.c + T.sum((self.a - self.b) ** 2 / self.c) - self.k - self.k * T.log(self.d / self.c))

    def energy(self):
        enrg = function([self.a, self.b, self.c, self.d], self.E)
        return float(enrg(self.mi, self.mj, self.si, self.sj))

    def gradient(self):
        grad = np.zeros(np.shape(self.pair))
        
        g1 = T.grad(self.E, self.a) # dE/dmi
        f1 = function([self.a, self.b, self.c, self.d], g1)
        
        g2 = T.grad(self.E, self.b) # dE/dmj
        f2 = function([self.a, self.b, self.c, self.d], g2)
        
        g3 = T.grad(self.E, self.c) # dE/dsi
        f3 = function([self.a, self.b, self.c, self.d], g3)
        
        g4 = T.grad(self.E, self.d) # dE/dsj
        f4 = function([self.a, self.b, self.c, self.d], g4)

        grad[0][:-1] = f1(self.mi, self.mj, self.si, self.sj)
        grad[1][:-1] = f2(self.mi, self.mj, self.si, self.sj)
        grad[0,-1] = f3(self.mi, self.mj, self.si, self.sj)
        grad[1,-1] = f4(self.mi, self.mj, self.si, self.sj)
        
        return grad        
 
