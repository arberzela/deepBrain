import theano
import theano.tensor as T
import numpy as np
from scipy.stats import norm
from theano import pp
from theano import function
from Vocabulary import *

LARGEST_UINT32 = 4294967295
DTYPE = np.float32

class GaussianDistribution(object):
    def __init__(self, N=None, size=1, mu0=0.1, sigma_mean0=10, sigma_std0=1.0, sigma_min=0.1, sigma_max=10,data=None):

        self.N = N
        self.K = size

        # Parameter initialization
        #random init
        if data is None:


            # mu = random normal with std mu0,mean 0
            self.mu = mu0 * np.random.randn(self.N, self.K).astype(DTYPE)

            # Sigma = random normal with mean sigma_mean0, std sigma_std0, and min/max of sigma_min, sigma_max
            self.Sigma = np.random.randn(self.N, 1).astype(DTYPE)
            self.Sigma *= sigma_std0
            self.Sigma += sigma_mean0
            self.Sigma = np.maximum(sigma_min, np.minimum(self.Sigma, sigma_max))
            self.Gaussian = np.concatenate((self.mu, self.Sigma), axis=1)

            # TensorVariables for mi, mj, si, sj respectivelly.
            a, b = T.fvectors('a', 'b')
            c, d = T.fscalars('c', 'd')
            
            # Energy as a TensorVariable
            E = -0.5 * (self.K * d / c + T.sum((a - b) ** 2 / c) - self.K - self.K * T.log(d / c))
            self.enrg = function([a, b, c, d], E)

            g1 = T.grad(E, a) # dE/dmi
            self.f1 = function([a, b, c, d], g1)
        
            g2 = T.grad(E, b) # dE/dmj
            self.f2 = function([a, b, c, d], g2)
        
            g3 = T.grad(E, c) # dE/dsi
            self.f3 = function([a, b, c, d], g3)
        
            g4 = T.grad(E, d) # dE/dsj
            self.f4 = function([a, b, c, d], g4)
            
        #non random init
        else:
            self.mu=[]
            self.Sigma = []


            for i in range(len(data)):
                mu, std = norm.fit(data[i])
                var = np.power(std, 2)
                self.mu.append(mu)
                self.Sigma.append(var)

            self.Gaussian = np.concatenate((np.asarray(self.mu), np.asarray(self.Sigma)), axis=1)
            self.Gaussian = np.reshape(self.Gaussian,(2,N)).T

    def energy(self, i, j):
        return float(self.enrg(self.mu[i], self.mu[j], float(self.Sigma[i]), float(self.Sigma[j])))

    def gradient(self, i, j):
        grad = np.empty((2, self.K + 1), dtype=DTYPE)

        grad[0][:-1] = self.f1(self.mu[i], self.mu[j], float(self.Sigma[i]), float(self.Sigma[j]))
        grad[1][:-1] = self.f2(self.mu[i], self.mu[j], float(self.Sigma[i]), float(self.Sigma[j]))
        grad[0,-1] = self.f3(self.mu[i], self.mu[j], float(self.Sigma[i]), float(self.Sigma[j]))
        grad[1,-1] = self.f4(self.mu[i], self.mu[j], float(self.Sigma[i]), float(self.Sigma[j]))
        
        return grad

        
class GaussianEmbedding(object):
    def __init__(self, N, size=1, covariance_type='spherical',
                 energy='KL', C=1.0, m=0.1, M=10.0, Closs=1.0, eta=1.0):
        self.dist = GaussianDistribution(N, size, 0.1, M, 1.0, m, M)
        self.eta = eta

        self._acc_grad_mu = np.zeros(N)
        self._acc_grad_sigma = np.zeros(N)
        self.C = C
        self.m = m
        self.M = M
        self.Closs = Closs
        self.grad_mu = theano.shared(np.zeros_like(self.dist.mu))
        self.grad_sigma = theano.shared(np.zeros_like(self.dist.Sigma))

    #def loss(self, pos, neg):
     #   return max(0.0,
      #             self.Closs - self.energy.energy(*pos) + self.energy.energy(*neg)
       #            )
    def loss(self,posEng,negEng):
        return max(
            0.0,
            self.Closs - posEng + negEng
        )

    def train(self, batch):
        # pairs : (i_pos,j_pos) (i_neg,j_neg). comes from text_to_pairs
        posFac = -1.0
        negFac = 1.0
        for pair in batch:
            #print(pair)
            posi = pair[0]
            posj = pair[1]
            negi = pair[2]
            negj = pair[3]

            # if loss for this case is 0, there's nothing to update
            if self.loss(self.dist.energy(posi,posj), self.dist.energy(negi,negj)) < 1e-14:
                continue

            # update positive samples
            posGrad = self.dist.gradient(posi,posj)
            self.update(posGrad[0], self.eta, posFac, posi)
            self.update(posGrad[1], self.eta, posFac, posj)

            # update negative samples
            negGrad = self.dist.gradient(negi,negj)
            self.update(negGrad[0], self.eta, negFac, negi)
            self.update(negGrad[1], self.eta, negFac, negj)



    def update(self, gradients, eta, fac, k):
        global updates1
        global updates2
        global updates3
        global updates4
        # accumulate mu
        val = self._acc_grad_mu[k]
        val += np.sum(gradients[:-1] ** 2) / len(gradients[:-1])
        self._acc_grad_mu[k] = val


        # accumulate sigma
        val = self._acc_grad_sigma[k]
        val += gradients[-1] ** 2
        self._acc_grad_sigma[k] = val


        # updates
        # update mu
        val = self.grad_mu[k]

        eta_mu = eta / np.sqrt(self._acc_grad_mu[k] + 1.0)
        updates1 = (self.grad_mu, T.set_subtensor(val, val - (fac * eta_mu * gradients[:-1])))
        #updateFunc1 = function([], updates=[updates1])
        update1Mu()

        # regularization
        new_val = self.grad_mu[k]

        l2_mu = np.sqrt(np.sum(new_val.eval() ** 2))
        if l2_mu > self.C:
            updates2 = (self.grad_mu, T.set_subtensor(new_val, new_val * (self.C / l2_mu)))
            #updateFunc2 = function([], updates=[updates2])
            update2Mu()
        self.dist.mu[k] = self.grad_mu[k].eval()



        # update sigma
        val = self.grad_sigma[k]
        eta_sigma = eta / np.sqrt(self._acc_grad_sigma[k] + 1.0)
        updates3 = (self.grad_sigma, T.set_subtensor(val, val - (fac * eta_sigma * gradients[-1])))
        #updateFunc1 = function([], updates=[updates1])
        update1Sigma()
        # regularization
        new_val = self.grad_sigma[k]
        updates4 = (self.grad_sigma, T.set_subtensor(new_val, T.maximum(
            float(self.m), T.minimum(float(self.M), float(new_val.eval()))
        )))
        #updateFunc2 = function([], updates=[updates2])
        update2Sigma()

        self.dist.Sigma[k] = self.grad_sigma[k].eval()


       # val = self.grad_sigma[k].get_value()
       # eta_sigma = eta / np.sqrt(val + 1.0)
       # val -= fac * eta * gradients[-1]
       # self.grad_sigma[k].set_value(val)
       # self.grad_sigma[k].set_value(np.maximum(self.m, np.minimum(self.M, val)))


#pair = np.array([[1, 5, 3, 5, 2], [5, 5, 4, 3, 3]])
'''
with open('train_data.pkl', 'rb') as f:
    train_data = cPickle.load(f)
with open('dict.pkl', 'rb') as g:
    dict = cPickle.load(g)

print('Data Loaded')
vocab_size = len(dict)
vocab = Vocabulary(ntokens=vocab_size)
'''
embeddings = GaussianEmbedding(vocab_size)

print('Train Started')
batch_pairs = vocab.iter_pairs(train_data)
updates1 = (embeddings.grad_mu, embeddings.grad_mu)
updates2 = (embeddings.grad_mu, embeddings.grad_mu)
updates3 = (embeddings.grad_sigma, embeddings.grad_sigma)
updates4 = (embeddings.grad_sigma, embeddings.grad_sigma)
update1Mu = function([], updates=[updates1])
update2Mu = function([], updates=[updates2])
update1Sigma = function([], updates=[updates3])
update2Sigma = function([], updates=[updates4])

while True:
    try:
        embeddings.train(next(batch_pairs))
    except StopIteration:
        break

print('Train Finished')

with open('distrib.pkl','wb') as f:
    cPickle.dump(g.dist, f)

