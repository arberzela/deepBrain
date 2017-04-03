import theano
import theano.tensor as T
import numpy as np
from scipy.stats import norm
from theano import function
import _pickle as cPickle

# dictionary with the distributions
data = cPickle.load(open('C:\\Users\\user\\Desktop\\Master-Project-master\\save.pkl', 'rb'))
dict = {}
words = ['ich', 'das', 'ist', 'du', 'die']

# TensorVariables for mi, mj, si, sj respectivelly.
a, b = T.fscalars('a', 'b') # mu
c, d = T.fscalars('c', 'd') # sigma

# Energy as a TensorVariable
E = 0.5 * (d / c + (a - b) ** 2 / c - 1 - T.log(d / c))
enrg = function([a, b, c, d], E)
    
def KLdivergence(data):
    
    dictKLword = []
    distList = []

    for i in range(len(data)):
        mu, std = norm.fit(data[i])
        var = np.power(std, 2)
        distList.append([mu, var])
  
    mu_ref = np.asarray(distList[0][0], dtype='float32')
    var_ref = np.asarray(distList[0][1], dtype='float32')
    
    for mu, var in distList[1:]:
        dictKLword.append(float(enrg(mu_ref, np.asarray(mu, dtype='float32'), var_ref, np.asarray(var, dtype='float32'))))

    return dictKLword

for word in words:  
    dict[word] = KLdivergence(data[word])
'''
dist1 = data['ich'][5]
mu1, var1 = norm.fit(dist1)

for i in range(len(data['das'])):
    dist2 = data['das'][i]
    mu2, var2 = norm.fit(dist2)

    e_diff = float(enrg(np.asarray(mu1, dtype='float32'), np.asarray(mu2, dtype='float32'), np.asarray(var1, dtype='float32'), np.asarray(var2, dtype='float32')))
    print(e_diff)
'''
