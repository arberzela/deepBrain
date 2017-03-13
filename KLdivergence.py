from Word2Gauss import *
import _pickle as cPickle

# dictionary with the distributions
data = _pickle.load(open('C:\\Users\\user\\Desktop\\Master-Project-master\\save.pkl', 'rb'))

for key in data:  
    data[key] = GaussianDistribution(data[key])

def KLdivergence(word, data, dist):
    '''
    dist1, dist2 lists [mu, sigma] respectively
    '''
    # TensorVariables for mi, mj, si, sj respectivelly.
    a, b = T.fscalars('a', 'b') # mu
    c, d = T.fscalars('c', 'd') # sigma
    dictKLword = []

    # Energy as a TensorVariable
    E = -0.5 * (d / c + (a - b) ** 2 / c - 1 - T.log(d / c))
    enrg = function([a, b, c, d], E)

    dist_data = data[word].Gaussian
    word_id = dictwrds[word]
    mu_w2g = dist.mu[word_id]
    sigma_w2g = dist.Sigma[word_id]
    
    for mu, sigma in dist_data:
        dictKLword.append(float(enrg(mu_w2g, mu, float(sigma_w2g), float(sigma)))

    return dictKLword

def dictKLdiv(data, dist):

    dictn = {}
    for word in data:
        dictn[word] = KLdivergence(word, data, dist)

    return dictn
