import numpy as np
from six.moves import cPickle as pickle

# batch size should be a factor of the training set length e.g. BATCH_SIZE = len(train_set) / 5
BATCH_SIZE = 500

max_length = 0
word_dictionary={}
reshape_dictionary={}
path='wordObjects.pickle'
with open(path,'rb') as file:
    words = pickle.load(file)

print('Finished pickle...')
for word in words:
    name = word.getName().lower()
    volts = word.getVoltageList()
    length = len(volts[0])
    if length>max_length:
        max_length = length
    if name in word_dictionary:
        word_dictionary[name].append(volts)
    else:
        word_dictionary[name]=[volts]

for element in word_dictionary:
    reshape_dictionary[element]=[]
    for matrix in word_dictionary[element]:
        matrix = np.asarray(matrix)
        matrix[3]=matrix[4]=matrix[11]=matrix[12]=matrix[19]=matrix[20]=matrix[16]=matrix[35]=matrix[42]=0
        matrix = np.reshape(matrix,(8,8,matrix.shape[1]))
        resMat = np.zeros((matrix.shape[0],matrix.shape[1],max_length))
        resMat[:matrix.shape[0],:matrix.shape[1],:matrix.shape[2]]=matrix
        reshape_dictionary[element].append(resMat)


def separate_data(dict_words):
    train_set_x = []
    train_set_y = []
    valid_set_x = []
    valid_set_y = []
    test_set_x = []
    test_set_y = []
    
    for word in dict_words:
        if len(dict_words[word]) >= 5:
            for i in range(len(dict_words[word][0:round((3/5) * (len(dict_words[word])))])):
                train_set_x.append(dict_words[word][i])
                train_set_y.append(word)
            for i in range(len(dict_words[word][round((3/5) * (len(dict_words[word]))):round((4/5) * (len(dict_words[word])))])):
                valid_set_x.append(dict_words[word][i])
                valid_set_y.append(word)
            for i in range(len(dict_words[word][round((4/5) * (len(dict_words[word]))):len(dict_words[word])])):
                test_set_x.append(dict_words[word][i])
                test_set_y.append(word)

    return (np.asarray(train_set_x), np.asarray(train_set_y)), (np.asarray(valid_set_x), np.asarray(valid_set_y)), (np.asarray(test_set_x), np.asarray(test_set_y))

def generate_batch(data, batch_size=BATCH_SIZE):
    train_set_x = data[0]
    train_set_y = data[1]
    for i in range(0, len(data[0]) - batch_size + 1, batch_size):
        yield (train_set_x[i:i+batch_size], train_set_y[i:i+batch_size])

