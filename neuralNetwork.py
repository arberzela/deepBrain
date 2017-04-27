import _pickle as Cpickle
import theano
import theano.tensor as T
import lasagne
import numpy as np
import time

class NeuralNetwork:


    def build_cnn(self, nr_epochs = 10, batch_size = 100):

        # Get the data
        with open("data.pickle", "rb") as file:
            data = Cpickle.load(file)
        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]
        vocabulary = list(set(y_train))
        vocabulary_length = len(vocabulary)
        word2vec = {}

        for i in range(0, vocabulary_length):
            vec = np.zeros(vocabulary_length)
            vec[i] = 1
            word2vec[vocabulary[i]] = vec

        y_train = [word2vec[item] for item in y_train]
        y_val = [word2vec[item] for item in y_val]
        y_test = [word2vec[item] for item in y_test]


        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        neuralNetwork = self.create_network(input_var)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(neuralNetwork)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(neuralNetwork, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(neuralNetwork, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        # Finally, launch the training loop.
        with open("networkInfo.txt", "a") as file:
            file.write("Started Training")

        # We iterate over epochs:
        for epoch in range(nr_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.generate_batch(X_train, y_train, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.generate_batch(X_val, y_val, shuffle=True):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch: # Finally, launch the training loop.
            with open("networkInfo.txt", "a") as file:
                    file.write("Epoch {} of {} took {:.3f}s".format(epoch + 1, 10, time.time() - start_time))
                    file.write("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                    file.write("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                    file.write("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.generate_batch(X_test, y_test, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        with open("networkInfo.txt", "a") as file:
            file.write("Final results:")
            file.write("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            file.write("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

        # Dumping network parameters
        with open("parameters.pickle" , "w") as parameterFile:
            Cpickle.dump(lasagne.layers.get_all_param_values(neuralNetwork), parameterFile)


    def create_network(self, input_var=None, nr_filters = 32, fully_units = 256, softmax_units = 108):
        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=(2649, 8, 8), input_var=input_var)

        # Convolutional layer with 32 kernels of size 3x3.
        network = lasagne.layers.Conv2DLayer(network, num_filters= nr_filters, filter_size=(3, 3), stride=1, pad=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        # Convolutional layer with 32 kernels of size 3x3.
        network = lasagne.layers.Conv2DLayer(network, num_filters= nr_filters, filter_size=(3, 3), stride=1, pad=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        # pooling layer
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        # Convolutional layer with 32 kernels of size 3x3.
        network = lasagne.layers.Conv2DLayer(network, num_filters= nr_filters, filter_size=(3, 3), stride=1, pad=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        # Fully connected layer
        network = lasagne.layers.DenseLayer(network, num_units= fully_units, nonlinearity=lasagne.nonlinearities.rectify)
        # Softmax
        network = lasagne.layers.DenseLayer(network, num_units= softmax_units, nonlinearity=lasagne.nonlinearities.softmax)

        return network

    @classmethod
    def generate_batch(self, x, y, batch_size = 10, shuffle=False):
        assert(len(x) == len(y))
        if shuffle:
            indices = np.arange(len(x))
            np.random.shuffle(indices)
        for i in range(0, len(x) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[i:i + batch_size]
            else:
                excerpt = slice(i, i + batch_size)
            yield (x[excerpt], y[excerpt])

if __name__ == '__main__':
    network = NeuralNetwork()
    network.build_cnn()
