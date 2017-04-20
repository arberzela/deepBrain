import _pickle as Cpickle
class NeuralNetwork:

    def create_network(self):
        #merr te dhenat
        with open("data.pickle", "r") as file:
            data = Cpickle.load(file)

        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]


        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        network = build_cnn(input_var)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        # Finally, launch the training loop.
        with open("networkInfo.txt", "a") as file:
            file.write("Started Training")

        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in generate_batch(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch: # Finally, launch the training loop.
            with open("networkInfo.txt", "a") as file:
                    file.write("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
                    file.write("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                    file.write("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                    file.write("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in generate_batch(X_test, y_test, 500, shuffle=False):
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
        np.savez('model.npz', *lasagne.layers.get_all_param_values(network))


    def build_cnn(input_var=None):
        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=(500, 2649, 8, 8), input_var=input_var)

        # Convolutional layer with 32 kernels of size 3x3.
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), stride=1, pad=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        # Convolutional layer with 32 kernels of size 3x3.
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), stride=1, pad=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        # pooling layer
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        # Convolutional layer with 32 kernels of size 3x3.
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), stride=1, pad=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        # Fully connected layer
        network = lasagne.layers.DenseLayer(network, num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
        # Softmax
        network = lasagne.layers.DenseLayer(network, num_units= 108, nonlinearity=lasagne.nonlinearities.softmax)

    def generate_batch(data, batch_size=BATCH_SIZE):
        train_set_x = data[0]
        train_set_y = data[1]
        assert(len(train_set_x) == len(train_set_y))
        for i in range(0, len(data[0]) - batch_size + 1, batch_size):
            yield (train_set_x[i:i + batch_size], train_set_y[i:i + batch_size])