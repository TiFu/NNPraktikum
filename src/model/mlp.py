
import numpy as np

from util.loss_functions import BinaryCrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        #self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        #Number of units in each Layer
        #In this Instance  we consider units 128 for the input layer
        #and 10 for the output layer.
        #another example could be : units = [128,128,64,10]
        units = [128,10]
        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], units[0],
                           None, inputActivation, False))

        #Hidden layers
        for i in range (1, len(units)-1):
            layerActivation = "sigmoid"
            self.layers.append(LogisticLayer(units[i-1]-1, units[i],
                           None, layerActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(units[-2]-1, units[-1],
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """

        for l in range(0, len(self.layers)):
            inp = self._get_layer(l).forward(inp)
        return inp

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        self._get_output_layer().computeDerivative(
                                self.loss.calculateDerivative(
                                target,self._get_output_layer().outp), 1.0)

    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        self._get_output_layer().updateWeights(learningRate)

        for l in range(len(self.layers)-1, 0, -1):
            nextWeights = self._get_layer(l).weights
            nextDerivatives = self._get_layer(l).deltas
            self._get_layer(l-1).computeDerivative(nextWeights, nextDerivatives)
            self._get_layer(l-1).updateWeights(learningRate)


    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        for img, label in zip(self.trainingSet.input,
                              self.trainingSet.label):

            # Use LogisticLayer to do the job
            # Feed it with inputs

            # Do a forward pass to calculate the output and the error
            self._feed_forward(img)

            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            self._compute_error(self.encodeLabel(label))

            # Update weights in the online learning fashion
            self._update_weights(self.learningRate)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        return np.argmax(self._feed_forward(test_instance))


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def encodeLabel(self, label):
        encodedLabel = np.zeros(self._get_output_layer().nOut)
        encodedLabel[label] = 1
        return encodedLabel

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
