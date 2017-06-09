# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from util.loss_functions import BinaryCrossEntropyError
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
from report.evaluator import Evaluator

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

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
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1] + 1)

        # Model
        self.logistic_layer=LogisticLayer(nIn=self.trainingSet.input.shape[1],
                                        nOut=1,
                                        activation='sigmoid',
                                        isClassifierLayer=True)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Loss function
        loss = BinaryCrossEntropyError()
        #Evaluator
        evaluator = Evaluator()

        for epoch in range(0, self.epochs):

            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):

                output = self.logistic_layer.forward(self.biasInput(input))
                error = loss.calculateDerivative(label, output)
                #error = -label/output + (1-label)/(1-output)
                self.updateWeights(error)

            if verbose:
                logging.info("Epoch: %i", epoch)
                evaluator.printAccuracy(self.validationSet, self.evaluate(self.validationSet))

        '''
        input = map(self.biasInput, self.trainingSet.input)
        label = np.array(self.trainingSet.label)
        # Train for some epochs
        for epoch in range(0, self.epochs):
            output = np.array(map(self.logistic_layer.forward, input))
            #error  = loss.calculateError(label, output)
            error = np.zeros(len(output[0]))
            for o in range(0, len(output[0])):
                error[o] = sum(label - output[:,o])

            self.updateWeights(error)

            if verbose:
                logging.info("Epoch: %i", epoch)
                evaluator.printAccuracy(self.validationSet, self.evaluate(self.validationSet))
        '''

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance) >= 0.5

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

    def updateWeights(self, grad):
        self.logistic_layer.computeDerivative(
                                        grad,
                                        np.ones((self.logistic_layer.nOut,
                                        self.logistic_layer.nOut)))
        self.logistic_layer.updateWeights(self.learningRate)
        self.weight = self.logistic_layer.weights



    def biasInput(self, input):
        return np.append(np.array(input), [1])

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(self.biasInput(input), self.weight[0]))
