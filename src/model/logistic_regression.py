# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import util.loss_functions as lf
from util.activation_functions import Activation
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

        # Model
        self.logistic_layer=LogisticLayer(nIn=self.trainingSet.input.shape[1],
                                        nOut=10,
                                        activation='softmax',
                                        isClassifierLayer=True)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        loss = lf.CrossEntropyError()
        #Evaluator
        evaluator = Evaluator()

        epoch = 0
        learned = False
        while not learned:
            totalError = 0
            trainingSet = zip(self.trainingSet.input, self.trainingSet.label)
            for input, label in trainingSet:

                output = self.logistic_layer.forward(input)
                label = self.encodeLabel(label)
                grad = loss.calculateDerivative(label, output)
                self.updateWeights(self.learningRate*grad)

                totalError += loss.calculateError(label, output).sum()

            np.random.shuffle(trainingSet)

            if verbose:
                logging.info("Epoch: %i; error: %.2f", epoch, totalError)
                evaluator.printAccuracy(self.validationSet,
                                        self.evaluate(self.validationSet))

            epoch += 1
            learned = (epoch > self.epochs) or totalError == 0



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
        return np.argmax(self.fire(testInstance))

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
        self.logistic_layer.computeDerivative(grad)
        self.logistic_layer.updateWeights()

    def encodeLabel(self, label):
        encodedLabel = np.zeros(self.logistic_layer.nOut)
        encodedLabel[label] = 1
        return encodedLabel

    def fire(self, input):
        return self.logistic_layer.fire(input)
