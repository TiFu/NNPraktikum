#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)

    myLogisticClassifier = LogisticRegression(data.trainingSet,
                                              data.validationSet,
                                              data.testSet,
                                              learningRate=0.005,
                                              epochs=15)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nLogistic Classifier has been training..")
    myLogisticClassifier.train()
    print("Done..")


    # Do the recognizer
    # Explicitly specify the test set to be evaluated

    logisticPred = myLogisticClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the Logistic recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, logisticPred)


if __name__ == '__main__':
    main()
