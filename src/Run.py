#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from report.evaluator import Evaluator

import matplotlib.pyplot as plt


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)

    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.1,
                                        epochs=10)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nTraining the Perceptron..")
    myPerceptronClassifier.train()
    print("Done..")


    # Do the recognizer
    perceptronPred = myPerceptronClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the Perceptron recognizer:")
    #evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)

if __name__ == '__main__':
    main()
