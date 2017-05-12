#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    perceptronClassifiers = []
    for epochs in [10,50,100,250, 500]:
        for learningRate in [0.1, 0.05, 0.01, 0.005, 0.001]:
            myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=learningRate,
                                        epochs=epochs)
            perceptronClassifiers.append(myPerceptronClassifier)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron has been training..")
    for myPerceptronClassifier in perceptronClassifiers:
        myPerceptronClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    for myPerceptronClassifier in perceptronClassifiers:
        perceptronPred = myPerceptronClassifier.evaluate()
        print("Classifier: Epochs " + str(myPerceptronClassifier.epochs) + " / Learning Rate: " + str(myPerceptronClassifier.learningRate))
        evaluator.printAccuracy(data.testSet, perceptronPred)


if __name__ == '__main__':
    main()
