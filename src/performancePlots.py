from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
import numpy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    i = 10
    x = []
    y = []
    while i <= 500:
        logisticRegressionClassifier = LogisticRegression(data.trainingSet,
                                            data.validationSet,
                                            data.testSet,
                                            learningRate=0.005,
                                            epochs=i)
        # Train the classifiers
        print("=========================")
        print("Training..")
        print("\nLogisticRegression has been training..")
        logisticRegressionClassifier.train()
        print("Done..")
        logisticPred = logisticRegressionClassifier.evaluate()
        # Report the result
        print("=========================")
        accuracy = accuracy_score(data.testSet.label, logisticPred)
        print("Epoch " + str(i) + ', Accuracy: ' + str(accuracy))
        x.append(i)
        y.append(accuracy)
        i += 10 # 10 steps

    plt.plot(x, y)
    plt.ylim(0.8 ,1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Set Accuracy by Epoch (10 Epoch steps)')
    plt.savefig('../epochs.png')



if __name__ == '__main__':
    main()
