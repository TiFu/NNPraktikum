# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide
import numpy as np

class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        return 1 / (1 + np.exp(-netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        # assumption: netOutput = sigmoid(originalInput)
        return netOutput * (1 - netOutput)

    @staticmethod
    def tanh(netOutput):
        return divide(exp(2*netOutput) - 1, exp(2 * netOutput) + 1)

    # we assume that netOutput = tanh(originalInput)
    @staticmethod
    def tanhPrime(netOutput):
        return 1 - netOutput * netOutput

    @staticmethod
    def rectified(netOutput):
        # Question: We send you an e-mail about this. Shouldn't this just return max(0.0, x) instead of a function?
        return lambda x: max(0.0, x)

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        if netOutput > 0:
            return 1
        else:
            return 0

    @staticmethod
    def identity(netOutput):
        # Question: see rectified
        return lambda x: x

    @staticmethod
    def identityPrime(netOutput):
        # Here you have to code the derivative of identity function
        return 1

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        e_x = exp(netOutput)
        return e_x / sum(e_x)

    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
