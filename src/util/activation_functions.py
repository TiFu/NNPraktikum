# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide
from numpy import sum
from numpy import round
class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        return 1 / (1 + exp(-netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return netOutput * (1 - netOutput)

    @staticmethod
    def tanh(netOutput):
        return divide(exp(2*netOutput) - 1, exp(2*netOutput) + 1)

    @staticmethod
    def tanhPrime(netOutput):
        return 1 - Activation.tanh(netOutput) * Activation.tanh(netOutput)

    @staticmethod
    def rectified(netOutput):
        # TODO: lambda x: max(0.0, x) was wrong?
        return max(0.0, netOutput)

    @staticmethod
    def rectifiedPrime(netOutput):
        if netOutput > 0:
            return 1
        else:
            return 0

    @staticmethod
    def identity(netOutput):
        # TODO: was this wrong? (lambda x: x)
        return netOutput

    @staticmethod
    def identityPrime(netOutput):
        return 1

    @staticmethod
    def softmax(netOutput):
        z_exp = exp(netOutput)
        z_sum = sum(z_exp)
        softmax = divide(round(netOutput), z_sum)
        # Here you have to code the softmax function
        return softmax

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
