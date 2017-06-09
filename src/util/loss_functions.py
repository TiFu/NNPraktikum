# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'absolute'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'different'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return target - output


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'mse'

    def calculateError(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        nOut = len(output[0])
        n = len(target)
        meanSquaredErrors = np.zeros(nOut)
        for o in range(0, nOut):
            s = 0
            for ti in target:
                s += ti - output[ti][o]
            meanSquaredErrors[o] = 1./n*(s**2)
        return meanSquaredErrors

    def calculateDerivative(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        nOut = len(output[0])
        n = len(target)
        meanSquaredDerivatives = np.zeros(nOut)
        for o in range(0, nOut):
            for ti in target:
                meanSquaredDerivatives[o] += -0.5*n*(ti - output[ti][o])
        return meanSquaredDerivatives


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'sse'

    def calculateError(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        nOut = len(output[0])
        sumSquaredErrors = np.zeros(nOut)
        for o in range(0, nOut):
            s = 0
            for ti in target:
                s += ti - output[ti][o]
            sumSquaredErrors[o] = 0.5*s**2
        return sumSquaredErrors

    def calculateDerivative(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        nOut = len(output[0])
        sumSquaredDerivatives = np.zeros(nOut)
        for o in range(0, nOut):
            for ti in target:
                sumSquaredDerivatives[o] += -(ti - output[ti][o])
        return sumSquaredDerivatives


class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        return -sum(target*np.log(output) + (1-target)*np.log(1-output))


class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        pass
