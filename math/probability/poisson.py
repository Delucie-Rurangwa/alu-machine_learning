#!/usr/bin/env python3
""" defines Poisson class that represents Poisson distribution """


class Poisson:
    """
    class that represents Poisson distribution

    """

    def __init__(self, data=None, lambtha=1.):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            lambtha [float]: the expected number of occurences on a given time

        """
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                lambtha = float(sum(data) / len(data))
                self.lambtha = lambtha

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        factorial = 1
        for i in range(k):
            factorial *= (i + 1)
        pmf = ((lambtha ** k) * (e ** -lambtha)) / factorial
        return pmf

    def cdf(self, k):
        """
        calculates the value of the CDF for a given number of successes
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
