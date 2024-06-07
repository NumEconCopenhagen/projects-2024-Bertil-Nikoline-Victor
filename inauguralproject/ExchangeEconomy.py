# Import packages
from types import SimpleNamespace
import numpy as np


class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A


    def utility_A(self, x1A, x2A):
        return (x1A**self.par.alpha)*(x2A**(1 - self.par.alpha))

    def utility_B(self, x1B, x2B):
        return (x1B**self.par.beta)*(x2B**(1 - self.par.beta))

    def demand_A(self,p1):

        # The numeraire is p2 = 1
        p2 = 1
        
        # Find consumer A's demand for good 1 and 2
        x1A = self.par.alpha*((p1*self.par.w1A + p2*self.par.w2A) / p1)
        x2A = (1 - self.par.alpha)*( (p1*self.par.w1A + p2*self.par.w2A) / p2 )
        
        # Returning the demand in a tuple
        return (x1A,x2A)

    def demand_B(self,p1):

        # The numeraire is p2 = 1
        p2 = 1
        
        # Find consumer B's demand for good 1 and 2
        x1B = self.par.beta*((p1*self.par.w1B + p2*self.par.w2B) / p1)
        x2B = (1 - self.par.beta)*( (p1*self.par.w1B + p2*self.par.w2B) / p2 )
        
        # Returning the demand in a tuple
        return (x1B,x2B)

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def find_market_clearing(self):

        # 1. Defining p1 - you can adjust the price list to be more detailed
        detail_level = 75
        p1 = [(0.5 + 2*i/detail_level) for i in range(detail_level + 1)]

        # 2. Calculate the errors
        errors = [self.check_market_clearing(x) for x in p1]
        eps1 = [x[0] for x in errors]
        eps2 = [x[1] for x in errors]

        # 3. Finding the index of the value that is closest to zero in the eps1 list
        index_closest_to_zero, closest_to_zero = min(enumerate(eps1), key=lambda x: abs(x[1]))

        # 4. Getting the price where eps1 is closest to zero - the market clearing price
        price = p1[index_closest_to_zero]

        # 5. Getting the market clearing demand for consumer A
        (x1A, x2A) = self.demand_A(price)

        return (price, x1A, x2A)