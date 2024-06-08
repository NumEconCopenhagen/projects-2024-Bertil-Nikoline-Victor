from types import SimpleNamespace

# To use the np.abs(x).argmin() command
import numpy as np

class ExchangeEconomyClass:

    def __init__(self):
       
        """
        Initialize an instance of the class.

        This method sets up the parameters for the model.

        Parameters:
        alpha (float): The preference parameter for good 1. Default is 1/3.
        beta (float): The preference parameter for good 2. Default is 2/3.
        w1A (float): The endowment of good 1 for consumer A. Default is 0.8.
        w2A (float): The endowment of good 2 for consumer A. Default is 0.3.
        w1B (float): The endowment of good 1 for consumer B. It's calculated as the total endowment of good 1 (assumed to be 1) minus w1A.
        w2B (float): The endowment of good 2 for consumer B. It's calculated as the total endowment of good 2 (assumed to be 1) minus w2A.
        """

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
        """
        This method calculates the utility of consumer A for given quantities of goods.

        Parameters:
        x1A (float): Quantity of good 1 consumed by consumer A.
        x2A (float): Quantity of good 2 consumed by consumer A.

        Returns:
        float: The utility of consumer A, calculated as (x1A**alpha)*(x2A**(1 - alpha)), 
        where alpha is the preference parameter for good 1.
        """
        return (x1A**self.par.alpha)*(x2A**(1 - self.par.alpha))

    def utility_B(self, x1B, x2B):
        """
        This method calculates the utility of consumer B for given quantities of goods.

        Parameters:
        x1B (float): Quantity of good 1 consumed by consumer B.
        x2B (float): Quantity of good 2 consumed by consumer B.

        Returns:
        float: The utility of consumer B, calculated as (x1B**beta)*(x2B**(1 - beta)), 
        where alpha is the preference parameter for good 1.
        """
        return (x1B**self.par.beta)*(x2B**(1 - self.par.beta))

    def demand_A(self,p1):
        """
        Calculates and returns the demand of consumer A for two goods.

        Parameters:
        p1 (float): The price of good 1.

        Returns:
        tuple: The demand for good 1 and good 2 (x1A, x2A).
        """

        # The numeraire is p2 = 1
        p2 = 1
        
        # Find consumer A's demand for good 1 and 2
        x1A = self.par.alpha*((p1*self.par.w1A + p2*self.par.w2A) / p1)
        x2A = (1 - self.par.alpha)*( (p1*self.par.w1A + p2*self.par.w2A) / p2 )
        
        # Returning the demand in a tuple
        return (x1A,x2A)

    def demand_B(self,p1):
        """
        Calculates and returns the demand of consumer B for two goods.

        Parameters:
        p1 (float): The price of good 1.

        Returns:
        tuple: The demand for good 1 and good 2 (x1B, x2B).
        """

        # The numeraire is p2 = 1
        p2 = 1
        
        # Find consumer B's demand for good 1 and 2
        x1B = self.par.beta*((p1*self.par.w1B + p2*self.par.w2B) / p1)
        x2B = (1 - self.par.beta)*( (p1*self.par.w1B + p2*self.par.w2B) / p2 )
        
        # Returning the demand in a tuple
        return (x1B,x2B)

    def check_market_clearing(self,p1):
        """
        Checks if the market is clearing for two goods.

        Parameters:
        p1 (float): The price of good 1.

        Returns:
        tuple: The excess demand for good 1 and good 2 (eps1, eps2).
        """

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def find_market_clearing(self):
        """
        This method calculates the market clearing price and the corresponding demand for consumer A.

        Returns:
            price (float): The market clearing price.
            x1A (float): The quantity of good 1 demanded by consumer A at the market clearing price.
            x2A (float): The quantity of good 2 demanded by consumer A at the market clearing price.
        """

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

