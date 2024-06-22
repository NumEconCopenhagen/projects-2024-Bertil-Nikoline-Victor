#Question2
#Importing normal distribution for error term
from scipy.stats import norm
import numpy as np
from types import SimpleNamespace

class RandomUtilityClass:

    def __init__(self):
        par = self.par = SimpleNamespace()
        par.J = 3
        par.N = 10 
        par.K = 10000

        par.F = np.arange(1,par.N+1)
        par.sigma = 2

        par.v = np.array([1,2,3])
        par.c = 1
    
    def copy(self):
        other=RandomUtilityClass()
        other.par=deepcopy(self.par)
        other.utility=self.utility_random1
        other.L_opt=self.ErrorTerm_i
        return other

    def utility_random1(self,v,x):
        return v+x

    def ErrorTerm_i(self):
        par = self.par
        np.random.seed(42)
        return norm(loc=0,scale=par.sigma)
    
    def MonteCarlo_ErrorTerm_i_exp(self, v):
        par=self.par
        error_term_dist=self.ErrorTerm_i()
        X = error_term_dist.rvs(size=par.K)
        mean_X=np.mean(X)
        return self.utility_random1(v,mean_X)
    
    def MonteCarlo_ErrorTerm_i_avg(self,v):
        error_term_dist=self.ErrorTerm_i()
        X = error_term_dist.rvs(size=self.par.K)
        return np.mean(self.utility_random1(v,X))
    
    def ExpUtility_i(self):
        par=self.par
        exp_utilities = [self.MonteCarlo_ErrorTerm_i_exp(v) for v in par.v]
        for i, exp_utilities in enumerate(exp_utilities, start=1):
            print(f'Expected utility of choosing career path {i}: {exp_utilities:.3f}')
    
    def AvgUtility_i(self):
        par=self.par
        avg_utilities = [self.MonteCarlo_ErrorTerm_i_avg(v) for v in par.v]
        for i, avg_utilities in enumerate(avg_utilities, start=1):
            print(f'Avg. realised utility of choosing career path {i}: {avg_utilities:.3f}')

