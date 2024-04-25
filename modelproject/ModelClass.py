# Import statements
from scipy import optimize
from types import SimpleNamespace
import sympy as sm


class MalthusModel():

    def __init__(self):

        self.par = SimpleNamespace()
        self.val = SimpleNamespace()

        self.setup()


    def setup(self):

        par = self.par
        val = self.val

        # Initial values
        par.alpha = sm.symbols('alpha')
        par.technology = sm.symbols('A')
        par.land = sm.symbols('X')
        par.beta = sm.symbols('beta')
        par.small_lambda = sm.symbols('lambda')
        par.tau = sm.symbols('tau')
        par.mu = sm.symbols('mu')
        par.L0 = sm.symbols('L_0')

        par.eta = sm.symbols('eta')


        par.L_t = sm.symbols('L_t')
        par.y_t_1 = sm.symbols('y_{t-1}')


        # Model parameters
        val.alpha = 0.5
        val.technology = 1
        val.land = 1
        val.beta = 0.5
        val.small_lambda = 0.5
        val.tau = 0.2
        val.mu = 0.5
        val.L0 = 1


    def L_transition_eq(self):

        par = self.par
        #eta = ( (1 - par.beta) / par.small_lambda ) * (1 - par.tau)
        Y = par.L_t**(1 - par.alpha) * (par.technology * par.land)**par.alpha
        L_transition_eq = ( par.eta * Y ) + (1 - par.mu)*par.L_t

        return L_transition_eq


    def L_steady_state(self):
        
        # Access model parameters
        par = self.par

        # Isolate L in steady state
        L_ss = sm.Eq(par.L_t, self.L_transition_eq())
        lss = sm.solve(L_ss, par.L_t)[0]

        return lss
    

    def L_numerical_solution():

        # Access model parameters
        par = self.par

        # Access model variables
        val = self.val

        




