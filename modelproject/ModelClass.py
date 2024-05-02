# Import statements
from scipy import optimize
from types import SimpleNamespace
import sympy as sm
import numpy as np


class MalthusModel():
    """
    A class representing the Malthusian growth model with symbolic and numerical methods.

    This class models population growth and technology progress. It includes functions to:
    - Initialize the simulation settings.
    - Calculate stable conditions symbolically.
    - Provide numerical solutions.

    Attributes:
    par (SimpleNamespace): A namespace for storing symbolic parameters of the model.
    val (SimpleNamespace): A namespace for storing actual values of the model parameters.
    tmp (SimpleNamespace): A namespace for storing temporary model parameters that are updated.

    Methods:
    __init__: Constructor for the MalthusModel class. Initializes namespaces and calls the setup method.
    setup: Initializes the model's parameters with symbolic and actual values.
    symbolic_L: Returns the symbolic expression for the labor force transition equation.
    symbolic_ss_L: Solves and returns the symbolic steady state level of the labor force.
    symbolic_ss_L_lambdify: Returns a lambdified function for the steady state labor force.
    symbolic_y: Returns the symbolic expression for output per worker.
    symbolic_ss_y: Solves and returns the symbolic steady state output per worker.
    symbolic_ss_y_lambdify: Returns a lambdified function for the steady state output per worker.
    Y_t: Calculates the production output given labor force and other parameters.
    n_t: Calculates the births per capita given output and labor force.
    L_t1: Calculates the next period's labor force given current labor force, output, and parameters.

    """

    # Function for initial setup

    def __init__(self):

        self.par = SimpleNamespace()
        self.val = SimpleNamespace()

        self.tmp = SimpleNamespace()

        self.setup()


    def setup(self):

        par = self.par
        val = self.val
        tmp = self.tmp

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
        par.y_t = sm.symbols('y_{t-1}')


        ### Model parameters
        val.alpha = 0.15
        val.beta = 0.3
        val.small_lambda = 0.4
        val.tau = 0.25
        val.mu = 0.45
        val.eta = ((1 - val.beta) / val.small_lambda) * (1 - val.tau)
        
        # Growth factor (g > 1 means growth and g = 1 means stagnation in the technological level) 
        val.g = 1.02

        ### Model settings
        val.T = 400

        ### Initial values
        val.L0 = 1
        val.technology = 1
        val.land = 1


        # Temporary model parameters that are updated
        tmp.alpha = 0.15
        tmp.beta = 0.3
        tmp.small_lambda = 0.4
        tmp.tau = 0.25
        tmp.mu = 0.45
        tmp.eta = ((1 - val.beta) / val.small_lambda) * (1 - val.tau)
        
        # Growth factor (g > 1 means growth and g = 1 means stagnation in the technological level) 
        tmp.g = 1.02

        ### Model settings
        tmp.T = 400

        ### Initial values
        tmp.L0 = 1
        tmp.technology = 1
        tmp.land = 1



    # Functions for symbolic math

    def symbolic_L(self):

        # Access model parameters
        par = self.par

        Y = par.L_t**(1 - par.alpha) * (par.technology * par.land)**par.alpha
        L_transition_eq = ( par.eta * Y ) + (1 - par.mu)*par.L_t

        return L_transition_eq


    def symbolic_ss_L(self):
        
        # Access model parameters
        par = self.par

        # Isolate L in steady state
        L_ss = sm.Eq(par.L_t, self.symbolic_L())
        lss = sm.solve(L_ss, par.L_t)[0]

        return lss
    

    def symbolic_ss_L_lambdify(self):
        
        # Access model parameters
        par = self.par

        # Return a lambdified version of the symbolic math - turned into a python function
        return sm.lambdify((par.technology, par.land, par.alpha, par.mu, par.eta), self.symbolic_ss_L(), 'numpy')



    def symbolic_y(self, L_ss):

        # Access model parameters
        par = self.par
        
        return ( ( par.technology*par.land ) / L_ss )**par.alpha


    def symbolic_ss_y(self):

        # Access model parameters
        par = self.par

        # Symbolic steady state for L
        L_ss = self.symbolic_ss_L()

        # Isolate y in stedy state
        y_t = self.symbolic_y(L_ss)

        y_equation = sm.Eq(par.y_t, y_t)
        y_ss = sm.solve(y_equation, par.y_t)[0]

        return y_ss

    def symbolic_ss_y_lambdify(self):

        # Access model parameters
        par = self.par

        # Get symbolic math for the output pr. worker steady state expression
        symbolic_y = self.symbolic_ss_y()

        # Return a lambdified version of the symbolic math - turned into a python function
        return sm.lambdify((par.technology, par.land, par.alpha, par.mu, par.eta), symbolic_y, 'numpy')



    # Model equations for numerical solution

    def Y_t(self, L_t, alpha, A, X):
        # Production function
        return ( L_t**(1 - alpha) )*(A*X)**alpha


    def n_t(self, L_t, Y_t, eta):
        # Births pr. capita
        return eta * (Y_t / L_t)


    def L_t1(self, L_t, Y_t, eta, mu):
        # Law of motion for the labor force
        return self.n_t(L_t, Y_t, eta) * L_t + (1 - mu) * L_t
    

    def y_t(self, L_t, Y_t):
        # Production pr. labor force participant
        return Y_t / L_t

    # Law of motion with technological growth
    def l_t1(self, l_t, g, eta, alpha, X, mu):
        # Workforce adjusted for technological growth
        return eta * ( g**(-1) ) * (l_t**(1 - alpha)) * ( X**alpha ) + ( g**(-1) )*(1 - mu)*l_t


    # Functions to solve model numerically

    def L_diff_between_periods(self, variable):

        # Access model variables
        val = self.val

        # Setting value of the current labor force
        L_current = variable

        # Checks for edge cases, used in multi_start
        if L_current <= 0:
            # Return a very large residual to indicate a poor solution
            return np.inf


        # Finding the output in the economy
        Y = self.Y_t(L_current, val.alpha, val.technology, val.land)

        # Finding the labor force in the next period
        L_next = self.L_t1(L_current, Y, val.eta, val.mu)

        # Return the difference between the current labor force and the labor force in the next period
        return L_next - L_current
    

    def l_diff_between_periods(self, variable):

        # Access model variables
        val = self.val

        # Setting value of the current labor force
        l_current = variable

        # Checks for edge cases, used in multi_start
        if l_current < 0:
            # Return a very large residual to indicate a poor solution
            return np.inf

        # Finding the labor force in the next period
        l_next = self.l_t1(l_current, val.g, val.eta, val.alpha, val.land, val.mu)

        # Return the difference between the current labor force and the labor force in the next period
        return l_next - l_current


    def numerical_solution_steady_state(self, with_A_growth = False):

        # Access model parameters
        par = self.par

        # Access model variables
        val = self.val


        # Define the bounds for the search
        # The lower bounds has been added since the initial guesses become better by not guessing too low values
        bounds = (500, 10000)  # Adjust the bounds as needed

        # Number of multistarts
        num_starts = 100

        # Initialize the smallest residual as infinity
        smallest_residual = np.inf
        
        # Set steady state labor force to infinity
        labor_ss = np.inf

        # Loop through each random initial guess
        for _ in range(num_starts):
            
            # Generate random initial guess within the bounds
            x0 = np.random.uniform(bounds[0], bounds[1])

            # Optimize using L_diff_between_periods if there is no technological growth and use l_diff_between_periods if there is technological growth
            result = with_A_growth == True and optimize.root(self.l_diff_between_periods, x0, method="hybr") or optimize.root(self.L_diff_between_periods, x0, method="hybr")

            # If difference is smaller than the current difference, the update the steady state value for the labor force
            if result.x < smallest_residual:
                smallest_residual = with_A_growth == True and self.l_diff_between_periods(result.x) or self.L_diff_between_periods(result.x)

                labor_ss = result.x

        # Steady state output
        output_ss = self.Y_t(labor_ss, val.alpha, val.technology, val.land)

        # Steady state output pr. worker
        output_pr_worker_ss = self.y_t(labor_ss, output_ss)

        # Steady state birth rate
        birth_rate_ss = self.n_t(labor_ss, output_ss, val.eta)

        # Returning the steady state values for multiple variables in the model and the smallest residual to check if something has gone wrong with the optimization
        if with_A_growth:
            return labor_ss, 0, output_ss, output_pr_worker_ss, birth_rate_ss, smallest_residual
        else:
            return 0, labor_ss, output_ss, output_pr_worker_ss, birth_rate_ss, smallest_residual




    # Simulate transition to steady state

    def simulate_transition_ss(self, A_growth, shocks, g, alpha, beta, small_lambda, tau, mu, X_shock_size, A_shock_size, X_shock_time, A_shock_time):

        # Access model variables
        tmp = self.tmp
        val = self.val

        # Update values for parameters in the model
        tmp.alpha = alpha
        tmp.beta = beta
        tmp.small_lambda = small_lambda
        tmp.tau = tau
        tmp.mu = mu
        tmp.eta = ((1 - tmp.beta) / tmp.small_lambda) * (1 - tmp.tau)
        tmp.g = g


        # Number of periods to iterate over
        T = val.T

        # Lists to store transition values in
        L = np.zeros(T)     # Workforce
        Y = np.zeros(T)     # Output
        y = np.zeros(T)     # Output pr. worker
        X = np.zeros(T)     # Land
        A = np.zeros(T)     # Technology
        n = np.zeros(T)     # Birth rate
        l = np.zeros(T)     # Workforce adjusted by technology level
        
        # Set initial values
        L[0] = tmp.L0
        X[0] = tmp.land
        A[0] = tmp.technology
        Y[0] = self.Y_t(L[0], tmp.alpha, A[0], X[0])
        y[0] = self.y_t(L[0], Y[0])
        n[0] = self.n_t(L[0], Y[0], tmp.eta)
        l[0] = L[0] / A[0]

        # Interate over periods to create transition towards steady state
        for t in range(1, T):

            # Make sure to update the land and technology level
            A[t] = A[t - 1]
            X[t] = X[t - 1]

            # Checks if there is technological growth
            if A_growth == True:
                # A growth
                A[t] = A[t - 1]*tmp.g

            # Set values in period t
            L[t] = self.L_t1(L[t - 1], Y[t - 1], tmp.eta, tmp.mu)
            Y[t] = self.Y_t(L[t], tmp.alpha, A[t], X[t])

            # Transition path when there is technological growth
            l[t] = L[t] / A[t]

            y[t] = self.y_t(L[t], Y[t])
            n[t] = self.n_t(L[t], Y[t], tmp.eta)

            # Checks if there should be shocks to the economy
            if shocks == True:
                # Add shock to the amount of land
                if X_shock_time == t: 
                    X[t] = X_shock_size
                # Add shock to the technology level
                if A_shock_time == t: 
                    A[t] = A_shock_size
            

        return (L, Y, y, X, A, n, l) 
