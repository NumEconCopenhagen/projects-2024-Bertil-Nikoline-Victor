# Import statements
from scipy import optimize
from types import SimpleNamespace
import sympy as sm
import numpy as np


class MalthusModel():


    # Function for initial setup

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
        par.y_t = sm.symbols('y_{t-1}')


        # Model parameters
        val.alpha = 0.15
        val.beta = 0.3
        val.small_lambda = 0.4
        val.tau = 0.25
        val.mu = 0.45
        val.eta = ((1 - val.beta) / val.small_lambda) * (1 - val.tau)

        # Model settings
        val.T = 400

        # Initial values
        val.L0 = 1
        val.technology = 1
        val.land = 1



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



    # Functions to solve model numerically

    def diff_between_periods(self, variable):

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


    def numerical_solution_steady_state(self):

        # Access model parameters
        par = self.par

        # Access model variables
        val = self.val


        # Define the bounds for the search
        bounds = (0, 100)  # Adjust the bounds as needed

        # Number of multistarts
        num_starts = 20

        # Initialize the smallest residual as infinity
        smallest_residual = np.inf
        
        # Set steady state labor force to infinity
        labor_ss = np.inf

        # Loop through each random initial guess
        for _ in range(num_starts):
            
            # Generate random initial guess within the bounds
            x0 = np.random.uniform(bounds[0], bounds[1])

            # Find the root
            result = optimize.root(self.diff_between_periods, x0)

            # If difference is smaller than the current difference, the update the steady state value for the labor force
            if result.x < smallest_residual:
                smallest_residual = self.diff_between_periods(result.x)
                labor_ss = result.x

        # Steady state output
        output_ss = self.Y_t(labor_ss, val.alpha, val.technology, val.land)

        # Steady state output pr. worker
        output_pr_worker_ss = self.y_t(labor_ss, output_ss)

        # Steady state birth rate
        birth_rate_ss = self.n_t(labor_ss, output_ss, val.eta)

        # Returning the steady state values for multiple variables in the model and the smallest residual to check if something has gone wrong with the optimization
        return labor_ss, output_ss, output_pr_worker_ss, birth_rate_ss, smallest_residual
    


    # Simulate transition to steady state

    def simulate_transition_ss(self, alpha, beta, small_lambda, tau, mu, X_shock_size, A_shock_size, X_shock_time, A_shock_time):

        # Access model variables
        val = self.val

        # Update values for parameters in the model
        val.alpha = alpha
        val.beta = beta
        val.small_lambda = small_lambda
        val.tau = tau
        val.mu = mu
        val.eta = ((1 - val.beta) / val.small_lambda) * (1 - val.tau)


        # Number of periods to iterate over
        T = val.T

        # Lists to store transition values in
        L = np.zeros(T)     # Work force
        Y = np.zeros(T)     # Output
        y = np.zeros(T)     # Output pr. worker
        X = np.zeros(T)     # Land
        A = np.zeros(T)     # Technology
        n = np.zeros(T)     # Birth rate
        
        # Set initial values
        L[0] = val.L0
        Y[0] = self.Y_t(L[0], val.alpha, val.technology, val.land)
        y[0] = self.y_t(L[0], Y[0])
        X[0] = val.land
        A[0] = val.technology
        n[0] = self.n_t(L[0], Y[0], val.eta)

        # Interate over periods to create transition towards steady state
        for t in range(1, T):

            # Set values in period t
            L[t] = self.L_t1(L[t - 1], Y[t - 1], val.eta, val.mu)
            Y[t] = self.Y_t(L[t], val.alpha, A[t - 1], X[t - 1])
            y[t] = self.y_t(L[t], Y[t])
            n[t] = self.n_t(L[t], Y[t], val.alpha)

            # Add shock to the amount of land
            if X_shock_time == t: 
                X[t] = X_shock_size
            else:
                X[t] = X[t - 1]

            # Add shock to the technology level
            if A_shock_time == t: 
                A[t] = A_shock_size
            else:
                A[t] = A[t - 1]
        
        
        return (L, Y, y, X, A, n) 

