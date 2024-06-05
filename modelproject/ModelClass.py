# Import statements
from scipy import optimize
from types import SimpleNamespace
import sympy as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets


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
        """
        Constructor for the MalthusModel class. Initializes namespaces and calls the setup method.
        """

        self.par = SimpleNamespace()
        self.val = SimpleNamespace()

        self.tmp = SimpleNamespace()

        self.setup()


    def setup(self):
        """
        Initializes the model's parameters with symbolic and actual values.
        """    

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
        """
        Returns the symbolic expression for the labor force transition equation.
        """

        # Access model parameters
        par = self.par

        Y = par.L_t**(1 - par.alpha) * (par.technology * par.land)**par.alpha
        L_transition_eq = ( par.eta * Y ) + (1 - par.mu)*par.L_t

        return L_transition_eq


    def symbolic_ss_L(self):
        """
        Solves and returns the symbolic steady state level of the labor force.
        """
        
        # Access model parameters
        par = self.par

        # Isolate L in steady state
        L_ss = sm.Eq(par.L_t, self.symbolic_L())
        lss = sm.solve(L_ss, par.L_t)[0]

        return lss
    

    def symbolic_ss_L_lambdify(self):
        """
        Returns a lambdified function for the steady state labor force.
        """
        
        # Access model parameters
        par = self.par

        # Return a lambdified version of the symbolic math - turned into a python function
        return sm.lambdify((par.technology, par.land, par.alpha, par.mu, par.eta), self.symbolic_ss_L(), 'numpy')



    def symbolic_y(self, L_ss):
        """
        Returns the symbolic expression for output per worker.
        """

        # Access model parameters
        par = self.par
        
        return ( ( par.technology*par.land ) / L_ss )**par.alpha


    def symbolic_ss_y(self):
        """
        Solves and returns the symbolic steady state output per worker.
        """

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
        """
        Returns a lambdified function for the steady state output per worker.
        """

        # Access model parameters
        par = self.par

        # Get symbolic math for the output pr. worker steady state expression
        symbolic_y = self.symbolic_ss_y()

        # Return a lambdified version of the symbolic math - turned into a python function
        return sm.lambdify((par.technology, par.land, par.alpha, par.mu, par.eta), symbolic_y, 'numpy')



    # Model equations for numerical solution

    def Y_t(self, L_t, alpha, A, X):
        """
        Calculates the production output given labor force and other parameters.
        """

        # Production function
        return ( L_t**(1 - alpha) )*(A*X)**alpha


    def n_t(self, L_t, Y_t, eta):
        """
        Calculates the births per capita given output and labor force.
        """

        # Births pr. capita
        return eta * (Y_t / L_t)


    def L_t1(self, L_t, Y_t, eta, mu):
        """
        Calculates the next period's labor force given current labor force, output, and parameters.
        """

        # Law of motion for the labor force
        return self.n_t(L_t, Y_t, eta) * L_t + (1 - mu) * L_t
    

    def y_t(self, L_t, Y_t):
        """
        Computes the production per labor force participant.
        """
        
        # Production pr. labor force participant
        return Y_t / L_t

    # Law of motion with technological growth
    def l_t1(self, l_t, g, eta, alpha, X, mu):
        """
        Adjusts the workforce for technological growth and attrition.
        """
        # Workforce adjusted for technological growth
        return eta * ( g**(-1) ) * (l_t**(1 - alpha)) * ( X**alpha ) + ( g**(-1) )*(1 - mu)*l_t


    # Functions to solve model numerically

    def L_diff_between_periods(self, variable):
        """
        Calculates the difference in labor force between the current and next period.

        Parameters:
        variable (float): The current labor force value.

        Returns:
        float: The difference between the labor force in the next period and the current labor force. 
        If the current labor force is less than or equal to zero, it returns a large number (np.inf) to indicate an unsuitable solution.

        Raises:
        ValueError: If the input 'variable' is not a positive number.
        """

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
        """
        Calculates the difference in the labor force between the current and next period.

        Parameters:
        variable (float): The current labor force value, which should be a non-negative number.

        Returns:
        float: The difference between the labor force in the next period and the current labor force. 
        If the current labor force is negative, it returns a large number (np.inf) to indicate an unsuitable solution.

        Raises:
        ValueError: If the input 'variable' is not a non-negative number.
        """

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
        """
        Finds the steady state of the model numerically and handles edge cases where initial guesses might lead to non-viable solutions.

        Parameters:
        with_A_growth (bool): A flag indicating whether to account for technological growth in the calculations. Defaults to False.

        Returns:
        tuple: A tuple containing steady state values for the labor force, output, output per worker, birth rate, 
        and the smallest residual from the optimization process. The tuple structure differs based on the 'with_A_growth' flag:
            - If 'with_A_growth' is True, the tuple is (labor_ss, 0, output_ss, output_pr_worker_ss, birth_rate_ss, smallest_residual).
            - If 'with_A_growth' is False, the tuple is (0, labor_ss, output_ss, output_pr_worker_ss, birth_rate_ss, smallest_residual).

        Raises:
        ValueError: If the bounds for the initial guesses are not set properly or if the number of multistarts is not a positive integer.
        """

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
        """
        Simulates the transition of the economy towards a steady state over a series of periods.

        Parameters:
        A_growth (bool): Flag indicating whether there is technological growth.
        shocks (bool): Flag indicating whether there are shocks to the economy.
        g (float): The growth rate of technology.
        alpha (float): Output elasticity with respect to labor.
        beta (float): Discount factor.
        small_lambda (float): The rate at which consumption turns into utility.
        tau (float): Tax rate.
        mu (float): Rate of labor leaving the workforce.
        X_shock_size (float): The size of the shock to the amount of land.
        A_shock_size (float): The size of the shock to the technology level.
        X_shock_time (int): The period when the shock to land occurs.
        A_shock_time (int): The period when the shock to technology occurs.

        Returns:
        tuple: A tuple containing arrays of the workforce (L), output (Y), output per worker (y), land (X), technology (A), birth rate (n), and workforce adjusted by technology level (l) for each period.

        Raises:
        ValueError: If any of the parameters are out of their logical or practical range.
        """

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
    


    ###########################
    #     Transition plot     #
    ###########################

    def transition_to_ss_plot(self):
        """
        Generates a steady-state plot for a given model using seaborn and matplotlib. It is designed to work with a model
        that has a 'T' attribute representing the time period.

        Parameters:
        None

        Returns:
        matplotlib.pyplot: A matplotlib pyplot object with the plot configuration set.
                        
        """
        
        # Set figure
        plt.figure(figsize=(7, 3))
        sns.set_style("whitegrid")
        
        # Set labels
        plt.xlabel('t - period', fontsize=11)
        plt.ylabel('Value', fontsize=11)

        # Adding a custom color palette
        custom_palette = sns.color_palette("husl", 5)
        sns.set_palette(custom_palette)

        # Set limits on the x axis
        plt.xlim(1, self.val.T)

        return plt
    

    def model_parameter_sliders(self):
        """
        Returns sliders to adjust the parameters in the Malthus model.
        """   

        alpha_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=self.val.alpha, description='Alpha', continuous_update=False)
        beta_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=self.val.beta, description='Beta', continuous_update=False)
        small_lambda_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=self.val.small_lambda, description='Lambda', continuous_update=False)
        tau_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=self.val.tau, description='Tau', continuous_update=False)
        mu_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=self.val.mu, description='Mu', continuous_update=False)

        return (alpha_slider, beta_slider, small_lambda_slider, tau_slider, mu_slider)


    def model_shock_sliders(self):
        """
        Returns sliders to add shocks to the land and technology level in the economy.
        """

        X_shock_size_slider = widgets.FloatSlider(min=0, max=5, step=0.01, value=0.8, description='X shock size', continuous_update=False)
        X_shock_time_slider = widgets.FloatSlider(min=0.0, max=self.val.T, step=1, value=200, description='X shock time period', continuous_update=False)
        A_shock_size_slider = widgets.FloatSlider(min=0, max=5, step=0.01, value=1.2, description='A shock size', continuous_update=False)
        A_shock_time_slider = widgets.FloatSlider(min=0.0, max=self.val.T, step=1, value=300, description='A shock time period', continuous_update=False)

        X_shock_size_slider.style = {'description_width': 'initial'} # Set the width to 'initial' (default) or a custom value
        X_shock_size_slider.layout.width = '50%'  # Adjust the width of the slider (e.g., '50%', '500px', etc.)

        X_shock_time_slider.style = {'description_width': 'initial'} # Set the width to 'initial' (default) or a custom value
        X_shock_time_slider.layout.width = '50%'  # Adjust the width of the slider (e.g., '50%', '500px', etc.)

        A_shock_size_slider.style = {'description_width': 'initial'} # Set the width to 'initial' (default) or a custom value
        A_shock_size_slider.layout.width = '50%'  # Adjust the width of the slider (e.g., '50%', '500px', etc.)

        A_shock_time_slider.style = {'description_width': 'initial'} # Set the width to 'initial' (default) or a custom value
        A_shock_time_slider.layout.width = '50%'  # Adjust the width of the slider (e.g., '50%', '500px', etc.)

        return (X_shock_size_slider, X_shock_time_slider, A_shock_size_slider, A_shock_time_slider)


    def plot_transition_towards_ss(self, L, Y, y, n, X, A):
        """
        Plots to show the transition towards steady state in the Malthus economy.
        """

        # L Y plot
        L_Y_plot = self.transition_to_ss_plot()
        L_Y_plot.plot(L, label="L", color='blue')
        L_Y_plot.plot(Y, label="Y", color='skyblue')
        L_Y_plot.title('Labor force (L) and output (Y)')
        L_Y_plot.legend(title='Series', loc='upper left')
        L_Y_plot.show()

        # y n plot
        y_n_plot = self.transition_to_ss_plot()
        y_n_plot.plot(y, label="y", color='blue')
        y_n_plot.plot(n, label="n", color='skyblue')
        y_n_plot.title('Output pr. worker (y) and birth rate (n)')
        y_n_plot.legend(title='Series', loc='upper left')
        y_n_plot.show()

        # X A plot
        X_A_plot = self.transition_to_ss_plot()
        X_A_plot.plot(X, label="X", linestyle=':', alpha=0.7, linewidth=2, color='blue')
        X_A_plot.plot(A, label="A", linestyle='-', alpha=0.7, linewidth=3, color='skyblue')
        X_A_plot.title('Land (X) and technology (A)')
        X_A_plot.legend(title='Series', loc='upper left')
        X_A_plot.show()
    

    def plot_transition_towards_ss_with_g_growth(self, l, A, L, Y, n, y):
        """
        Plots to show the transition towards steady state in the Malthus economy with technology growth.
        """


        # l plot
        l_plot = self.transition_to_ss_plot()
        l_plot.plot(l, label="l_t", color='red')
        l_plot.title('Workforce adjusted for technological growth (l)')
        l_plot.legend(title='Series', loc='upper left')
        l_plot.show()

        # A plot
        A_plot = self.transition_to_ss_plot()
        A_plot.plot(A, label="A_t", color='green')
        A_plot.title('Technology level (A)')
        A_plot.legend(title='Series', loc='upper left')
        A_plot.show()

        # L Y plot
        L_Y_plot = self.transition_to_ss_plot()
        L_Y_plot.plot(L, label="L_t", color='blue')
        L_Y_plot.plot(Y, label="Y_t", color='skyblue')
        L_Y_plot.title('Labor force (L) and output (Y)')
        L_Y_plot.legend(title='Series', loc='upper left')
        L_Y_plot.show()

        # n plot
        n_plot = self.transition_to_ss_plot()
        n_plot.plot(n, label="n_t", color='blue')
        n_plot.title('Birth rate (n)')
        n_plot.legend(title='Series', loc='upper left')
        n_plot.show()

        # y plot
        y_plot = self.transition_to_ss_plot()
        y_plot.plot(y, label="y_t", color='orange')
        y_plot.title('Output pr. worker (y)')
        y_plot.legend(title='Series', loc='upper left')
        y_plot.show()


