##############
# Question 1 #
##############


# Import SimpleNamespace
from types import SimpleNamespace

# Import numpy
import numpy as np

# Import math
import math

# Import itertools
import itertools

# Import minimize
from scipy.optimize import minimize

# Import copy
import copy



def optimal_script_l_j_firms(p_j, w, par):

    value = ( p_j * par.A * par.gamma ) / w

    return value**(1 / (1 - par.gamma))



def optimal_y_j_firms(p_j, w, par):

    return par.A * ( ( optimal_script_l_j_firms(p_j, w, par) )**(par.gamma) )



def firm_profit(p_j, w, par):

    value1 = ( (p_j * par.A * par.gamma) / w)**(1 / (1 - par.gamma))

    value2 = ((1 - par.gamma) / par.gamma) * w * value1

    return value2



def c1_optimal(l, p1, p2, w, par):

    numerator = par.alpha * ( w * l + par.T + firm_profit(p1, w, par) + firm_profit(p2, w, par) )

    denominator = p1

    return numerator / denominator



def c2_optimal(l, p1, p2, w, par):

    numerator = (1 - par.alpha) * ( w * l + par.T + firm_profit(p1, w, par) + firm_profit(p2, w, par) )

    denominator = p2 + par.tau

    return numerator / denominator



def optimal_l_consumer(l, p1, p2, w, par):

    c1 = c1_optimal(l, p1, p2, w, par)
    c2 = c2_optimal(l, p1, p2, w, par)

    value = math.log( (c1**par.alpha) * c2**(1 - par.alpha) ) - ( par.nu * ( l**(1 + par.epsilon) / (1 + par.epsilon) ) )

    return value



def script_l_to_optimize(l, params):

    p1 = params.p1
    p2 = params.p2
    w = params.w

    return -1 * optimal_l_consumer(l, p1, p2, w, params)



def find_market_clearing_conditions_for_p1_p2(p1_array, p2_array, w, par):

    # Create a list of all combinations of p1 and p2
    price_combinations = list(itertools.product(p1_array, p2_array))

    # Script l combinations
    script_l1 = []
    script_l2 = []
    script_l_total = []

    # Production (y) combinations
    y1 = []
    y2 = []

    # Consumption
    c1 = []
    c2 = []


    for (p1, p2) in price_combinations:
        
        #########################
        # Find script l 1 and 2 #
        #########################
        script_l1.append(optimal_script_l_j_firms(p1, w, par))
        script_l2.append(optimal_script_l_j_firms(p2, w, par))

        ##################
        # Find y 1 and 2 #
        ##################
        y1.append(optimal_y_j_firms(p1, w, par))
        y2.append(optimal_y_j_firms(p2, w, par))


        ######################
        # Find script l star #
        ######################
        
        # Initial guess
        x0 = np.array([0])

        # Bounds to ensure positive values
        bounds = [(0.0001, None)]  # x[0] and x[1] should be >= 1


        # Perform the minimization
        par.p1 = p1
        par.p2 = p2
        par.w = w
        result = minimize(script_l_to_optimize, x0, args=(par,), method='Powell', bounds=bounds)
        script_l_total.append(result.x[0])


    ######################
    # Find script l star #
    ######################

    for (index, l) in enumerate(script_l_total):

        price1 = price_combinations[index][0]
        price2 = price_combinations[index][1]

        c1.append(c1_optimal(l, price1, price2, w, par))
        c2.append(c2_optimal(l, price1, price2, w, par))

    return (price_combinations, script_l1, script_l2, script_l_total, y1, y2, c1, c2)



##############
# Question 2 #
##############

def find_equilibrium_prices(information):

    (price_combinations, script_l1, script_l2, script_l_total, y1, y2, c1, c2) = information

    p1s = []
    p2s = []

    condition1 = []
    condition2 = []
    condition3 = []


    for (index, (p1, p2)) in enumerate(price_combinations):

        # Calculating difference of all three market clearing conditions
        # Using Walras' law to find the equilibrium

        condition1.append(round(c1[index] - y1[index],2))

        condition2.append(round(c2[index] - y2[index], 2))

        condition3.append(round(script_l_total[index] - script_l1[index] - script_l2[index], 2))

        p1s.append(p1)
        p2s.append(p2)


    return (p1s, p2s, condition1, condition2, condition3)



##############
# Question 3 #
##############


def consumer_utility(tau, T, script_l, p1, p2, w, par):

    c1 = c1_optimal(script_l, p1, p2, w, par)
    c2 = c2_optimal(script_l, p1, p2, w, par)

    util_func = math.log((c1**par.alpha) * (c2**(1 - par.alpha))) - par.nu * ((script_l**(1 + par.epsilon)) / (1 + par.epsilon))

    return util_func



def SWF(U, kappa, y2_optimal, tau, T, script_l, p1, p2, w, par):
    U = consumer_utility(tau, T, script_l, p1, p2, w, par)
    return U - kappa*y2_optimal


def tau_to_maximize_SWF(tau_array, script_l, y2, p1, p2, w, par):

    SWF_array = []
    T_array = []

    for tau in tau_array:

        parameters = copy.deepcopy(par)

        parameters.tau = tau

        T = tau * c2_optimal(script_l, p1, p2, w, parameters)
        parameters.T = T
        T_array.append(T)

        U = consumer_utility(tau, T, script_l, p1, p2, w, parameters)
        swf = SWF(U, parameters.kappa, y2, tau, T, script_l, p1, p2, w, parameters)

        SWF_array.append(swf)
    

    return (tau_array, T_array, SWF_array)


