##############
# Question 1 #
##############


# Import SimpleNamespace
from types import SimpleNamespace


def optimal_script_l_j_firms(p_j, w, par):

    value = ( p_j * par.A * par.gamma ) / w

    return value**(1 / (1 - par.gamma))


def optimal_y_j_firms(p_j, w, par):

    return par.A * ( ( optimal_script_l_j_firms(p_j, w, par) )**(par.gamma) )


def firm_profit(p_j, w, par):

    value1 = ( p_j*par.A*par.gamma / w)**(1 / (1 - par.gamma))

    value2 = (1 - par.gamma / par.gamma)*w*value1

    return value2


def find_optimal_l_and_y_firms(prices, w, par):

    optimal_l_array = []
    optimal_y_array = []
    optimal_profit_array = []

    for p in prices:

        l = optimal_script_l_j_firms(p, w, par)
        y = optimal_y_j_firms(p, w, par)
        profit = firm_profit(p, w, par)

        optimal_l_array.append(l)
        optimal_y_array.append(y)
        optimal_profit_array.append(profit)

    # Returns a tuple of optimal l, y and profit
    return (optimal_l_array, optimal_y_array, optimal_profit_array)
    


def c1_optimal(l, p1, p2, w, par):

    numerator = par.alpha * ( w * l + par.T + firm_profit(p1, w, par) + firm_profit(p2, w, par) )

    denominator = p1

    return numerator / denominator



def c2_optimal(l, p2, w, par):

    numerator = 

    denominator = 

    return numerator / denominator



def find_optimal_script_l_consumers():

