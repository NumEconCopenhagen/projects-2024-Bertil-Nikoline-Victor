##############
# Question 1 #
##############

# Import random
import random

# Import math
import math


def error_term(par):

    # Define the mean and standard deviation
    mean = 0
    standard_deviation = (par.sigma**2)

    return random.gauss(mean, standard_deviation)


def simulate_utility(par, career_choice):

    v_j = par.v[career_choice]

    error = error_term(par)

    return v_j + error


def expected_utility(par, career_choice):
    # Return the expected utility

    # Subtracting 1 for the career choice to be consistent with the mathematical formulation
    career_choice = career_choice - 1

    return par.v[career_choice]


def average_realised_utility(par, career_choice):
    # Return an average realised utility

    # Subtracting 1 for the career choice to be consistent with the mathematical formulation
    career_choice = career_choice - 1

    # Array for storing utilities
    utilities = []

    # Simulate K times
    for k in range(par.K):

        utility = simulate_utility(par, career_choice)

        utilities.append(utility)

    average = sum(utilities) / len(utilities)

    return average



##############
# Question 2 #
##############

def simulate_for_individual(par, type):


    range_of_individuals = range(0, type)

    # Store the results
    best_expected_utility = -1 * math.inf
    best_realized_utility = None
    best_j = None

    # J has been hardcoded to 3
    J = 3

    for j in range(J):

        expected_utilities = []

        for i in range_of_individuals:
        # Note that individual i starts from 0 in this case but 
        # actually represents individul 1 in the math stated in the question

            utility = simulate_utility(par, j)

            expected_utilities.append(utility)

        utility_value = sum(expected_utilities) / len(expected_utilities)

        if utility_value > best_expected_utility:
            best_expected_utility = utility_value
            best_realized_utility = simulate_utility(par, j)
            best_j = j + 1

    info = (best_j, best_expected_utility, best_realized_utility)

    return info


def mean_expected_utility(array):
    
    utilities = [expected_utility for _, expected_utility, _ in array]

    return sum(utilities) / len(utilities)



def mean_realized_utility(array):

    utilities = [realized_utility for _, _, realized_utility in array]

    return sum(utilities) / len(utilities)


def career_distribution(array, par):

    counter_array = [0,0,0]

    for (j, _, _) in array:
        i = j - 1
        counter_array[i] = counter_array[i] + 1

    fractions_array = [x / par.K for x in counter_array]

    return fractions_array
