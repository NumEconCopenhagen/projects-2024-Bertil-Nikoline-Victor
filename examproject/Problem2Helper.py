##############
# Question 1 #
##############

# Import random
import random


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

    # Set the seed for reproducibility
    random.seed(1996)

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

def results_from_2(par):

    range_of_individuals = range(0,par.N)

    for i in range_of_individuals:
        # Note that individual i starts from 0 in this case but 
        # actually represents individul 1 in the math stated in the question

        # Number of friends person i has
        Fi = par.F[i]

        # J has been hardcoded to 3
        J = 3 

        # Array to store calculated info
        career_track_info = []

        highest_utility = -1 * math.inf

        for j in range(J):
            
            v_j = (j + 1)

            # Error terms for further calculations
            error_term_array = []

            for index in range(Fi):
                error = error_term_friends(par)
                error_term_array.append(error)
            
            utility_friends = v_j + (sum(error_term_array) / len(error_term_array))

            utility_actual = v_j + error_term(par)

            info = ((j + 1), utility_friends, utility_actual)



            career_track_info.append(info)
        
        for x in career_track_info:
            print(x)



        # # Calculating error terms
        # for index in range(J_times_Fi):
        #     error_term = error_term_friends(par)
        #     error_term_array.append(error_term)

        # # This is an array of expected utility from each career path
        # # where object 0 is career path 1
        # expected_utility_from_carrers = []

        # for j in range(J):
            


        # print(error_term_array)


        # #[print(f'Individual {i} with F value {Fi}')]



def error_term_friends(par):

    # Define the mean and standard deviation
    mean = 0
    standard_deviation = (par.sigma**2)

    return random.gauss(mean, standard_deviation)



def average_utility_of_friends(par, Fi, j_career_choice):

    for j in range(J):
        print(f'j = {j}')


    



