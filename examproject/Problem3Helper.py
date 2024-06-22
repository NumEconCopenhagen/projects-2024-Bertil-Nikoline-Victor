# Import math
import math



def triangle_value_function(x1, y1, x2, y2):
    value1 = (x1 - y1)**2
    value2 = (x2 - y2)**2
    return math.sqrt(value1 + value2)


def find_x1_x2_triangle(X, y, type):

    smallest_value = math.inf
    optimal_coordinates = (0, 0)

    (y1, y2) = y

    for (x1, x2) in X:
        if ((type == "A" and x1 > y1 and x2 > y2) or
           (type == "B" and x1 > y1 and x2 < y2) or
           (type == "C" and x1 < y1 and x2 < y2) or
           (type == "D" and x1 < y1 and x2 > y2)):
            value = triangle_value_function(x1, y1, x2, y2)
            if value < smallest_value:
                
                smallest_value = value
                optimal_coordinates = (x1, x2)
    
    if smallest_value == None:
        return None
    else:
        return optimal_coordinates



def calculate_r1_ABC():
    
    


def calculate_r2_ABC():



def calculate_r1_CDA():




def calculate_r2_CDA():





def calculate_r3():



