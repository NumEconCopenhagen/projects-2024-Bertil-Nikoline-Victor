# Import math
import math


##############
# Question 1 #
##############


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



#**Question 2:** Compute the barycentric coordinates of the point $y$ 
# with respect to the triangles $ABC$ and $CDA$. Which triangle 
# is $y$ located inside?

##############
# Question 2 #
##############


def f(X):
    return X[0]*X[1]



def r1_ABC(A, B, C, y):

    (y1, y2) = y
    
    numerator = (B[1] - C[1])*(y1 - C[0]) + (C[0] - B[0])*(y2 - C[1])

    denominator = (B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1])

    return numerator / denominator



def r1_CDA(C, D, A, y):
    return r1_ABC(C, D, A, y)



def r2_ABC(A, B, C, y):

    (y1, y2) = y

    numerator = (C[1] - A[1])*(y1 - C[0]) + (A[0] - C[0])*(y2 - C[1])

    denominator = (B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1])
    
    return numerator / denominator



def r2_CDA(C, D, A, y):
    return r2_ABC(C, D, A, y)



def r3(r1, r2):
    return 1 - r1 - r2



def point_inside_ABC(A, B, C, y):

    r1 = r1_ABC(A, B, C, y)
    r2 = r2_ABC(A, B, C, y)
    r3 = 1 - r1 - r2

    if (is_number_in_range_0_to_1(r1) and
        is_number_in_range_0_to_1(r2) and
        is_number_in_range_0_to_1(r3)):
        
        return r1*f(A) + r2*f(B) + r3*f(C)
    else:
        return None



def is_number_in_range_0_to_1(num):
    return 0 <= num <= 1 



def point_inside_CDA(C, D, A, y):
    return point_inside_ABC(C, D, A, y)



def y_calculated(A, B, C, r1, r2, r3):
    return None
