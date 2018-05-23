import numpy as np
from math import sqrt
from svm.utils import flatten_image

def get_x(X, i):
    x_filepath = X[i]
    return flatten_image(x_filepath)


def scale(X, I_positive, I_negative, lmbda):
    """
    returns a scaled input list where every elment is 
    a one dimensional list i.e dimension is 1xn. 
    Eg : If input image is of dim 25x25 this would be 1x625 
    """

    # cal centroids of +ve and -ve classes
    m_pos = cal_centroid(X, I_positive) 
    m_neg = cal_centroid(X, I_negative) 

    X_dash = [None]*len(X)
    for index in I_positive:
        x = get_x(X, index)
        x_dash = lmbda*np.array(x) + (1-lmbda)*np.array(m_pos)
        X_dash[index] = x_dash.tolist() 

    for index in I_negative:
        x = get_x(X, index)
        x_dash = lmbda*np.array(x) + (1-lmbda)*np.array(m_neg)
        X_dash[index] = x_dash.tolist() 
    return X_dash, m_pos, m_neg

def cal_centroid(X, I):
    """
    returns the centroid of give data points 
    """
    # cal the size of each input data element 
    input_dim = len(get_x(X, 0))
    
    # initializing centroid 
    m = [0]*input_dim

    # cal centroid 
    for index in I:
        x = get_x(X, index)
        for i in range(input_dim):
            m[i] += x[i]
    for i in range(input_dim):
        m[i] = m[i]/len(I)
    return m

def cal_distance(x,y):
    """
    calculates the distance two points of any dimensions 
    return -1 if input data points are of diff dimensions
    """
    # no of dimensions in given data point 
    n = len(x)

    # if both data points are not of same dimensions
    if n!= len(y):
       return -1

    # cal distance b/w points 
    val = 0 
    for i in range(n):
        val += (x[i] - y[i])**2
    d = sqrt(val)
    return d  

def cal_lmbda_max(X, I_positive, I_negative):
    """
    returns the max value of a 'lmbda' using the 
    formula lmbda <= r/(r_pos + r_neg)
    """

    # cal centroids of +ve and -ve classes
    m_pos = cal_centroid(X, I_positive)
    m_neg = cal_centroid(X, I_negative)

    # cal 'r' 
    r = cal_distance(m_pos, m_neg) 

    # cal 'r_pos'
    x = get_x(X, I_positive[0])
    r_pos = cal_distance(x,m_pos) 
    for index in I_positive[1:]:
        x = get_x(X, index)
        d = cal_distance(x,m_pos)
        if d > r_pos:
           r_pos = d
    
    # cal 'r_neg'
    x = get_x(X, I_negative[0])
    r_neg = cal_distance(x,m_neg) 
    for index in I_negative[1:]:
        x = get_x(X, index)
        d = cal_distance(x,m_neg)
        if d > r_neg:
           r_neg = d

    # cal val of 'lmbda_max'
    lmbda_max = r/(r_pos+r_neg)
    print "Lambda", lmbda_max
    return lmbda_max
     
"""
# code used for testing fun inside this module 
if __name__ == "__main__":
   X = [[1,2], [3,4], [5,6]] 
   I_1  = [0,1]
   I_2  = [2]
   y = cal_centroid(X,I_1)
   x_1 = scale(X, I_1, I_2, 0.5) 
   print ("---",y)
   print ("!!!",x_1)

   x = [1,2]
   y = [0,0]
   d = cal_distance(x,y)
   print ("---",d)

   val = cal_lmbda_max(X, I_1, I_2)
   print ("--",val)
"""

