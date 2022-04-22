#===============================================================================
# Author: James Wenk
#===============================================================================



from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import random
from operator import add


SET_SIZE = 2 # The side length of the initial box the set lies in
NUM_POINTS_SET = 25 # The number of points per row / col of bounding box
N = 10 # The number of segments of gamma
STEP_SIZE = 1/N # The length of each segment of gamma
NUM_ANGLES = 10000 # The number of angles to store in the lookup table

USE_FAST_TRIG = True # Set to True to use pre computed trig look up approximations


# Initializes trig lookup tables
cos_table = np.zeros(NUM_ANGLES)
sin_table = np.zeros(NUM_ANGLES)

for j in range(NUM_ANGLES):
    cos_table[j] = np.cos(2 * np.pi * j / NUM_ANGLES)
    sin_table[j] = np.sin(2 * np.pi * j / NUM_ANGLES)

# Determines if the trig lookup is used or not
def f_cos(t):
    if USE_FAST_TRIG:
        return cos_table[int(np.floor(np.mod(t, 2 * np.pi) * NUM_ANGLES / (2 * np.pi)))]
    else:
        return np.cos(t)
    
# Determines if the trig lookup is used or not
def f_sin(t):
    if USE_FAST_TRIG:
        return sin_table[int(np.floor(np.mod(t, 2 * np.pi) * NUM_ANGLES / (2 * np.pi)))]
    else:
        return np.sin(t)

# This multiplies all the elements of a list by constant
def list_mult(lst, a):
    return [a * x for x in lst]

# This returns the area of S, where S is a NUM_POINTS_SET^2 length list with values in [0,1] saying how
# much it takes a rectangle at (i,j) by value S[NUM_POINTS_SET * i + j]
def area (S):
    return sum(map(np.abs, S)) * (SET_SIZE ** 2) / (NUM_POINTS_SET ** 2) # Uses abs to penalize non solutions

# Takes in a list gamma = (x,y, t1,..,tN), and an integer k, and outputs the location of the kth point of gamma
def kth_pt (gamma, k):
    return gamma[0] + STEP_SIZE * sum(map(f_cos,gamma[2:k + 2])), gamma[1] + STEP_SIZE * sum(map(f_sin,gamma[2:k + 2]))

# Takes in a shape S and a curve gamma, and outputs how much of gamma lies outside of S
def error (S, gamma):
    err = 0
    for j in range(N):
        err = err + error_seg(S, kth_pt(gamma,j), kth_pt(gamma, j+1))
    return err

# Returns all integers between two floats
def float_range(a,b):
    if a < b:
        return list(range(np.ceil(a), np.floor(b) + 1))
    else:
        return list(range(np.ceil(b), np.floor(a) + 1))

# Takes in points p1, p2 and outputs two arrays of all of the x,y locations of where line segment intersects grid that S lies on
# It also includes the endpoints of the segment
def split(p1,p2):
    (x0, y0) = p1
    (x1, y1) = p2
    # Initialize return lists
    x = [x0]
    y = [y0]
    # Repeatedly calculates next intersection and appends it
    tempx = x0
    tempy = y0

# Takes in a x,y location and outputs the index for this point in S
def cord_to_idx (x,y):
    i = np.floor(x * NUM_POINTS_SET / SET_SIZE)
    j = np.floor(y * NUM_POINTS_SET / SET_SIZE)
    return int(NUM_POINTS_SET * i + j)

# Takes in a shape S and a segment p1p2, and outputs how much of gamma lies outside of S
# To make error work outside of the bounding box we add a penalty that increases outside of box
def error_seg (S, p1, p2):
    (x0, y0) = p1
    (x1, y1) = p2
    # This approximates segment by num_pts line segments
    num_pts = 10
    err_sum = 0
    tempx = 0
    tempy = 0
    for j in range(num_pts):
        tempx = (x1-x0) * j / num_pts + x0
        tempy = (y1-y0) * j / num_pts + y0
        if 0 <= tempx <= SET_SIZE and 0 <= tempy <= SET_SIZE:
            # Note the (1 - expression) is because we want the error not overlap
            err_sum = err_sum + np.abs(1-S[cord_to_idx(tempx, tempy)]) / num_pts
        else:
            err_sum = err_sum + (tempx - SET_SIZE / 2) ** 2 + (tempx - SET_SIZE / 2) ** 2
    return err_sum

# Takes in gamma and T, and outputs the transformed gamma, namely the x,y shift and altering all angles
def apply_transform (gamma, T):
    Tgamma = gamma.copy()
    Tgamma[0] = gamma[0] + T[0]
    Tgamma[1] = gamma[1] + T[1]
    for j in range (2, len(gamma)):
        Tgamma[j] = gamma[j] + T[2]
    return Tgamma

# This damps a change by taking two lists, applying a scaling coefficient to S2, and taking pointwise max
def damp (S1, S2, damp_factor):
    return list(map(max, list_mult(S2, damp_factor), S1))

# The desired function we want to optimize, with three versions based on parameter order
def area_and_error1 (S, gamma, T):
    return area(S) + error(S, apply_transform(gamma, T))

# The desired function we want to optimize, with three versions based on parameter order
# This allows for damping inside the optimization function
def area_and_error1_damp (S, oldS, damp_factor, gamma, T):
    return area_and_error1(damp(S, oldS, damp_factor), gamma, T)

# The desired function we want to optimize, with three versions based on parameter order
def area_and_error2 (gamma, S, T):
    # This checks to make sure T(gamma) lies inside of the bounding box, and returns a positive value otherwise
    max_dist = 0
    for j in range(N):
        x, y = kth_pt(apply_transform(gamma, T),j)
        max_dist = max(np.abs(x - SET_SIZE / 2), np.abs(y - SET_SIZE / 2), max_dist)
    if max_dist > SET_SIZE / 2:
        return max_dist
    return (-1) * (area(S) + error(S, apply_transform(gamma, T)))

# The desired function we want to optimize, with three versions based on parameter order
def area_and_error3 (T, S, gamma):
    return area(S) + error(S, apply_transform(gamma, T))

# This function uses pre-packaged methods to optimize by calling them sequentially
def bb_optimize1 (S, gamma, T, k, alg):

    # Sets the maximum number of iterations per step
    max_itr = 1
    if(alg == 'Nelder-Mead'):
        max_itr = 5

    # This stores the values of the error over iterations
    err = [area_and_error1(S, gamma, T)]

    # This calls the optimize functions in sequence
    for j in range(k):
        S = optimize.minimize(area_and_error1, S, args = (gamma, T), method = alg, options = {'maxiter':max_itr}).x #, constraints = S_bounds + gamma_bounds + T_bounds).x
        gamma = optimize.minimize(area_and_error2, gamma, args = (S, T), method = alg, options = {'maxiter':max_itr}).x #, constraints = gamma_bounds+ S_bounds + T_bounds).x
        T = optimize.minimize(area_and_error3, T, args = (S, gamma), method = alg, options = {'maxiter':max_itr}).x #, constraints = T_bounds + S_bounds + gamma_bounds).x
        
        # This updates error list
        err.append(area_and_error1(S, gamma, T))
    return S, gamma, T, err

# This function uses pre-packaged methods to optimize by calling them sequentially
# but adds a damping factor on the change of the set to smooth behavior hoping for convergence
def bb_optimize2 (S, gamma, T, k, damp_factor, alg):

    # Sets the maximum number of iterations per step
    max_itr = 1
    if(alg == 'Nelder-Mead'):
        max_itr = 5

    # This stores the values of the error over iterations
    err = [area_and_error1(S, gamma, T)]

    # This calls the optimize functions in sequence
    for j in range(k):
        # We will average the new S into the old S to reduce the jitterness of the outputs
        old_S = S.copy()
        temp_S = optimize.minimize(area_and_error1, S, args = (gamma, T), method = alg, options = {'maxiter':max_itr}).x #, constraints = S_bounds + gamma_bounds + T_bounds).x
        # Applies the damping factor
        S = damp(temp_S, old_S, damp_factor)

        gamma = optimize.minimize(area_and_error2, gamma, args = (S, T), method = alg, options = {'maxiter':max_itr}).x #, constraints = gamma_bounds+ S_bounds + T_bounds).x
        T = optimize.minimize(area_and_error3, T, args = (S, gamma), method = alg, options = {'maxiter':max_itr}).x #, constraints = T_bounds + S_bounds + gamma_bounds).x
        
        # This updates error list
        err.append(area_and_error1(S, gamma, T))
                
    return S, gamma, T, err

# This function uses pre-packaged methods to optimize by calling them sequentially
# but adds a damping factor on the change of the set to smooth behavior hoping for convergence
# This version adds the damping inside of the optimization function
def bb_optimize3 (S, gamma, T, k, damp_factor, alg):

    # Sets the maximum number of iterations per step
    max_itr = 1
    if(alg == 'Nelder-Mead'):
        max_itr = 5

    # This stores the values of the error over iterations
    err = [area_and_error1(S, gamma, T)]

    # This calls the optimize functions in sequence
    for j in range(k):
        # We will average the new S into the old S to reduce the jitterness of the outputs
        old_S = S.copy()
        S = optimize.minimize(area_and_error1_damp, S, args = (old_S, damp_factor, gamma, T), method = alg, options = {'maxiter':max_itr}).x #, constraints = S_bounds + gamma_bounds + T_bounds).x

        gamma = optimize.minimize(area_and_error2, gamma, args = (S, T), method = alg, options = {'maxiter':max_itr}).x #, constraints = gamma_bounds+ S_bounds + T_bounds).x
        T = optimize.minimize(area_and_error3, T, args = (S, gamma), method = alg, options = {'maxiter':max_itr}).x #, constraints = T_bounds + S_bounds + gamma_bounds).x
        
        # This updates error list
        err.append(area_and_error1(S, gamma, T))
                
    return S, gamma, T, err

# Uses randimization and a gradient descent method to try to optimize
def rand_optimize1 (S, gamma, T, num_iter):
    # Controls the maximum possible deviation of the perturbations
    # Possible to lower this over iterations
    max_change = 0.1

    #This initializes the the perturbation lists
    S_perturb = S.copy()
    gamma_perturb = gamma.copy()
    T_perturb = T.copy()

    # This stores the values of the error over iterations
    err = [area_and_error1(S, gamma, T)]

    for i in range(num_iter):
        # This creates a random perturbation each iteration
        for j in range(len(S_perturb)):
            S_perturb[j] = S[j] + random.uniform(-max_change, max_change)
        for j in range(len(gamma_perturb)):
            gamma_perturb[j] = gamma[j] + random.uniform(-max_change, max_change)
        for j in range(len(T_perturb)):
            T_perturb[j] = T[j] + random.uniform(-max_change, max_change)

        # This checks to see if the new point is in bounds and better, and if better moves
        S_perturb_valid = all(0 <= x <= 1 for x in S_perturb)
        if S_perturb_valid and area_and_error1(S_perturb, gamma, T) < area_and_error1(S, gamma,T):
            S = S_perturb.copy()

        gamma_perturb_valid = all(0 <= kth_pt(apply_transform(gamma_perturb, T),k)[0] <= SET_SIZE and \
                                  0 <= kth_pt(apply_transform(gamma_perturb, T),k)[1] <= SET_SIZE for k in range(N+1)) \
                                and all(0 <= x <= 2 * np.pi for x in gamma_perturb[2:])
        if gamma_perturb_valid and area_and_error1(S, gamma_perturb, T) > area_and_error1(S, gamma,T):
            gamma = gamma_perturb.copy()

        T_perturb_valid = (0 <= T_perturb[0] <= SET_SIZE) and (0 <= T_perturb[1] <= SET_SIZE) and (0 <= T_perturb[2] <= 2 * np.pi)
        if T_perturb_valid and area_and_error1(S, gamma, T_perturb) < area_and_error1(S, gamma,T):
            T = T_perturb.copy()

        # This updates error list
        err.append(area_and_error1(S, gamma, T))

    return S, gamma, T, err

# Uses randimization to change an individiual point of S, so is a limited gradient descent method to try to optimize
# Also uses a random curve each time as an advisarial choice
def rand_optimize2 (S, gamma, T, num_iter):
    
    #This initializes the the perturbation lists
    S_perturb = S.copy()
    rand_gamma = gamma.copy()
    T_rand = T.copy()

    # This stores the values of the error over iterations
    err = [area_and_error1(S, gamma, T)]

    for i in range(num_iter):
        # Creates a peturbation by randomly changing one point of S
        S_perturb[random.randint(0, len(S)-1)] = random.randint(0,1)

        # Creates a random gamma
        rand_gamma[0] = random.uniform(0,2)
        rand_gamma[1] = random.uniform(0,2)
        for j in range(len(rand_gamma)):
            rand_gamma[j] = random.uniform(0,2 * np.pi)

        # Creates a random transformation
        T_rand[0] = random.uniform(0,2)
        T_rand[1] = random.uniform(0,2)
        T_rand[2] = random.uniform(0,2 * np.pi)

        # This checks to see if the new point is in bounds and better, and if better moves
        S_perturb_valid = all(0 <= x <= 1 for x in S_perturb)
        if S_perturb_valid and area_and_error1(S_perturb, gamma, T) < area_and_error1(S, gamma,T):
            S = S_perturb.copy()

        rand_gamma_valid = all(0 <= kth_pt(apply_transform(rand_gamma, T),k)[0] <= SET_SIZE and \
                                  0 <= kth_pt(apply_transform(rand_gamma, T),k)[1] <= SET_SIZE for k in range(N+1)) \
                                and all(0 <= x <= 2 * np.pi for x in rand_gamma[2:])
        if rand_gamma_valid and area_and_error1(S, rand_gamma, T) > area_and_error1(S, gamma,T):
            gamma = rand_gamma.copy()

        if area_and_error1(S, gamma, T_rand) < area_and_error1(S, gamma,T):
            T = T_rand.copy()

        # This updates error list
        err.append(area_and_error1(S, gamma, T))

    return S, gamma, T, err

def rand_and_bb_opt (S, gamma, T):
    
    return S, gamma, T, err

### This was used for methods besides nelder mead which could accomodate bounds
##S_bounds = []
##gamma_bounds = []
##T_bounds = []
##
##for j in range(NUM_POINTS_SET ** 2):
##    S_bounds.append((0,1))
##gamma_bounds.append((0, SET_SIZE))
##gamma_bounds.append((0, SET_SIZE))
##for j in range(2, N + 2):
##    gamma_bounds.append((0, 2 * np.pi))

S = np.ones(NUM_POINTS_SET ** 2) / 2
gamma = np.ones(2 + N) / 2
T = [0,0,0] # Values are x-shift, y-shift, rotation theta

# Sets the number of iterations to use of the optimization algorithm
num_iter = 100

# Sets the desired algorithm to use
method = 'Powell'

#S, gamma, T, err = bb_optimize1(S, gamma, T, num_iter, method)
#S, gamma, T, err = bb_optimize2(S, gamma, T, num_iter, 0.95, method)
S, gamma, T, err = bb_optimize3(S, gamma, T, num_iter, 0.95, method)
#S, gamma, T, err = rand_optimize2(S, gamma, T, num_iter, method)


# This creates a mesh for the domain of a contour plot
xx = np.linspace(0, SET_SIZE, NUM_POINTS_SET)
yy = np.linspace(0, SET_SIZE, NUM_POINTS_SET)
X, Y = np.meshgrid(xx, yy)
# This creates the values of the contour plot, namely the values of S
Z = np.reshape(S,(NUM_POINTS_SET, NUM_POINTS_SET))

# This plots the contour plot of S
plt.contourf(X,Y,Z)
plt.colorbar()
plt.title('The final set S after ' + str(num_iter) + ' iterations \n Method used: bb_optimize3, ' + method + ', damping_factor = 0.95')
plt.show()

# This plots the error plot per iteration
plt.title('The total error of the set S by iteration \n Method used: bb_optimize3, ' + method)
plt.scatter(np.arange(0, len(err)), err)
plt.show()
