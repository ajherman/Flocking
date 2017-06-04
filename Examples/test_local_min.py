# Test square local minimum


# Authors: Ari Herman & Taiyo Terada

# Displays counter example to lemma 3 from Olfati paper

import sys
sys.path.insert(0,"../ClassDefinitions")
import numpy as np
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation
from matplotlib import pyplot as plt

def dist(X,i,j):
    return np.sqrt(np.sum((X[:,i,:]-X[:,j,:])**2,axis=1))

def get_dist_vect(X):
    d1 = dist(X[-1:],0,1)
    d2 = dist(X[-1:],0,2)
    d3 = dist(X[-1:],0,3)
    d4 = dist(X[-1:],1,2)
    d5 = dist(X[-1:],1,3)
    d6 = dist(X[-1:],2,3)
    vect = np.concatenate((d1,d2,d3,d4,d5,d6))
    return vect

l1, l2 = 6.235837, 8.81880526 # Side and diagonal
square_dist_vect = np.array([l1,l1,l2,l2,l1,l1])

#######
# Setup
#######

# Set simulation parameters
sim_params = SimulationParams()
sim_params.set_dim(2)
sim_params.c_p = 0
sim_params.c_q = 0
sim_params.set_num_boids(4)
sim_params.set_gamma_path('circle')
sim_params.set_d(7.0) 
sim_params.set_r(13.0)
sim_params.set_num_iters(1000)
sim_params.set_save(False)
s = 6.2358 # Equilibrium side length for square
noise = 1.0

# Our counter example (stable square)
q_init = np.array([[0,0],[0,s],[s,0],[s,s]])

# Make simulation object
flock_sim = OlfatiFlockingSimulation()

######
# Test
######

for i in range(1000000): # Test random perturbations
    pert = 2*np.random.rand(sim_params.num_boids,sim_params.dim)-1
    # Square with noise
    sim_params.set_q_init(q_init+pert) 
    sim_params.set_p_init(np.random.normal(0.0,noise,size=(sim_params.num_boids,sim_params.dim)))

    # Set simulation parameters
    flock_sim.params = sim_params

    # Init sim
    flock_sim.initSim()

    # Run sim
    X,V = flock_sim.runSim()
 
    diff = np.sqrt(np.sum((get_dist_vect(X)-square_dist_vect))**2)
    
    if diff > 0.0005:
        print("Fail")
        print(diff)
        print(X[0])

    if i%1000 == 0:
        print("Iteration "+str(i)+": Pass")




