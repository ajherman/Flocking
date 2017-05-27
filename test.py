# Runs tests on the other files.  Make sure this passes before merging to dev or master!

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:46:44 2017

@author: Ari
"""

import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from FlockAnimation import ScatterAnimation,QuiverAnimation 
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation, OlfatiFlockingSimulationTF

###########################
# Set simulation parameters
###########################
sim_params = SimulationParams()
sim_params.num_boids = 25
sim_params.num_iters = 25

##########################
# Set animation parameters
##########################
ani_params = AnimationParams()
ani_params.save = False
ani_params.show = False
ani_params.fname = None

###########
# Run tests
###########

# Basic
for run_method in ['NP','TF']:     
    for dim in [2,3]:
        print("Basic test on dimension " + str(dim) + " with " + run_method + ":  ", end ="")
        
        try:
            if run_method == 'NP':
                flock_sim = OlfatiFlockingSimulation()
            
            elif run_method == 'TF':
                flock_sim = OlfatiFlockingSimulation()
            
            else:
                assert("Invalid dimension given in test program")
                
            # Set dimension
            sim_params.dim = dim

            # Set gamma path
            sim_params.gamma_path = 'circle'

            # Set simulation parameters
            flock_sim.params = sim_params
        
            # Init simulation
            flock_sim.initSim()

            # Run simulation
            X,V = flock_sim.runSim()
            assert(isinstance(X,np.ndarray))
            assert(isinstance(V,np.ndarray))
            assert(np.shape(X) == (25,26,dim))
            assert(np.shape(V) == (25,26,dim))
            print("Pass!")
        
        except:
            print("Fail.")


# Test quiver animation
print("Quiver animation test:  ", end = "")

try:
    flock = QuiverAnimation(X,0.01*V/norm(V,axis=2,keepdims=True))
    flock.animate(show=False,save=True,fname="Quiver_test")
    print("Pass!")

except:
    print("Fail.")

# Test scatter animation
print("Scatter animation test:  ", end = "")

try:
    flock = ScatterAnimation(X)
    flock.animate(show=False,save=True,fname="Scatter_test")
    print("Pass!")

except:
    print("Fail.")

print("There should be two animations in folder: Quiver_test.mp4 and Scatter_test.mp4")
