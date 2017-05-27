# Runs tests on the other files.  Make sure this passes before merging to dev or master!

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:46:44 2017

@author: Ari
"""

import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from AnimateFunc import ScatterAnimation 
from AnimateFunc import QuiverAnimation 
from UserInput import SimulationParams, AnimationParams
from FlockFuncs import OlfatiFlockingSimulation, OlfatiFlockingSimulationTF

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
                flock_sim.eps,flock_sim.num_boids,flock_sim.a,flock_sim.b,flock_sim.c,flock_sim.h,flock_sim.r_a,flock_sim.d_a,flock_sim.dt,flock_sim.num_iters,flock_sim.gamma_path,flock_sim.dim,flock_sim.c_q,flock_sim.c_p = sim_params.eps,sim_params.num_boids,sim_params.a,sim_params.b,sim_params.c,sim_params.h,sim_params.r_a,sim_params.d_a,sim_params.dt,sim_params.num_iters,sim_params.gamma_path,sim_params.dim,sim_params.c_q,sim_params.c_p
            
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
    flock.animate(show=False,save=True,fname="Quiver test")
    print("Pass!")
except:
    print("Fail.")

# Test scatter animation

print("Scatter animation test:  ", end = "")
try:
    flock = ScatterAnimation(X)
    flock.animate(show=False,save=True,fname="Scatter test")
    print("Pass!")
except:
    print("Fail.")

print("There should be two animations in folder")
