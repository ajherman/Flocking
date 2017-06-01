# Authors: Ari Herman & Taiyo Terada

# Displays counter example to lemma 3 from Olfati paper

import sys
sys.path.insert(0,"../ClassDefinitions")
import numpy as np
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation,QuiverAnimation
from matplotlib import pyplot as plt

# WIP
#############################################################

def dist(X,i,j):
    return np.sqrt(np.sum((X[:,i,:]-X[:,j,:])**2,axis=1))

def potential(X):
    ans = 0
    for i in range(4):
        for j in range(i):
            ans += np.abs(np.sqrt(np.sum((X[:,i,:]-X[:,j,:])**2,axis=1))-sim_params.d)
    return ans

################################################################



###########################
# Set simulation parameters
###########################

sim_params = SimulationParams()
sim_params.dim = 2
sim_params.c_p = 0
sim_params.c_q = 0
sim_params.num_boids = 4
sim_params.gamma_path = 'circle'
sim_params.d = 7.0 
sim_params.r = 13.0 
sim_params.d_a = (np.sqrt(1+sim_params.eps*sim_params.d**2)-1)/sim_params.eps
sim_params.r_a = (np.sqrt(1+sim_params.eps*sim_params.r**2)-1)/sim_params.eps
sim_params.num_iters = 1000

################
# Init positions
################

# Square
s = 6.2358 # Equilibrium side length for square
sim_params.q_init = np.array([[0,0],[0,s],[s,0],[s,s]])

# Init velocities (0)
sim_params.p_init = np.zeros((sim_params.num_boids,sim_params.dim))

# Add noise
noise_level=1.0
sim_params.q_init += np.random.normal(0.0,noise_level,size=(sim_params.num_boids,sim_params.dim))

####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.getUserInput()

##################
# Setup simulation
##################

# Make simulation object
flock_sim = OlfatiFlockingSimulation()

# Set simulation parameters
flock_sim.params = sim_params

################
# Run simulation
################

# Init sim
flock_sim.initSim()

# Run sim
X,V = flock_sim.runSim()

# Save simulation array
save_array = input("Do you want to save the simmulation array? [y/n]: ")
if save_array == 'y':
    np.save(ani_params.fname,X)

###########
# Animation
###########

if ani_params.quiver:
    flock = QuiverAnimation(X[:,:-1,:],V[:,:-1,:],ran=2*s)
    
else:
    flock = ScatterAnimation(X[:,:-1,:],ran=2*s)

flock.animate(show=ani_params.show,save=ani_params.save,fname=ani_params.fname)


################
# Plot distances
################
plt.plot(dist(X,0,1))
plt.plot(dist(X,0,2))
plt.plot(dist(X,0,3))
plt.plot(dist(X,1,2))
plt.plot(dist(X,1,3))
plt.plot(dist(X,2,3))
plt.show()

