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

###########################
# Set simulation parameters
###########################

sim_params = SimulationParams()
sim_params.set_dim(2)
sim_params.c_p = 0
sim_params.c_q = 0
sim_params.set_num_boids(4)
sim_params.gamma_path = 'circle'
sim_params.set_d(5.0)
sim_params.set_r(10.0)
sim_params.set_num_iters(1000)
sim_params.get_save()
# Triangle
s=6.4599
noise=0.001
sim_params.q_init = np.array([[0,0],[0,s*np.sqrt(3)/3],[-s/2,-s*np.sqrt(3)/6],[s/2,-s*np.sqrt(3)/6]]+np.random.normal(0.0,noise,size=(sim_params.num_boids,sim_params.dim)))
sim_params.p_init = np.zeros((sim_params.num_boids,sim_params.dim))

####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.get_save()
ani_params.set_show(True)
ani_params.get_quiver()

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

###########
# Animation
###########

flock = ScatterAnimation(ran=2*s)
flock.params = ani_params
flock.setQ(X[:,:-1,:])
flock.setP(V[:,:-1,:])
flock.initAnimation()
flock.animate()

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
