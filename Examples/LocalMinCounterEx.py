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

def dist1(X):
    return np.sqrt(np.sum((X[:,0,:]-X[:,1,:])**2,axis=1))

def dist2(X):
    return np.sqrt(np.sum((X[:,0,:]-X[:,2,:])**2,axis=1))

def dist3(X):
    return np.sqrt(np.sum((X[:,0,:]-X[:,3,:])**2,axis=1))

def dist4(X):
    return np.sqrt(np.sum((X[:,1,:]-X[:,2,:])**2,axis=1))

def dist5(X):
    return np.sqrt(np.sum((X[:,1,:]-X[:,3,:])**2,axis=1))

def dist6(X):
    return np.sqrt(np.sum((X[:,2,:]-X[:,3,:])**2,axis=1))



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
sim_params.d = 5.0 #7.0 #5.0
sim_params.r = 10.0 #13.0 #10.0 
sim_params.d_a = (np.sqrt(1+sim_params.eps*sim_params.d**2)-1)/sim_params.eps
sim_params.r_a = (np.sqrt(1+sim_params.eps*sim_params.r**2)-1)/sim_params.eps
sim_params.num_iters = 50000

################
# Init positions
################

# Triangle
s=6.4599
sim_params.q_init = np.array([[0,0],[0,s*np.sqrt(3)/3],[-s/2,-s*np.sqrt(3)/6],[s/2,-s*np.sqrt(3)/6]])

## Square
#s = 6.2358  #4.3471034
#sim_params.q_init = np.array([[0,0],[0,s],[s,0],[s,s]])

# Init velocities (0)
sim_params.p_init = np.zeros((sim_params.num_boids,sim_params.dim))

# Add noise
noise_level=0.00001
sim_params.q_init += np.random.normal(0.0,noise_level,size=(sim_params.num_boids,sim_params.dim))
sim_params.p_init += np.random.normal(0.0,noise_level,size=(sim_params.num_boids,sim_params.dim))

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




# WIP
##################################
# Graphing potential and stuff 
p=potential(X)
plt.plot(p)
plt.show()
print("d1")
print(dist1(X))
print("d2")
print(dist2(X))
print("d3")
print(dist3(X))
print("d4")
print(dist4(X))
print("d5")
print(dist5(X))
print("d6")
print(dist6(X))
#####################################
