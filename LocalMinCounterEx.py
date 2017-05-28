# Authors: Ari Herman & Taiyo Terada

# Displays counter example to lemma 3 from Olfati paper

import numpy as np
from scipy.linalg import norm
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation,QuiverAnimation
from matplotlib import pyplot as plt

def dist1(X):
    return np.sqrt(np.sum((X[:,0,:]-X[:,1,:])**2,axis=1))


def dist2(X):
    return np.sqrt(np.sum((X[:,2,:]-X[:,1,:])**2,axis=1))

def potential(X):
    ans = 0
    for i in range(4):
        for j in range(i):
            ans += np.abs(np.sqrt(np.sum((X[:,i,:]-X[:,j,:])**2,axis=1))-sim_params.d)
    return ans

###########################
# Set simulation parameters
###########################

sim_params = SimulationParams()
sim_params.dim = 2
sim_params.c_p = 0
sim_params.c_q = 0
sim_params.num_boids = 4
sim_params.gamma_path = 'circle'
sim_params.d = 7.0 # 0.8
sim_params.r = 1.2*d # 1.6*d
sim_params.d_a = (np.sqrt(1+sim_params.eps*sim_params.d**2)-1)/sim_params.eps
sim_params.r_a = (np.sqrt(1+sim_params.eps*sim_params.r**2)-1)/sim_params.eps
sim_params.num_iters = 20000

# Init points
d=sim_params.d
sim_params.q_init = np.array([[0,0],[0,d*np.sqrt(3)/3],[-d/2,-d*np.sqrt(3)/6],[d/2,-d*np.sqrt(3)/6]])
sim_params.p_init = np.zeros((sim_params.num_boids,sim_params.dim))

# Add noise
noise_level=0.000001
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
print(np.shape(X))
# Save simulation array
save_array = input("Do you want to save the simmulation array? [y/n]: ")
if save_array == 'y':
    np.save(ani_params.fname,X)

###########
# Animation
###########

if ani_params.quiver:
    flock = QuiverAnimation(X[:,:-1,:],V[:,:-1,:],ran=2*d)
    
else:
    flock = ScatterAnimation(X[:,:-1,:],ran=2*d)

flock.animate(show=ani_params.show,save=ani_params.save,fname=ani_params.fname)



##################################
# Graphing potential and stuff WIP
p=potential(X)
plt.plot(p)
plt.show()
d1=dist1(X)
plt.plot(d1)
plt.show()
d2=dist2(X)
plt.plot(d2)
plt.show()
print(d1)
print(d2)
