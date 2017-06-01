# Authors: Ari Herman & Taiyo Terada

# Prompts user for input and runs flocking simulation using Algorithm 2 from Olfati paper

import sys
sys.path.insert(0,"ClassDefinitions")
import numpy as np
from FlockAnimation import ScatterAnimation,QuiverAnimation 
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation, OlfatiFlockingSimulationTF


#####################################
# Get simulation parameters from user
#####################################

sim_params = SimulationParams()
sim_params.get_num_boids()
sim_params.get_num_iters()
sim_params.get_dim()
sim_params.get_gamma_path()
sim_params.set_q_init('random')
sim_params.set_p_init('random')
run_method = input("Do you want to run this simulation with Numpy or Tensorflow? ['NP'/'TF']: ")


####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.get_save()
ani_params.get_show()
ani_params.get_quiver()

###########################
# Setup flocking simulation
###########################

if run_method == 'NP':
    flock_sim = OlfatiFlockingSimulation()

elif run_method == 'TF':
    flock_sim = OlfatiFlockingSimulationTF()

else:
    print("Invalid run method.  Must select Numpy or Tensorflow. ['NP'/'TF']")
    assert(False)

# Set simulation parameters
flock_sim.params = sim_params


################
# Run simulation
################

# Init simulation
flock_sim.initSim()

# Run simulation
X,V = flock_sim.runSim()

# Save simulation array?
save_array = input("Do you want to save the simulation array? [y/n]: ") != 'n'
if save_array:
    np.save(ani_params.fname,[X,V])


#########
# Animate
#########

if ani_params.quiver:
    flock = QuiverAnimation(X,V)
    flock.animate(show=ani_params.show,save=ani_params.save,fname=ani_params.fname)

else:
    flock = ScatterAnimation(X)
    flock.animate(show=ani_params.show,save=ani_params.save,fname=ani_params.fname)


