# Authors: Ari Herman & Taiyo Terada

# Runs tests on the other files.  Make sure this passes before merging to dev or master!

import sys
sys.path.insert(0,"ClassDefinitions")
import numpy as np
from FlockAnimation import ScatterAnimation 
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation, OlfatiFlockingSimulationTF

###########################
# Set simulation parameters
###########################
sim_params = SimulationParams()
sim_params.set_num_boids(25)
sim_params.set_num_iters(25)


#######
# Tests
#######

# Basic
for run_method in ['NP','TF']:     
    for dim in [2,3]:
        print("Basic test on dimension " + str(dim) + " with " + run_method + ":  ", end ="")
        
        try:
            if run_method == 'NP':
                flock_sim = OlfatiFlockingSimulation()
            
            elif run_method == 'TF':
                flock_sim = OlfatiFlockingSimulationTF()
            
            else:
                assert("Invalid dimension given in test program")
                
            # Set dimension
            sim_params.set_dim(dim)

            # Set gamma path
            sim_params.set_gamma_path('circle')
            
            # Init points
            sim_params.set_q_init('random')
            sim_params.set_p_init('random')
            
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

##########################
# Set animation parameters
##########################
ani_params = AnimationParams()
ani_params.set_save(False)
ani_params.set_show(False)

# Test quiver animation
print("Quiver animation test:  ", end = "")

try:
    flock = ScatterAnimation()
    ani_params.set_save(True)
    ani_params.set_quiver(True)
    ani_params.set_fname("Quiver_test")
    flock.params = ani_params
    flock.setQ(X)
    flock.setP(V)
    flock.initAnimation()
    flock.animate()
    print("Pass!")

except:
    print("Fail.")

# Test scatter animation
print("Scatter animation test:  ", end = "")

try:
    flock.params.quiver = False
    ani_params.set_save(True)
    ani_params.set_quiver(False)
    ani_params.set_fname("Scatter_test")
    flock.setQ(X)
    flock.initAnimation()
    flock.animate()
    print("Pass!")

except:
    print("Fail.")

print("There should be two animations in folder: Quiver_test.mp4 and Scatter_test.mp4")
