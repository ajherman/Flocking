# Authors: Ari Herman & Taiyo Terada

import numpy as np

#############################################
# Class stores parameters and gets user input
#############################################
class Params():

    def __init__(self):
        # Initialize all parameters
        pass

    def getUserInput(self):
        # Prompt user to set some parameters
        pass

#################################
# Parameters for flock simulation
#################################
class SimulationParams(Params):

    def __init__(self):
        self.eps = 0.1
        self.d=0.7 
        self.r=1.2*self.d
        self.d_a = (np.sqrt(1+self.eps*self.d**2)-1)/self.eps
        self.r_a = (np.sqrt(1+self.eps*self.r**2)-1)/self.eps
        self.a=5
        self.b=5
        self.c=.5*np.abs(self.a-self.b)/np.sqrt(self.a*self.b)
        self.h=.2 #0<h<1
        self.dt=0.01
        self.c_q=10
        self.c_p=5
        self.num_boids = None
        self.num_iters = None
        self.dim = None
        self.gamma_path = None

    def getUserInput(self):
        # Get parameters from user
        self.num_boids = input("Enter number of boids: ")
        self.num_iters = input("Enter number of iterations: ")
        self.dim = input("Enter number of dimensions [2/3]: ")
        self.num_boids,self.num_iters,self.dim = int(self.num_boids),int(self.num_iters),int(self.dim)
        if self.dim == 2:
            self.gamma_path = input("Select path for gamma agent ['circle','eight']: ")
        
        elif self.dim == 3:
            self.gamma_path = input("Select path for gamma agent ['circle','wild']: ")
        
        else:
            print("Invalid dimension")
            assert(False)

##################################
# Stores parameters for animations
##################################
class AnimationParams(Params):

    def __init__(self):
        self.show = True
        self.save = False
        self.fname = None
        self.quiver = False

    def getUserInput(self):
        self.save = input("Do you want to save this animation [y/n]: ")

        if self.save=='y':
            self.save = True
            self.fname = input("Type file name [no extension]: ")
        
        else:
            self.save = False
            self.fname = None

        display = input("Do you want to show this animation [y/n]: ")
        if display == 'y':
            self.show = True
            quiver_display = input("Do you want a quiver animation [y/n]: ")
            if quiver_display == 'y':
                self.quiver = True

