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
        self.q_init = None
        self.p_init = None
        self.save = False
        self.fname = None

    def set_epsilon(self,eps):
        self.eps = eps
    def get_epsilon(self):
        eps = input("Enter a value for epsilon: ")
        self.set_epsilon(float(eps))

    # Call only if epsilon is set
    def set_d(self,d):
        self.d = d
        self.d_a = (np.sqrt(1+self.eps*self.d**2)-1)/self.eps
    def get_d(self):
        d = input("Enter a value for d: ")
        self.set_d(float(d))

    # Call only if epsilon is set
    def set_r(self,r):
        self.r = r
        self.r_a = (np.sqrt(1+self.eps*self.r**2)-1)/self.eps
    def get_r(self):
        r = input("Enter a value for r: ")
        self.set_r(float(r))

    def set_dim(self,dim):
        self.dim = dim
    def get_dim(self):
        dim = input("Enter number of dimensions [2/3]: ")
        self.set_dim(int(dim))

    # Call only if dim is set
    def set_gamma_path(self,gamma_path):
        self.gamma_path = gamma_path
    def get_gamma_path(self):
        if self.dim == 2:
            gamma_path = input("Select path for gamma agent ['circle','eight']: ")
        elif self.dim == 3:
            gamma_path = input("Select path for gamma agent ['circle','wild']: ")
        else:
            print("Invalid dimension")
            assert(False)
        self.set_gamma_path(gamma_path)

    def set_num_iters(self,num_iters):
        self.num_iters = num_iters
    def get_num_iters(self):
        num_iters = input("Enter number of interations: ")
        self.set_num_iters(int(num_iters))

    def set_num_boids(self,num_boids):
        self.num_boids = num_boids
    def get_num_boids(self):
        num_boids = input("Enter number of boids: ")
        self.set_num_boids(int(num_boids))
    
    # Call only if dim and num_boids are set
    def set_q_init(self,q_init):
        if q_init is 'random':
            self.q_init = np.random.normal(0.0,1.0,size=(self.num_boids,self.dim))
        else:
            self.q_init = q_init
    def get_q_init(self):
        q_init = input("Enter " + str(self.num_boids) + " positions as a comma-separated list or 'random'")
        if q_init is 'random':
            self._q_init(q_init)
        else:
            self.set_q_init(np.array(q_init))

    # Call only if dim and num_boids are set
    def set_p_init(self,p_init):
        if p_init is 'random':
            self.p_init = np.random.normal(0.0,0.1,size=(self.num_boids,self.dim))
        else:
            self.p_init = p_init
    def get_p_init(self):
        p_init = input("Enter " + str(self.num_boids) + " velocities as a comma-separated list or 'random'")
        if p_init is 'random':
            self.set_p_init(p_init)
        else:
            self.set_p_init(np.array(p_init))
   
    def set_save(self,save):
        self.save = save
    def get_save(self):
        want_to_save = input("Do you want to save the simulation arrays? [y/n]: ")
        if want_to_save != 'n':
            self.set_save(True)
            fname = input("Enter file name with path [path/fname]: ")
            self.set_fname(fname)
        else:
            self.set_save(False)

    def set_fname(self,fname):
        self.fname = fname
#    def getUserInput(self):
#        # Get parameters from user
#        self.num_boids = input("Enter number of boids: ")
#        self.num_iters = input("Enter number of iterations: ")
#        self.dim = input("Enter number of dimensions [2/3]: ")
#        self.num_boids,self.num_iters,self.dim = int(self.num_boids),int(self.num_iters),int(self.dim)
#        if self.dim == 2:
#            self.gamma_path = input("Select path for gamma agent ['circle','eight']: ")
#        
#        elif self.dim == 3:
#            self.gamma_path = input("Select path for gamma agent ['circle','wild']: ")
#        
#        else:
#            print("Invalid dimension")
#            assert(False)
#
#        self.q_init = np.random.normal(0.0,1.0,size=(self.num_boids,self.dim))
#        self.p_init = np.random.normal(0.0,0.1,size=(self.num_boids,self.dim))

##################################
# Stores parameters for animations
##################################
class AnimationParams(Params):

    def __init__(self):
        self.show = True
        self.save = False
        self.fname = None
        self.quiver = False

    def set_show(self,show):
        self.show = show
    def get_show(self):
        want_to_show = input("Do you want to show this animation? [y/n]: ")
        if want_to_show != 'n':
            self.set_show(True)
        else:
            self.set_show(False)

    def set_save(self,save):
        self.save = save
    
    def set_fname(self,fname):
        self.fname = fname

    def get_save(self):
        want_to_save = input("Do you want to save this animation? [y/n]: ")
        if want_to_save != 'n':
            self.set_save(True)
            fname = input("Enter file name [path/filename]: ")
            self.set_fname(fname)
        else:
            self.set_save(False)
    
    def set_quiver(self,quiver):
        self.quiver = quiver
    def get_quiver(self):
        use_quiver = input("Do you want to make a vector animation? [y/n]: ")
        if use_quiver == 'y':
            self.set_quiver(True)
        elif use_quiver == 'n':
            self.set_quiver(False)
        else:
            print("Invalid option")
            assert(False)

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

