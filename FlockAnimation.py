# Authors: Ari Herman & Taiyo Terada

import numpy as np
from scipy.linalg import norm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

#################################
# General purpose animation class
#################################
class Animation():
    def __init__(self):
        pass

    def update(self):
        pass

    def animate(self):
        pass

#########################
# Scatter plot animations
#########################
class ScatterAnimation(Animation):
    def __init__(self,Q):
        assert(isinstance(Q,np.ndarray))
        
        self.num_iters,self.num_points,self.dim = np.shape(Q)
        
        # Set up figure
        self.fig = plt.figure()

        if self.dim == 2:
            self.Q = Q

            # Set axes
            self.ax = self.fig.add_axes([0, 0, 1, 1])
            self.ax.set_xlim(-2, 2)
            self.ax.set_xticks([])
            self.ax.set_ylim(-2,2)
            self.ax.set_yticks([])

            # Plot init points
            self.sc = plt.scatter(self.Q[0,:,0],self.Q[0,:,1],s=5)

        elif self.dim == 3:
            self.Q = np.swapaxes(Q,1,2) # Necessary re-ordering of axes

            # Set axes
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim3d([-2,2])
            self.ax.set_ylim3d([-2,2])
            self.ax.set_zlim3d([-2,2])
            
            # Plot init points
            self.sc = self.ax.scatter(self.Q[0,0],self.Q[0,1],self.Q[0,2],s=5)

        else:
            print("Invalid dimension for animation array")
            assert(False)

    def update(self,num): # Update function for matplotlib funcAnimation
        if self.dim == 2:
            self.sc.set_offsets(self.Q[num])
        
        elif self.dim == 3:
            self.sc._offsets3d = self.Q[num]
        
        else:
            print("Invalid dimension for animation array")
            assert(False)

    def animate(self,show=True,save=False,fname=None): # Display and save animation
        ani = matplotlib.animation.FuncAnimation(self.fig,self.update,frames=range(self.num_iters),interval=20)
        
        if save and fname != None:
            ani.save(fname+".mp4",fps=20)
    
        if show:
            plt.show()

###################
# Vector animations
###################
class QuiverAnimation(Animation):
    def __init__(self,Q,P):
        assert(isinstance(Q,np.ndarray))
        assert(isinstance(P,np.ndarray))
        assert(np.shape(Q) == np.shape(P))
        
        self.num_iters,self.num_points,self.dim = np.shape(Q)

        # Setup figure
        self.fig = plt.figure()

        if self.dim == 2:
            self.Q = Q
            self.P = P
            self.P = 0.01*self.P/norm(self.P,axis=2,keepdims=True) # Normalize P
            
            # Set axes
            self.ax = self.fig.add_axes([0, 0, 1, 1])
            self.ax.set_xlim(-2, 2)
            self.ax.set_xticks([])
            self.ax.set_ylim(-2,2)
            self.ax.set_yticks([])

            # Plot init points
            self.sc = plt.quiver(self.Q[0,:,0],self.Q[0,:,1],self.P[0,:,0],self.P[0,:,1])

        elif self.dim == 3:
            self.Q = np.swapaxes(Q,1,2) # Necessary re-ordering of axes
            self.P = np.swapaxes(P,1,2)
            self.P = 0.01*self.P/norm(self.P,axis=2,keepdims=True) # Normalize P
            
            # Set axes
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim3d([-2,2])
            self.ax.set_ylim3d([-2,2])
            self.ax.set_zlim3d([-2,2])
            
            # Plot init points
            self.sc = self.ax.quiver(self.Q[0,0],self.Q[0,1],self.Q[0,2],self.P[0,0],self.P[0,1],self.P[0,2],length=0.2,lw=1)

        else:
            print("Invalid dimension for animation array")
            assert(False)

    def update(self,num): # Update function for matplotlib funcAnimation
        if self.dim == 2:
            self.sc.set_offsets(self.Q[num])
            self.sc.set_UVC(self.P[num,:,0],self.P[num,:,1])
        
        elif self.dim == 3:
            # Necessary to reset axes each frame due to issue with quiver when in 3D (has not set_data methods)
            self.ax.clear()
            self.ax.set_xlim3d([-2,2])
            self.ax.set_ylim3d([-2,2])
            self.ax.set_zlim3d([-2,2])
            self.ax.quiver(self.Q[num,0],self.Q[num,1],self.Q[num,2],self.P[num,0],self.P[num,1],self.P[num,2],length=0.2,lw=1)
        
        else:
            print("Invalid dimension for animation array")
            assert(False)

    def animate(self,show=True,save=False,fname=None): # Display and save animation
        ani = matplotlib.animation.FuncAnimation(self.fig,self.update,frames=range(self.num_iters),interval=20)
        
        if save and fname != None:
            ani.save(fname+".mp4",fps=20)
    
        if show:
            plt.show()

