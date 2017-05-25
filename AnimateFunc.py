import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

class ScatterAnimation():

    def __init__(self,Q):
        
        assert(isinstance(Q,np.ndarray))

        self.num_iters,self.num_points,self.dim = np.shape(Q)
        self.fig = plt.figure()

        if self.dim == 2:
            self.Q = Q
            self.ax = self.fig.add_axes([0, 0, 1, 1])
            self.ax.set_xlim(-2, 2)
            self.ax.set_xticks([])
            self.ax.set_ylim(-2,2)
            self.ax.set_yticks([])
            self.sc = plt.scatter(self.Q[0,:,0],self.Q[0,:,1],s=5)

        elif self.dim == 3:
            self.Q = np.swapaxes(Q,1,2) # Necessary re-ordering of axes

            # Set axes
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim3d([-2,2])
            self.ax.set_ylim3d([-2,2])
            self.ax.set_zlim3d([-2,2])
            
            # Init points
            self.sc = self.ax.scatter(self.Q[0,0],self.Q[0,1],self.Q[0,2],s=5)

        else:
            print("Invalid dimension for animation array")
            assert(False)

    def update(self,num):
        
        if self.dim == 2:
            self.sc.set_offsets(self.Q[num])
        elif self.dim == 3:
            self.sc._offsets3d = self.Q[num]
        else:
            print("Invalid dimension for animation array")
            assert(False)

    def animate(self,fname=None,show=True):

        ani = matplotlib.animation.FuncAnimation(self.fig,self.update,frames=range(self.num_iters),interval=20)
        
        if fname != None:
            ani.save(fname+".mp4",fps=20)
    
        if show:
            plt.show()


class QuiverAnimation():
    
    def __init__(self,Q,P):
        
        assert(isinstance(Q,np.ndarray))
        assert(isinstance(P,np.ndarray))
        assert(np.shape(Q) == np.shape(P))

        self.num_iters,self.num_points,self.dim = np.shape(Q)
        self.fig = plt.figure()

        if self.dim == 2:
            self.Q = Q
            self.P = P
            self.ax = self.fig.add_axes([0, 0, 1, 1])
            self.ax.set_xlim(-2, 2)
            self.ax.set_xticks([])
            self.ax.set_ylim(-2,2)
            self.ax.set_yticks([])
            self.sc = plt.quiver(self.Q[0,:,0],self.Q[0,:,1],self.P[0,:,0],self.P[0,:,1])

        elif self.dim == 3:
            self.Q = np.swapaxes(Q,1,2) # Necessary re-ordering of axes
            self.P = np.swapaxes(P,1,2)
            # Set axes
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim3d([-2,2])
            self.ax.set_ylim3d([-2,2])
            self.ax.set_zlim3d([-2,2])
            
            # Init points
            self.sc = self.ax.quiver(self.Q[0,0],self.Q[0,1],self.Q[0,2],self.P[0,0],self.P[0,1],self.P[0,2])

        else:
            print("Invalid dimension for animation array")
            assert(False)

    def update(self,num):
        
        if self.dim == 2:
            self.sc.set_offsets(self.Q[num])
            self.sc.set_UVC(self.P[num,:,0],self.P[num,:,1])
        elif self.dim == 3:
            self.ax.clear()
            self.ax.quiver(self.Q[num,0],self.Q[num,1],self.Q[num,2],self.P[num,0],self.P[num,1],self.P[num,2])
        else:
            print("Invalid dimension for animation array")
            assert(False)

    def animate(self,fname=None,show=True):

        ani = matplotlib.animation.FuncAnimation(self.fig,self.update,frames=range(self.num_iters),interval=20)
        
        if fname != None:
            ani.save(fname+".mp4",fps=20)
    
        if show:
            plt.show()


