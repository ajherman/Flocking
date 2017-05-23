import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


class ScatterAnimation():

    def __init__(self,X):
        
        assert(isinstance(X,np.ndarray))

        self.num_iters,self.num_points,self.dim = np.shape(X)
        self.fig = plt.figure()

        if self.dim == 2:
            self.array2animate = X
            self.ax = self.fig.add_axes([0, 0, 1, 1])
            self.ax.set_xlim(-2, 2)
            self.ax.set_xticks([])
            self.ax.set_ylim(-2,2)
            self.ax.set_yticks([])
            self.sc = plt.scatter(X[0,:,0],X[0,:,1],s=5)

        elif self.dim == 3:
            self.array2animate = np.swapaxes(X,1,2) # Necessary re-ordering of axes

            # Set axes
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim3d([-2,2])
            self.ax.set_ylim3d([-2,2])
            self.ax.set_zlim3d([-2,2])
            
            # Init points
            self.sc = self.ax.scatter(self.array2animate[0,0],self.array2animate[0,1],self.array2animate[0,2],s=5)

        else:
            print("Invalid dimension for animation array")
            assert(False)

    def update(self,num):
        
        if self.dim == 2:
            self.sc.set_offsets(self.array2animate[num])
        elif self.dim == 3:
            self.sc._offsets3d = self.array2animate[num]
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

    def __init__(self,X,V):
        
        return 0
