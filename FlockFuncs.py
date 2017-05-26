# Flocking simulation objects store all of the functions/data needed to run a simulation

import numpy as np

class FlockingSimulation():
    
    def __init__(self):
        pass
    
    def initSim(self):
        pass

    def runSim(self):
        pass


class OlfatiFlockingSimulation(FlockingSimulation):

    def __init__(self):

        self.eps=None
        self.num_boids=None
        self.num_iters=None
        self.gamma_path=None
        self.dim=None
        self.a=None
        self.b=None
        self.c=None
        self.h=None
        self.r_a=None
        self.d_a=None
        self.dt=None
        self.q=None
        self.p=None
        self.q_g=None
        self.p_g=None
        self.c_q=None
        self.c_p=None
    def sig_norm(self,z): # Sigma norm
        return (np.sqrt(1+self.eps*np.sum(z**2,axis=2).reshape((self.num_boids,self.num_boids,1)))-1)/self.eps
    
    def sig_grad(self,z,norm=None): # Gradient of sigma norm
        if type(norm) == "NoneType":
            return z/(1+self.eps*self.sig_norm(z))
        else:
            return z/(1+self.eps*norm)
        
    def rho_h(self,z):
        return  np.logical_and(z>=0,z<self.h)+np.logical_and(z<=1,z>=self.h)*(0.5*(1+np.cos(np.pi*(z-self.h)/(1-self.h))))

    def phi(self,z):
        return 0.5*((self.a+self.b)*self.sig_grad(z+self.c,1)+(self.a-self.b))

    def phi_a(self,z):
        return self.rho_h(z/self.r_a)*self.phi(z-self.d_a)

    def differences(self,q):
        return q[:,None,:] - q

    def uUpdate(self,q,p):
        diff=self.differences(q)
        norms = self.sig_norm(diff)
        diffp=self.differences(p)
        return np.sum(self.phi_a(norms)*self.sig_grad(diff,norms),axis=0)+np.sum(self.rho_h(norms/self.r_a)*diffp,axis=0)

    def differentiate(self,v):
        dv = v.copy()
        dv[1:]-=v[:-1]
        return dv/self.dt

    def makeGamma(self):

        # Generate trajectory
        if self.dim == 2:
            if self.gamma_path == "circle":
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.sin(np.linspace(0,2*np.pi,self.num_iters))

            elif self.gamma_path == "eight":
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.sin(np.linspace(0,4*np.pi,self.num_iters))
            
            else:
                print("Not a valid gamma agent path for dimension 2")
                assert(False)

            self.q_g=np.stack((x,y),axis=1) 
            self.p_g=np.stack((self.differentiate(x),self.differentiate(y)),axis=1)

        elif self.dim ==3:
            if self.gamma_path == "circle":    
                # Gamma agent (moves in a circle)
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.sin(np.linspace(0,2*np.pi,self.num_iters))
                z=np.zeros(self.num_iters)
            
            elif self.gamma_path == "wild":
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.cos(np.linspace(0,4*np.pi,self.num_iters))
                z=np.sin(np.linspace(0,8*np.pi,self.num_iters))
            else:
                print("Not a valid gamma agent path for dimension 3")
                assert(False)

            self.q_g=np.stack((x,y,y),axis=1)
            self.p_g=np.stack((self.differentiate(x),self.differentiate(y),self.differentiate(z)),axis=1)
        else:
            print("Invalid dimension")
            assert("False")

    def initSim(self):        
        # Random init boids 
        self.q=np.random.normal(0.0,1.0,size=(self.num_boids,self.dim))
        self.p=0.01*np.random.rand(self.num_boids,self.dim)
        # Init gamma agent
        self.makeGamma()        

    def runSim(self):
        X = np.zeros((self.num_iters,self.num_boids,self.dim))
        V = np.zeros((self.num_iters,self.num_boids,self.dim))
        for i in range(self.num_iters):
            z=self.uUpdate(self.q,self.p)
            self.q+=self.p*self.dt
            X[i,:,:] = self.q
            V[i,:,:] = self.p
            self.p+=(z-self.c_q*(self.q-self.q_g[i])-self.c_p*(self.p-self.p_g[i]))*self.dt

        # Add the gamma agent
        X = np.concatenate((X,self.q_g[:,None,:]),axis=1) 
        V = np.concatenate((V,self.p_g[:,None,:]),axis=1)

        return X,V
            

