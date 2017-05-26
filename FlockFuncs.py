# Flocking functions# Flocking function

import numpy as np

class FlockingSimulation():
    
    def __init__(self):
        pass


class OlfatiFlockingSimulation(FlockingSimulation):

    def __init__(self):
        self.eps=None
        self.num_boids=None
        self.a=None
        self.b=None
        self.c=None
        self.h=None
        self.r_a=None
        self.d_a=None
        self.dt=None

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
        return self.rho_h(z/self.r_a)*self.ph9eful functions
    ##################
    
    def sig_norm(z): # Sigma norm
            return (np.sqrt(1+eps*np.sum(z**2,axis=2).reshape((num_boids,num_boids,1)))-1)/eps
        
        def sig_grad(z,norm=None,eps=eps): # Gradient of sigma norm
                if type(norm) == "NoneType":
                            return z/(1+eps*sig_norm(z))
                            else:
                                        return z/(1+eps*norm)
                                        
                                    def rho_h(z):
                                            return  np.logical_and(z>=0,z<h)+np.logical_and(z<=1,z>=h)*(0.5*(1+np.cos(np.pi*(z-h)/(1-h))))
                                        
                                        def phi(z):
                                                return 0.5*((a+b)*sig_grad(z+c,1)+(a-b))
                                            
                                            def phi_a(z):
                                                    return rho_h(z/r_a)*phi(z-d_a)
                                                    
                                                def differences(q):
                                                        return q[:,None,:] - q
                                                    
                                                    def uUpdate(q,p):
                                                            diff=differences(q)
                                                                norms = sig_norm(diff)
                                                                    diffp=differences(p)
                                                                        return np.sum(phi_a(norms)*sig_grad(diff,norms),axis=0)+np.sum(rho_h(norms/r_a)*diffp,axis=0)
                                                                    
                                                                    def differentiate(v):
                                                                            dv = v.copy()
                                                                                dv[1:]-=v[:-1]
                                                                                    return dv/dt
                                                                                (z-self.d_a)
        
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
