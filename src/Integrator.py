# Integrator.py
# contains a class to perform the numeric integration/propagation

import numpy as np
from Environment import Environment
import MatterInteraction as mi

class Integrator(object):
    def __init__(self, environ, m, Q, dt, nsteps, cutoff_dist=None, cutoff_axis=None, use_var_dt=False, lowv_dx=None,
                 multiple_scatter=None, do_energy_loss=False, update_func=None, suppress_stopped_warn=True):
        # environ: an Environment class object through which to propagate particle
        # m,Q : the mass and charge of the particle to propagate
        # dt, nsteps: timestep (in ns) and maximum number of steps
        # cutoff_dist: stop propagation early once it reaches this distance
        # cutoff_axis: how to measure this distance. 
        #              'x','y','z' for the x,y,z coordinates. 'r' for srqt(x^2+y2). 'R' for sqrt(x^2+y^2+z^2)
        # use_var_dt: if this is True, then use a variable dt when velocity is low (when momentum < mass)
        # lowv_dx: if use_var_dt is set, this is the fixed spatial displacement to use when recomputing dt (in m)
        # multiple_scatter: algorithm to use for multiple scattering. Can be 'none', 'pdg', 'kuhn'
        # do_energy_loss: simulate dE/dx energy loss
        # update func: if you want to specify a custom update function that computes dx/dt at each timestep

        self.environ = environ
        self.m = m
        self.Q = Q
        self.dt = dt
        self.nsteps = nsteps
        self.cutoff_dist = cutoff_dist
        self.cutoff_axis = cutoff_axis
        self.use_var_dt = use_var_dt
        self.lowv_dx = lowv_dx
        if use_var_dt and lowv_dx is None:
            raise Exception("must specify a lowv_dx if using variable dt!")

        self.multiple_scatter = multiple_scatter
        self.do_energy_loss = do_energy_loss

        # if this is None, will use the default self.dxdt_bfield
        # the signature of update_func must be update_func(itg, t, x),
        # where itg is the Integrator class object,
        # and x is a 6-element vector (x,y,z,px,py,pz)
        self.update_func = update_func

        self.suppress_stopped_warn = suppress_stopped_warn


    @property
    def multiple_scatter(self):
        return self.__multiple_scatter

    @multiple_scatter.setter
    def multiple_scatter(self, m):
        if m is None:
            m = 'none'
        m = m.lower()
        if m not in ['none', 'pdg', 'kuhn']:
            raise Exception("Unknown multiple scatter algo type: "+m)
        self.__multiple_scatter = m

    @property
    def cutoff_axis(self):
        return self.__cutoff_axis

    @cutoff_axis.setter
    def cutoff_axis(self, axis):
        if axis is None and self.cutoff_dist is not None:
            raise Exception("Must provided a cutoff_axis if providing a cutoff_dist")
        if axis not in [None, 'x','y','z','r','R']:
            raise Exception("Unknown cutoff axis: "+axis)
        self.__cutoff_axis = axis

    def dxdt_bfield(self, t, x):
        # x is a 6-element vector (x,y,z,px,py,pz)
        # returns dx/dt for propagation through a B-field
        #
        # if B is in Tesla, dt is in ns, p is in units of MeV/c,  then the basic eq is
        # dp/dt = (89.8755) Qv x B,
        
        dxdt = np.zeros(6)

        p = x[3:]
        magp = np.linalg.norm(p)
        E = np.sqrt(magp**2 + self.m**2)
        v = p/E
        dxdt[:3] = v * 2.9979e-1

        B = self.environ.GetBField(x[0],x[1],x[2])

        dxdt[3:] = (89.8755) * self.Q * np.cross(v,B)

        return dxdt

    # 4th order runge-kutta integrator
    def rk4(self, x0):
        # x0 is a vector of initial values e.g. (x0,y0,z0,px0,py0,pz0)
        # return value is an N by nsteps+1 array, where N is the size of x0
        # each column is x at the next time step

        x0 = np.array(x0)
        x = np.zeros((x0.size, self.nsteps+1))
        x[:,0] = x0
        t = 0
        tvec = np.zeros(self.nsteps+1)

        base_dt = self.dt

        # perform the runge-kutta integration
        for i in range(self.nsteps):
            
            dt = base_dt
            if self.use_var_dt and i>=1:
                p = np.linalg.norm(x[3:,i])    
                if p < self.m:
                    pOverM = p/self.m
                    beta = pOverM/np.sqrt(1+pOverM**2)        
                    dt = self.lowv_dx/(3e-1*beta)

            update_func = Integrator.dxdt_bfield if self.update_func is None else self.update_func
            k1 = update_func(self, t, x[:,i])
            k2 = update_func(self, t+dt/2., x[:,i]+dt*k1/2.)
            k3 = update_func(self, t+dt/2., x[:,i]+dt*k2/2.)
            k4 = update_func(self, t+dt, x[:,i]+dt*k3)
            dx_Bfield = dt/6. * (k1 + 2*k2 + 2*k3 + k4)
            #dx_Bfield = dt * k1  # this is simple euler integration

            # add on the effect of MSC if desired
            dx_MS = np.zeros(x0.size)
            if self.multiple_scatter == 'pdg':
                dx_MS = mi.multipleScatterPDG(self, x[:,i], dt)
            elif self.multiple_scatter == 'kuhn':
                dx_MS = mi.multipleScatterKuhn(self, x[:,i], dt)

            t += dt
            x[:,i+1] = x[:,i] + dx_Bfield + dx_MS
            tvec[i+1] = t
            
            if self.do_energy_loss:
                x[:,i+1] = mi.doEnergyLoss(self, x[:,i+1], dt)

            isStopped =  np.all(x[3:,i+1]==0)

            # check if particle has stopped
            if isStopped:
                if not self.suppress_stopped_warn:
                    print "Warning: stopped particle! (initial p ={0:.2f}, at r = {1:.2f})".format(
                        np.linalg.norm(x[3:,0])/1000, np.linalg.norm(x[:3,i+1]))
                return x[:,:i+2], tvec[:i+2]
        
            if self.cutoff_dist is not None:
                if self.cutoff_axis == 'R' and np.linalg.norm(x[:3,i+1]) >= self.cutoff_dist:
                    return x[:,:i+2], tvec[:i+2]
                elif self.cutoff_axis == 'r' and np.linalg.norm(x[:2,i+1]) >= self.cutoff_dist:
                    return x[:,:i+2], tvec[:i+2]
                elif self.cutoff_axis in ['x','y','z']:
                    idx = ['x','y','z'].index(self.cutoff_axis)
                    if x[idx,i+1] >= self.cutoff_dist:
                        return x[:,:i+2], tvec[:i+2]

        
        if self.cutoff_dist is not None:
            print "Warning: cutoff not reached!"

        return x, tvec
