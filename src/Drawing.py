## Drawing.py
## contains various routines to visualize trajectories

import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment

## 3d view

def Draw3Dtrajs(trajs, colors=None, ax = None, fig=None, subplot=111):
    # trajs is a list of trajectory arrays as returned by the Integrator.rk4 routine
    # colors is an optional array of colors to use
    # ax is the mpl axes to plot to. If None, createsnew axes
    # fig is the figure to use (if ax not specified). Defaults to plt.gcf()
    # subplot is the subplot to plot to

    from mpl_toolkits.mplot3d import Axes3D

    if ax==None:
        if fig==None:
            fig = plt.gcf()
        ax = fig.add_subplot(subplot, projection='3d')

    if colors==None:
        colors = ['r','g','b','c','m','y']
        # colors = ['r','c','y']

    nc = len(colors)
        
    for i in range(len(trajs)):
        ax.plot3D(xs=trajs[i][0,:], ys=trajs[i][1,:], zs=trajs[i][2,:], color=colors[i%nc])

    sr, sl = Environment.CMS_RADIUS, Environment.CMS_LENGTH
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(xs=sr*np.cos(t), ys=sr*np.sin(t), zs=sl/2, color='k')
    ax.plot(xs=sr*np.cos(t), ys=sr*np.sin(t), zs=-sl/2, color='k')
    for i in range(8):
        th = i * 2*np.pi/8
        x = sr*np.cos(th)
        y = sr*np.sin(th)
        ax.plot(xs=[x,x], ys=[y,y], zs=[-sl/2, sl/2], color='k')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    ax.set_xlim((-9,9))
    ax.set_ylim((-9,9))
    ax.set_zlim((-15,15))



def DrawXYslice(trajs, colors=None, ax=None, fig=None, subplot=111):
    # draws projected trajectories in z=0 plane
    # see above for argument descriptions

    if ax == None:
        if fig==None:
            fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    if colors==None:
        colors = ['r','g','b','c','m','y']

    nc = len(colors)

    # draw solenoid outline
    sr = Environment.CMS_RADIUS
    sl = Environment.CMS_LENGTH
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(sr*np.cos(t),sr*np.sin(t), '-')

    # draw trajectory
    for i in range(len(trajs)):
        ax.plot(trajs[i][0,:],trajs[i][1,:],'-', linewidth=2, color=colors[i%nc])

    ax.set_xlim((-9,9))
    ax.set_ylim((-9,9))

def DrawXZslice(trajs, colors=None, ax=None, fig=None, subplot=111, drawBField=None):
    # draws projected trajectories in z=0 plane
    # see above for argument descriptions

    if ax == None:
        if fig==None:
            fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    if colors==None:
        colors = ['r','g','b','c','m','y']

    if drawBField==None:
        if Params.BFieldLoaded:
            drawBField = True
        else:
            drawBField = False

    nc = len(colors)

    # draw B field
    x = np.arange(-Params.RMAX, Params.RMAX+1e-10, Params.DR)/100
    z = np.arange(Params.ZMIN, Params.ZMAX+1e-10, Params.DZ)/100
    Z,X = np.meshgrid(z,x)

    mag = np.append(Params.Bmag[::-1,:,0],Params.Bmag[1:,:,180/Params.DPHI],0)
    bmplot = plt.pcolor(Z,X,mag,cmap='afmhot',vmax=5.0)

    # draw trajectories
    for i in range(len(trajs)):
        plt.plot(trajs[i][2,:],trajs[i][0,:],'-', linewidth=2, color=colors[i%nc])

    plt.xlabel("z (m)")
    plt.ylabel("x (m)")

    ax.set_xlim((-15,15))
    ax.set_ylim((-9,9))

def DrawLine(p1, p2, ax=None, is3d=False, color='k', linestyle='-', marker='None'):
    # draws a line connecting 2 points

    if ax==None:
        ax = plt.gca()

    if is3d:
        ax.plot(xs=[p1[0],p2[0]],ys=[p1[1],p2[1]],zs=[p1[2],p2[2]], color=color, linestyle=linestyle, marker=marker)

    if not is3d:
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=color,linestyle=linestyle, marker=marker)
    

