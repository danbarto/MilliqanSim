from numba import jit, njit, float64, int32, char, boolean
import numpy as np

BFIELD_IDS = {
    "none": 0,
    "cms": 1,
}

MAT_IDS = {
    "justrock": 0,
    "cms": 1,
}

## parameters to load bfield
ZMIN = -1500
ZMAX = 1500
DZ = 10
RMIN = 0
RMAX = 900
DR = 10
PHIMIN = 0
PHIMAX = 355
DPHI = 5

CMS_LENGTH = 15.0
CMS_RADIUS = 3.6
    
@njit(float64[:](float64[:],float64[:]))
def _cross(u, v):
    return np.array([u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]])

@njit(float64[:](float64, float64, float64, int32, float64, float64))
def _get_material(x, y, z, mat_setup, rock_begins, rock_ends):
    withinLength = -CMS_LENGTH/2.0 < z < CMS_LENGTH/2.0
    r = np.sqrt(x**2+y**2)
    R = np.sqrt(r**2+z**2)

    # 0=rock, 1=air, 2=pbwo4, 3=fe
    mat = -1
    
    if rock_begins < R < rock_ends:
        mat = 0  
    else:
        if mat_setup == 1:
            if not withinLength:
                mat = 1    
            elif r < 1.25:
                mat = 1
            elif r < 1.50:
                mat = 2
            elif 1.8 < r < 2.8 or\
                 3.15 < r < 3.5 or\
                 3.85 < r < 4.05 or\
                 4.6 < r < 4.95 or\
                 5.35 < r < 5.95 or\
                 6.35 < r < 7.15:
                mat = 3
            else:
                mat = 1
        else:
            mat = 1

    if mat==0:
        return np.array([11.0, 22.0, 2.65, .1002, 136.4, .08301, 3.4210, .0492, 3.0549, 3.7738, 0.0])
    if mat==1:
        return np.array([7.34, 14.719, 1.205e-3, 3.04e2, 85.7, 0.10914, 3.3994, 1.7418, 4.2759, 10.5961, 0.0])
    if mat==2:
        return np.array([31.3, 75.8, 8.3, 0.008903, 600.7, 0.22758, 3.0, 0.4068, 3.0023, 5.8528, 0.0])
    if mat==3:
        return np.array([26, 55.845, 7.874, .01757, 286.0, 0.14680, 2.9632, -0.0012, 3.1531, 4.2911, 0.12])

    return np.zeros(11)

@njit(float64[:](float64, float64, float64[:], float64, float64[:]))
def _do_energy_loss(m, Q, x, dt, matdef):
    p = x[3:]
    magp = np.linalg.norm(p)
    E = np.sqrt(magp**2 + m**2)
    gamma = E / m
    beta = magp / E;
    me = 0.511;  #electron mass in MeV
    
    Wmax = 2*me*beta**2*gamma**2/(1+2*gamma*me/m + (me/m)**2)
    K = 0.307075  # in MeV cm^2/mol

    Z = matdef[0]
    A = matdef[1]
    rho = matdef[2]
    X0 = matdef[3]
    I = matdef[4]
    a = matdef[5]
    k = matdef[6]
    x0 = matdef[7]
    x1 = matdef[8]
    Cbar = matdef[9]
    delta0 = matdef[10]

    I = I/1e6  ## convert from eV to MeV

    xp = np.log10(magp/m)
    if xp>=x1:
        delta = 2*np.log(10)*xp - Cbar
    elif xp>=x0:
        delta = 2*np.log(10)*xp - Cbar + a*(x1-xp)**k
    else:
        delta = delta0*10**(2*(xp-x0))

    # mean energy loss in MeV/cm
    dEdx = K*rho*Q**2*Z/A/beta**2*(0.5*np.log(2*me*beta**2*gamma**2*Wmax/I**2) - beta**2 - delta/2)

    dE = dEdx * beta*2.9979e1 * dt

    if dE>(E-m):
        return np.array([0, 0, 0, -p[0], -p[1], -p[2]])

    newmagp = np.sqrt((E-dE)**2-m**2)
    dp = p*newmagp/magp - p
    return np.array([0, 0, 0, dp[0], dp[1], dp[2]])

@njit(float64[:](float64[:]))
def _getNormVector(v):
    # generate and return a random vector in the plane orthogonal to v

    nv = np.sqrt(v[0]**2+v[1]**2+v[2]**2)

    if nv == 0:
        return np.zeros(3)

    # angular coords of v
    thetav = np.arccos(v[2]/nv)
    phiv = np.arctan2(v[1],v[0])
    
    angle = 2*np.pi*np.random.rand()
    random_unit = np.array([np.cos(angle), np.sin(angle), 0])

    ct = np.cos(thetav)
    st = np.sin(thetav)
    cp = np.cos(phiv)
    sp = np.sin(phiv)

    # matrix for rotation about y axis by thetav
    rotateY = np.array((( ct, 0, st),
                        (  0, 1,  0),
                        (-st, 0, ct)))

    # matrix for rotation about z axis by phiv
    rotateZ = np.array(((cp, -sp, 0),
                        (sp,  cp, 0),
                        ( 0,   0, 1)))

    random_unit = np.dot(rotateZ, np.dot(rotateY, random_unit))
    # random_unit = random_unit.reshape((3,))

    return random_unit

@njit(float64[:](float64, float64, float64[:], float64, float64[:]))
def _getScatterAnglePDG(m, Q, x, dt, matdef):
    p = x[3:]
    magp = np.linalg.norm(p) # must be in MeV
    E = np.sqrt(magp**2 + m**2)
    v = p/E
    beta = np.linalg.norm(v)

    dx = (beta*2.9979e-1) * dt  # in m
    
    X0 = matdef[3]
    if X0 <= 0:
        return np.zeros(6)

    # rms of projected theta distribution.
    theta0 = 13.6/(beta*magp) * abs(Q) * np.sqrt(dx/X0) * (1 + 0.038*np.log(dx/X0))
    
    # correlation coefficient between theta_plane and y_plane
    rho = 0.87
    
    z1 = np.random.normal()
    z2 = np.random.normal()
    yx = z1*dx*theta0 * np.sqrt((1-rho**2)/3) + z2*rho*dx*theta0/np.sqrt(3)
    thetax = z2*theta0
    
    z1 = np.random.normal()
    z2 = np.random.normal()
    yy = z1*dx*theta0 * np.sqrt((1-rho**2)/3) + z2*rho*dx*theta0/np.sqrt(3)
    thetay = z2*theta0

    return np.array([thetax,thetay,yx,yy])

@njit(float64[:](float64, float64, float64[:], float64, float64[:]))    
def _multiple_scatter_PDG(m, Q, x, dt, matdef):
    # get the angles/displacements from above function and return the
    # net change in x=(x,y,z,px,py,pz)

    if matdef[2] < 1e-2:
        return np.zeros(6)

    p = x[3:]
    vx = _getNormVector(p)
    vy = _cross(vx, p/np.linalg.norm(p))

    vals = _getScatterAnglePDG(m, Q, x, dt, matdef)
    thetax = vals[0]
    thetay = vals[1]
    yx = vals[2]
    yy = vals[3]
    # transverse displacement
    disp = yx*vx + yy*vy
    # deflection in momentum
    defl = np.linalg.norm(p) * (thetax*vx + thetay*vy)

    return np.array((disp[0],disp[1],disp[2], defl[0],defl[1],defl[2]))

@njit(float64[:](float64,float64,float64,float64[:,:,:,:]))
def _get_b(x, y, z, B):
    x *= 100
    y *= 100
    z *= 100
    r = np.sqrt(x**2+y**2)

    if not (z>ZMIN and z<ZMAX and r<RMAX):
        return np.zeros(3)
        
    phi = np.arctan2(y,x) * 180/np.pi
    
    if phi<0:
        phi += 360
            
    nearZ = int(DZ*round(z/DZ))
    nearPHI = int(DPHI*round(phi/DPHI))
        
    if nearPHI==360:
        nearPHI = 0

    iz = (nearZ-ZMIN)/DZ
    iphi = (nearPHI-PHIMIN)/DPHI
    
    ir = r/DR
    irlow = int(np.floor(ir))
    irhigh = int(np.ceil(ir))
    irfrac = ir-irlow
    
    Bvec = irfrac*B[irhigh,iz,iphi,:] + (1-irfrac)*B[irlow,iz,iphi,:]            
    return Bvec

@njit(float64[:](float64, float64, float64, float64[:], int32, float64[:,:,:,:]))
def _dxdt_bfield(m, Q, t, x, bfield_type, B):
    dxdt = np.zeros(6)
    
    p = x[3:]
    magp = np.linalg.norm(p)
    E = np.sqrt(magp**2 + m**2)
    v = p/E
    dxdt[:3] = v * 2.9979e-1

    if bfield_type == 0:
        Bvec = np.zeros(3)
    if bfield_type ==  1:
        Bvec = _get_b(x[0], x[1], x[2], B)

    dxdt[3:] = (89.8755) * Q * _cross(v,Bvec)
    
    return dxdt

@njit(float64[:,:](int32, float64, float64, float64[:], float64, int32, int32, float64[:,:,:,:], 
                   int32, float64, float64, boolean, float64, float64, int32))
def propagate(seed, m, Q, x0, base_dt, nsteps, bfield_type, B, mat_setup, rock_begins, rock_ends, 
              use_var_dt, lowv_dx, cutoff_dist, cutoff_axis):
    np.random.seed(seed)
    x = np.zeros((x0.size+1, nsteps+1))    
    x[:6,0] = x0
    t = 0
    for i in range(nsteps):

        dt = base_dt
        if use_var_dt and i>=1:
            p = np.linalg.norm(x[3:6,i])
            if p < m:
                pOverM = p/m
                beta = pOverM/np.sqrt(1+pOverM**2)
                dt = lowv_dx/(3e-1*beta)

        k1 = _dxdt_bfield(m, Q, t, x[:6,i], bfield_type, B)
        k2 = _dxdt_bfield(m, Q, t+dt/2, x[:6,i]+dt*k1/2, bfield_type, B)
        k3 = _dxdt_bfield(m, Q, t+dt/2, x[:6,i]+dt*k2/2, bfield_type, B)
        k4 = _dxdt_bfield(m, Q, t+dt, x[:6,i]+dt*k3, bfield_type, B)
        dx_Bfield = dt/6. * (k1 + 2*k2 + 2*k3 + k4)

        mat = _get_material(x[0,i], x[1,i], x[2,i], mat_setup, rock_begins, rock_ends)        
        dx_MS = _multiple_scatter_PDG(m, Q, x[:6,i], dt, mat)
        dx_EL = _do_energy_loss(m, Q, x[:6,i], dt, mat)

        t += dt
        x[:6,i+1] = x[:6,i] + dx_Bfield + dx_MS + dx_EL
        x[6,i+1] = t

        isStopped = np.linalg.norm(x[3:6,i] + dx_EL[3:6]) < 1e-6

        if isStopped:
            x[3:6,i+1] *= 0
            return x[:,:i+2]

        if cutoff_dist > 0:
            if cutoff_axis == 3 and np.linalg.norm(x[:3,i+1]) >= cutoff_dist:
                return x[:,:i+2]
            elif cutoff_axis == 4 and np.linalg.norm(x[:2,i+1]) >= cutoff_dist:
                return x[:,:i+2]
            elif cutoff_axis >= 0 and cutoff_axis <= 2:
                if x[cutoff_axis,i+1] >= cutoff_dist:
                    return x[:,:i+2]

    return x
