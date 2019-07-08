import Detector
import numpy as np

mode = "VIS"
ntrajs = 200
pt_spect_filename = "../p_eta_dist/TT_chargedParticles_pt_eta.root"
dt = 0.2   #timestep in ns
max_nsteps = 500
use_var_dt = False
BFieldType = "cms"

particleQ = 1.0  # in electron charge units
particleM = 135. # in MEV
##
RockBegins = 99999.

# cutoff = 3.1
# cutoffaxis = 2
# distToDetector = 2.99
# centerOfDetector = np.array([0., 0., distToDetector])
# detWidth = 1.5 * 2
# detHeight = 1.5 * 2
# detDepth = 1.0
# etabounds = (1.2, 3.0)
# ptCut = 0.2
# useCustomIntersectionFunction = False

cutoff = 1.20
cutoffaxis = 4
distToDetector = 1.16
centerOfDetector = np.array([0., distToDetector, 0.])
detWidth = 1.29*2
detHeight = 6.0
detDepth = 1.0
etabounds = (-1.4,1.4)
ptCut = 0.70
useCustomIntersectionFunction = True
intersectFunction = Detector.FindRIntersection

useCustomMaterialFunction = False
useCustomOutput = False
