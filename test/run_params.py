import numpy as np

mode = "VIS"
ntrajs = 30
pt_spect_filename = "../p_eta_dist/combined_PtSpect_Eta0p16.root"
dt = 0.1   #timestep in ns
max_nsteps = 5000
cutoff = 5.
use_var_dt = False
bfield_type = "cms"

particleQ = 1.0  # in electron charge units
particleM = 105. # in MEV

distToDetector = 3.
eta = 2.0 # should be different
rock_begins = 16. #??

detWidth = 1.0
detHeight = 1.0
detDepth = 1.0

etabounds = (1.7, 2.8)
ptCut = 17.
phibounds = (-np.pi, np.pi)

useCustomMaterialFunction = False
#useCustomIntersectionFunction = False
useCustomOutput = False
