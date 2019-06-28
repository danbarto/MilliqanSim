# Params.py
# control all global variables

import Detector
import numpy as np

## NOTE: all numbers for the below two lists are found at http://pdg.lbl.gov/2015/AtomicNuclearProperties/

## these may be updated in main program
Q = 1  ## in units of e
m = 105.658  ## in MeV 
solRad = 3.6  ## in m
solLength = 15.0   ## in m
MSCtype = 'PDG'
EnergyLossOn = False
SuppressStoppedWarning = True
BFieldType = 'CMS'
BFieldUsePickle = True
UseFineBField = False
MatSetup = 'cms'
RockBegins = 999999.  #past this distance from IP, solid concrete
Interpolate = True
# matFunction = Detector.getMaterial

## internal parameters. don't touch
MSCWarning = False

## parameters to load fine bfield
ZMINf = -1500
ZMAXf = 1500
DZf = 1
RMINf = 0
RMAXf = 900
DRf = 1
PHIMINf = 0
PHIMAXf = 355
DPHIf = 5
Bxf = np.array([])
Byf = np.array([])
Bzf = np.array([])
Bmagf = np.array([])
