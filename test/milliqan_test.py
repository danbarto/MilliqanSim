#! /usr/bin/env python

## Test script to show how to generate particles from a random distribution,
## propagate throught the magnetic field, and simulate their incidence
## on an external Milliqan detector.
##
## For the purposes of visualization, the external detector has been moved
## much closer and enlarged to a 3m x 3m square.
##
## use mode "VIS" to generate a few trajectries and then visualize with matplotlib
## use mode "STATS" to collect statistics on particle hits on the detector, and output
## to a text file
##
## The output format is (Q,m,p,pT,eta,phi,theta,thetaW,thetaV,w,v,pInt), one per column
##  -Q is the charge (program randomly chooses each to be pos/neg)
##  -m is the particle mass
##  -p/pT are the initial (transverse) momentum in MeV
##  -eta/phi are the initial eta/phi of the particle
##  -theta is the of incidence on the detector plane, w.r.t. the normal
##  -w,v are coordinates in the detector plane, where (0,0) is the center.
##  -thetaW, thetaV are the angles of incidence projected on the w,v directions
##  -pInt is the momentum magnitude upon incidence with the detector plane
##
## The above variables are somewhat obscure and were used for testing/debugging.
## To get more useful variables, run the tools/formatOutput.py script.

import math
import time
import os.path
import sys
try:
    sys.path.remove('/home/users/bemarsh/.local/lib/python2.7/site-packages/matplotlib-1.4.3-py2.7-linux-x86_64.egg')
except:
    pass
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import Params
import Integrator
import Detector
import Drawing
from MilliTree import MilliTree
import run_params as rp

# do you want to VISualize, or collect STATS?
mode = rp.mode

if mode=="VIS":
    ntrajs = rp.ntrajs
    trajs = []
if mode=="STATS":
    # in STATS mode, this is the number of hits on the detector to generate
    # the total number of trajectories simulated will be greater
    ntrajs = rp.ntrajs
    trajs = []
    print "Simulating {0} hits on the detector.".format(ntrajs)
    print 

visWithStats = False

suffix = sys.argv[1]

# try:
#     os.makedirs("../output_data")
# except:
#     pass

# outname = "../output_data/test.txt"
outname = "../output_{0}.txt".format(suffix)
outnameCustom = "../customOutput_{0}.txt".format(suffix)

if mode=="STATS":
    print "Outputting to "+outname

# must run at the beginning of main script. Loads the B field map into memory
Detector.LoadCoarseBField("../bfield/bfield_coarse.pkl")

# turn on CMS magnetic field and PDG multiple scattering
Params.BFieldType = rp.BFieldType
Params.MSCtype = 'pdg'
Params.MatSetup='cms'
# turn on dE/dx energy loss (Bethe-Bloch)
Params.EnergyLossOn = True
# charge and mass of the particle. Q in units of electric charge, m in MeV
Params.Q = rp.particleQ
Params.m = rp.particleM
#suppress annoying warnings
Params.SuppressStoppedWarning = False
Params.RockBegins = rp.RockBegins

Params.solRad=1.29
Params.solLength=4.86

if rp.useCustomMaterialFunction:
    Params.matFunction = rp.matFunction

# make sure numbers are new each run
ROOT.gRandom.SetSeed(0)

rootfile = ROOT.TFile(rp.pt_spect_filename)
# this is a 1D pT distribution (taken from small-eta events)
pt_dist = rootfile.Get("h_pt_eta")

dt = rp.dt
nsteps = rp.max_nsteps    

center = rp.centerOfDetector
distToDetect = np.linalg.norm(center)
normToDetect = center/distToDetect

detV = np.array([1.0, 0., 0.])
detW = np.cross(normToDetect, detV)

detWidth = rp.detWidth
detHeight = rp.detHeight
detDepth = rp.detDepth

detectorDict = {"norm":normToDetect, "dist":distToDetect, "v":detV, 
            "w":detW, "width":detWidth, "height":detHeight, "depth":detDepth}

# the four corners (only for drawing)
c1,c2,c3,c4 = Detector.getDetectorCorners(detectorDict)

intersects = []
intersects_all = []
ntotaltrajs = 0

mt = MilliTree()

if mode=="STATS":
    # if file already exists, check if we want to overwrite or append
    if os.path.isfile(outname):
        ow = 'q'
        while ow not in 'yYnN':
            ow = raw_input("Overwrite file? (y/n) ")
        if ow in 'yY':
            txtfile = open(outname,'w')
            if rp.useCustomOutput:
                txtfile2 = open(outnameCustom, 'w')
        else:
            print "OK, appending"
            txtfile = open(outname,'a')
            if rp.useCustomOutput:
                txtfile2 = open(outnameCustom, 'a')
    else:
        txtfile = open(outname,'w')
        if rp.useCustomOutput:
            txtfile2 = open(outnameCustom, 'w')
    txtfile.close()
    if rp.useCustomOutput:
        txtfile2.close()


starttime = time.time()

# loop until we get ntrajs trajectories (VIS) or hits (STATS)
while len(trajs)<ntrajs:
    pt = ROOT.Double(-1)
    eta = ROOT.Double(-1)

    etalow =  rp.etabounds[0]
    etahigh =  rp.etabounds[1]

    # draw random pT values from the distribution.
    while pt < rp.ptCut or not etalow < eta < etahigh:
        pt_dist.GetRandom2(pt, eta)

    if etalow < 0:
        eta *= np.random.randint(2)*2 - 1

    rnd = np.random.rand()
    if rnd < 0.10:
        # proton
        Params.m = 938.3
        pdgId = 2212
    elif rnd < 0.20:
        # kaon
        Params.m = 493.7
        pdgId = 321
    else:
        # pion
        Params.m = 139.6
        pdgId = 211

    th = 2*np.arctan(np.exp(-eta))
    magp = pt / np.sin(th)
    # phi = np.random.rand() * 2*np.pi - np.pi
    phi = np.random.rand() * 2*np.pi
    Params.Q *= np.random.randint(2)*2 - 1 

    # convert to cartesian momentum
    p = 1000*magp * np.array([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)])
    x0 = np.array([0,0,0,p[0],p[1],p[2]])
    
    # simulate until nsteps steps is reached, or the particle passes z=3.1
    traj,tvec = Integrator.rk4(x0, dt, nsteps, cutoff=rp.cutoff, cutoffaxis=rp.cutoffaxis, use_var_dt=rp.use_var_dt)
    ntotaltrajs += 1
    if mode=="VIS":
        trajs.append(traj)

    # print np.linalg.norm(traj[:3,-1]), np.linalg.norm(traj[3:,-1])

    # compute the intersection. Will return None if no intersection
    findIntersection = Detector.FindIntersection
    if rp.useCustomIntersectionFunction:
        findIntersection = rp.intersectFunction
    intersection, t, theta, thW, thV, pInt = findIntersection(traj, tvec, detectorDict)
    intersects_all.append(intersection)
    if intersection is not None:
        intersects.append(intersection)
        print len(trajs), ": p =",magp, ", eta =", eta, ", phi =", phi, ", eff =", float(len(intersects))/ntotaltrajs
        if mode=="VIS":
            pass
        if mode=="STATS":
            if visWithStats:
                trajs.append(traj)
            else:
                trajs.append(0)
            w = np.dot(intersection, detectorDict['w'])
            v = np.dot(intersection, detectorDict['v'])
            magpint = np.linalg.norm(pInt)
            txtfile = open(outname,'a')
            txtfile.write("{0:5d} {1:8.1f} {2:5.1f} {3:10.4f} {4:8.4f} {5:8.4f} {6:10.4f} {7:10.4f} {8:10.4f} {9:10.4f} {10:10.4f} {11:10.4f}\n".format(pdgId,Params.m,Params.Q,magp*np.sin(th),eta,phi,intersection[0],intersection[1],intersection[2],pInt[0]/1000,pInt[1]/1000,pInt[2]/1000))
            txtfile.close()
            if rp.useCustomOutput:
                txtfile = open(outnameCustom, 'a')
                txtfile.write("\t".join(str(x) for x in rp.outputFunction(traj, detectorDict)) + '\n')
                txtfile.close()
            mt.SetValues(intersection, pInt)
            mt.Fill()
            
endtime = time.time()

print "Efficiency:", float(len(intersects))/ntotaltrajs
print "Total time: {0:.2f} sec".format(endtime-starttime)
print "Time/Hit: {0:.2f} sec".format((endtime-starttime)/ntrajs)

# mt.Write("../output_{0}.root".format(suffix))

# fid = ROOT.TFile("../output_{0}.root".format(suffix), "UPDATE")

# hhits = ROOT.TH1F("hhits","",1,0,2)
# hsims = ROOT.TH1F("hsims","",1,0,2)
# hhits.Fill(1, ntrajs)
# hsims.Fill(1, ntotaltrajs)
# hhits.Write()
# hsims.Write()

# fid.Close()

if mode=="VIS" or visWithStats:
    plt.figure(num=1, figsize=(15,7))

    Drawing.Draw3Dtrajs(trajs, subplot=121)

    fout = open("trajs.scad",'w')
    for i,traj in enumerate(trajs):
        fout.write("L{0} = {1};\n".format(i, traj.shape[1]))
        fout.write("a{0} = {1};\n".format(i, str(traj[:3,:].T.tolist())))
        if intersects_all[i] is None:
            fout.write("intersect{0} = [0.0, 0.0, 0.0];\n".format(i))
        else:
            fout.write("intersect{0} = {1};\n".format(i, str(intersects_all[i].tolist())))
        fout.write("pt{0} = {1};\n".format(i, np.linalg.norm(traj[3:5,0])/1000.0))
        fout.write("iEvt{0} = {1};\n\n".format(i, i%3))
    fout.close()

    Drawing.DrawLine(c1,c2,is3d=True)
    Drawing.DrawLine(c2,c3,is3d=True)
    Drawing.DrawLine(c3,c4,is3d=True)
    Drawing.DrawLine(c4,c1,is3d=True)

    for i in range(len(intersects)):
        Drawing.DrawLine(intersects[i],intersects[i],is3d=True,linestyle='None',marker='o',color='r')

    Drawing.DrawXYslice(trajs, subplot=122)

    plt.figure(num=2)
    Drawing.DrawXZslice(trajs)

    # plt.figure(3)
    # rvals = np.linalg.norm(trajs[0][:3,:], axis=0)
    # pvals = np.linalg.norm(trajs[0][3:,:], axis=0)
    # plt.plot(rvals,pvals)

    plt.show()
