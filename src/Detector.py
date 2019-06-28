## Detector.py
## methods relating to detector and environment properties

import cPickle as pickle
import numpy as np
import Params

def FindIntersection(traj, tvec, detectorDict):
    # find the intersection with a plane with normal norm
    # and distance to origin dist. returns None if no intersection

    norm = detectorDict["norm"]
    dist = detectorDict["dist"]

    for i in range(traj.shape[1]-1):
        p1 = traj[:3,i]
        p2 = traj[:3,i+1]
        
        proj2 = np.dot(p2,norm)

        if proj2>=dist:
            proj1 = np.dot(p1,norm)
            intersect = p1+(dist-proj1)/(proj2-proj1)*(p2-p1)
            t = tvec[i] + (dist-proj1)/(proj2-proj1) * (tvec[i+1] - tvec[i])
            
            vHat = detectorDict["v"]
            wHat = detectorDict["w"]
            center = norm*dist

            w = np.dot(intersect,wHat)
            v = np.dot(intersect,vHat)
            
            if abs(w) < detectorDict["width"]/2 and \
               abs(v) < detectorDict["height"]/2:
                unit = (p2-p1)/np.linalg.norm(p2-p1)
                theta = np.arccos(np.dot(unit,norm))
                
                projW = np.dot(unit,wHat)
                projV = np.dot(unit,vHat)

                thW = np.arcsin(projW/np.linalg.norm(unit-projV*vHat))
                thV = np.arcsin(projV/np.linalg.norm(unit-projW*wHat))

                # momentum when it hits detector
                pInt = traj[3:,i]

                return intersect,t,theta,thW,thV,pInt

            break

    return None, None, None, None, None, None

def FindRIntersection(traj, tvec, detectorDict):
    # find the intersection with a plane with normal norm
    # and distance to origin dist. returns None if no intersection

    dist = detectorDict["dist"]

    for i in range(traj.shape[1]-1):
        p1 = traj[:3,i]
        p2 = traj[:3,i+1]
        
        r2 = np.linalg.norm(p2[:2])

        if r2>=dist:
            r1 = np.linalg.norm(p1[:2])
            intersect = p1+(dist-r1)/(r2-r1)*(p2-p1)
            t = tvec[i] + (dist-r1)/(r2-r1) * (tvec[i+1] - tvec[i])
            pInt = traj[3:,i] + (dist-r1)/(r2-r1) * traj[3:,i+1]

            return intersect,t,None,None,None,pInt

            break

    return None, None, None, None, None, None


def getMilliqanDetector(distance=33.0, width=1.):
    
    # normal to the plane. Put it 9 m away on the x-axis
    normToDetect = np.array([1,0,0])
    # distance from origin to plane
    distToDetect = distance
    
    #center point
    center = normToDetect*distToDetect
    # the detector definition requires 2 orthogonal unit vectors (v,w) in the
    # plane of the detector. This serves to orient the detector in space
    # and sets up a coordinate system that is used in the output
    detV = np.array([0,1,0])
    detW = np.cross(normToDetect,detV)
    
    detWidth = width
    detHeight = width
    
    # this dictionary is passed to the FindIntersection method below
    return {"norm":normToDetect, "dist":distToDetect, "v":detV, 
            "w":detW, "width":detWidth, "height":detHeight}

# get the four corners, for drawing purposes
def getDetectorCorners(ddict):
    center = ddict['norm']*ddict['dist']
    w = ddict['w']
    v = ddict['v']
    width = ddict['width']
    height = ddict['height']

    c1 = center + w*width/2 + v*height/2
    c2 = center + w*width/2 - v*height/2
    c3 = center - w*width/2 - v*height/2
    c4 = center - w*width/2 + v*height/2

    return c1,c2,c3,c4








