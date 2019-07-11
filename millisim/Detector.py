## Detector.py
## methods relating to detector and environment properties
import numpy as np

class PlaneDetector(object):
    def __init__(self, dist_to_origin, eta, phi, width=None, height=None):
        # if width or height are None, detector plane has infinite extent

        self.dist_to_origin = float(dist_to_origin)
        self.eta = float(eta)
        self.phi = float(phi)
        self.width = float(width) if width is not None else None
        self.height = float(height) if height is not None else None

        if eta == float("inf"):
            theta = 0.0
        elif eta == float("-inf"):
            theta = np.pi
        else:
            theta = 2*np.arctan(np.exp(-eta))
        x = dist_to_origin * np.sin(theta) * np.cos(phi)
        y = dist_to_origin * np.sin(theta) * np.sin(phi)
        z = dist_to_origin * np.cos(theta)

        self.center = np.array([x,y,z])
        self.norm = self.center / np.linalg.norm(self.center)

        if np.isinf(eta):
            self.unit_v = np.array([0., 1., 0.])
        else:
            self.unit_v = np.cross(np.array([0., 0., 1.]), self.norm)
        self.unit_v /= np.linalg.norm(self.unit_v)
        self.unit_w = np.cross(self.norm, self.unit_v)
        

    # get the four corners, for drawing purposes
    def GetCorners(self):
        if self.width is None or self.height is None:
            raise Exception("Can't get corners of an infinite detector!")

        c1 = self.center + self.unit_w * self.width/2 + self.unit_v * self.height/2
        c2 = self.center - self.unit_w * self.width/2 + self.unit_v * self.height/2
        c3 = self.center - self.unit_w * self.width/2 - self.unit_v * self.height/2
        c4 = self.center + self.unit_w * self.width/2 - self.unit_v * self.height/2
        
        return c1,c2,c3,c4

    def FindIntersection(self, traj, tvec=None):
        # find the intersection with a plane with normal norm
        # and distance to origin dist. returns None if no intersection

        npoints = traj.shape[1]
        dists = np.sum(np.tile(self.norm, npoints).reshape(npoints,3).T * traj[:3,:], axis=0)
        idx = np.argmax(dists > self.dist_to_origin)

        if idx == 0:
            return None

        p1 = traj[:3,idx-1]
        p2 = traj[:3,idx]
        
        proj1 = np.dot(p1,self.norm)
        proj2 = np.dot(p2,self.norm)
        
        frac = (self.dist_to_origin-proj1)/(proj2-proj1)
        intersect = p1 + frac * (p2-p1)        
            
        v = np.dot(intersect,self.unit_v)
        w = np.dot(intersect,self.unit_w)
            
        if self.width is None or self.height is None or (abs(w) < self.width/2 and abs(v) < self.height/2):
            unit = (p2-p1)/np.linalg.norm(p2-p1)
            theta = np.arccos(np.dot(unit,self.norm))
            
            projW = np.dot(unit,self.unit_w)
            projV = np.dot(unit,self.unit_v)
            
            thW = np.arcsin(projW / np.linalg.norm(unit-projV*self.unit_v))
            thV = np.arcsin(projV / np.linalg.norm(unit-projW*self.unit_w))

            t = None
            if tvec is not None:
                t = tvec[idx-1] + frac * (tvec[idx] - tvec[idx-1])
            
            pInt = traj[3:,idx-1] + frac * (traj[3:,idx] - traj[3:,idx-1])

            return {
                "x_int" : intersect,
                "t_int" : t,
                "p_int" : pInt,
                "v" : v,
                "w" : w,
                "theta" : theta,
                "theta_w" : thW,
                "theta_v" : thV,
                }

        return None

class Box(object):
    def __init__(self, center, norm1, norm2, width, height, depth):
        # width corresponds to direction norm1
        # height corresponds to direction norm2
        # depth corresponds to direction norm1 x norm2

        if abs(np.dot(norm1, norm2)) > 1e-6:
            raise Exception("norm1 and norm2 must be perpendicular!")

        self.center = center
        self.unit_u = norm1 / np.linalg.norm(norm1)
        self.unit_v = norm2 / np.linalg.norm(norm2)
        self.unit_w = np.cross(self.unit_u, self.unit_v)

        self.width = width
        self.height = height
        self.depth = depth

    def get_line_segments(self):
        c1 = self.center - self.depth/2 * self.unit_w - self.width/2 * self.unit_u - self.height/2 * self.unit_v
        c2 = self.center - self.depth/2 * self.unit_w - self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c3 = self.center - self.depth/2 * self.unit_w + self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c4 = self.center - self.depth/2 * self.unit_w + self.width/2 * self.unit_u - self.height/2 * self.unit_v
        c5 = self.center + self.depth/2 * self.unit_w - self.width/2 * self.unit_u - self.height/2 * self.unit_v
        c6 = self.center + self.depth/2 * self.unit_w - self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c7 = self.center + self.depth/2 * self.unit_w + self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c8 = self.center + self.depth/2 * self.unit_w + self.width/2 * self.unit_u - self.height/2 * self.unit_v

        return [
            (c1, c2),
            (c2, c3),
            (c3, c4),
            (c4, c1),
            (c1, c5),
            (c2, c6),
            (c3, c7),
            (c4, c8),
            (c5, c6),
            (c6, c7),
            (c7, c8),
            (c8, c5),
            ]

    def draw(self, ax, **kwargs):
        if "color" not in kwargs and "linecolor" not in kwargs:
            kwargs["color"] = 'k'
        for p1, p2 in self.get_line_segments():
            ax.plot(xs=[p1[0],p2[0]], ys=[p1[1],p2[1]], zs=[p1[2],p2[2]], **kwargs)

# this finds intersection with a spherical shell
# if needed, can implement a SphereDetector class later

# def FindRIntersection(traj, tvec, detectorDict):
#     # find the intersection with a plane with normal norm
#     # and distance to origin dist. returns None if no intersection

#     dist = detectorDict["dist"]

#     for i in range(traj.shape[1]-1):
#         p1 = traj[:3,i]
#         p2 = traj[:3,i+1]
        
#         r2 = np.linalg.norm(p2[:2])

#         if r2>=dist:
#             r1 = np.linalg.norm(p1[:2])
#             intersect = p1+(dist-r1)/(r2-r1)*(p2-p1)
#             t = tvec[i] + (dist-r1)/(r2-r1) * (tvec[i+1] - tvec[i])
#             pInt = traj[3:,i] + (dist-r1)/(r2-r1) * traj[3:,i+1]

#             return intersect,t,None,None,None,pInt

#             break

#     return None, None, None, None, None, None

