## Detector.py
## methods relating to detector and environment properties
import numpy as np

class PlaneDetector(object):
    def __init__(self, dist_to_origin, eta, phi, width=None, height=None):
        # if width or height are None, detector plane has infinite extent
        # width corresponds to eta-hat direction (self.unit_w)
        # height corresonds to phi-hat direction (self.unit_v)

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
        # width corresponds to direction norm1 (unit_u)
        # height corresponds to direction norm2 (unit_v)
        # depth corresponds to direction norm1 x norm2 (unit_w)

        if abs(np.dot(norm1, norm2)) > 1e-6:
            raise Exception("norm1 and norm2 must be perpendicular!")

        self.center = center
        self.unit_u = norm1 / np.linalg.norm(norm1)
        self.unit_v = norm2 / np.linalg.norm(norm2)
        self.unit_w = np.cross(self.unit_u, self.unit_v)
        self.unit_w /= np.linalg.norm(self.unit_w)

        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)

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
        if "color" not in kwargs and "c" not in kwargs:
            kwargs["color"] = 'k'
        for p1, p2 in self.get_line_segments():
            ax.plot(xs=[p1[0],p2[0]], ys=[p1[1],p2[1]], zs=[p1[2],p2[2]], **kwargs)

    def contains(self,p):
        rel = p - self.center
        if not -self.width/2 < np.dot(rel, self.unit_u) < self.width/2:
            return False
        if not -self.height/2 < np.dot(rel, self.unit_v) < self.height/2:
            return False
        if not -self.depth/2 < np.dot(rel, self.unit_w) < self.depth/2:
            return False
        return True


class MilliqanDetector(object):
    def __init__(self, dist_to_origin, eta, phi, 
                 nrows=3, ncols=2, nlayers=3,
                 bar_width = 0.05, bar_height = 0.05, bar_length = 0.86,
                 bar_gap = 0.01, layer_gap = 0.30):

        width = ncols*bar_width + (ncols-1)*bar_gap
        height = nrows*bar_height + (nrows-1)*bar_gap
        self.face = PlaneDetector(dist_to_origin, eta, phi, width, height)

        self.__nrows = nrows
        self.__ncols = ncols
        self.__nlayers = nlayers
        self.__bar_width = bar_width
        self.__bar_height = bar_height
        self.__bar_length = bar_length
        self.__bar_gap = bar_gap
        self.__layer_gap = layer_gap

        mid_layer = (self.nlayers-1)/2.0
        self.center_3d = self.face.center + \
                         ((mid_layer+0.5)*bar_length + mid_layer*layer_gap) * self.face.norm

        self.total_length = self.nlayers*self.bar_length + (self.nlayers-1)*self.layer_gap
        self.containing_box = Box(self.center_3d, self.face.unit_w, self.face.unit_v, 
                                  width, height, self.total_length)

        # bars is an (nlayers x nrows x ncols) array of Box objects
        # counting from near layer to far, top row to bottom, left col to right
        self.bars = []
        for ilayer in range(nlayers):
            self.bars.append([])
            layer = self.bars[-1]
            for irow in range(nrows):
                layer.append([])
                row = layer[-1]
                rows_from_center = -(irow - (nrows-1)/2.0) # negative to start from bottom
                for icol in range(ncols):
                    cols_from_center = icol - (ncols-1)/2.0
                    center = self.face.center + \
                             ((ilayer+0.5)*bar_length + ilayer*layer_gap) * self.face.norm + \
                             rows_from_center*(bar_height+bar_gap) * self.face.unit_v +\
                             cols_from_center*(bar_width+bar_gap) * self.face.unit_w
                    row.append(Box(
                        center = center,
                        norm1 = self.face.unit_w,
                        norm2 = self.face.unit_v,
                        width = bar_width,
                        height = bar_height,
                        depth = bar_length
                    ))

    def draw(self, ax, draw_containing_box=False, **kwargs):
        for ilayer in range(self.nlayers):
            for irow in range(self.nrows):
                for icol in range(self.ncols):
                    self.bars[ilayer][irow][icol].draw(ax, **kwargs)
        if draw_containing_box:
            kwargs['c'] = 'r'
            self.containing_box.draw(ax, **kwargs)

    def FindEntriesExits(self, traj):
        # returns a list of tuples
        #  ((layer,row,col), entry_point, exit_point)
        # None if no intersections

        npoints = traj.shape[1]
        dists = np.sum(np.tile(self.face.norm, npoints).reshape(npoints,3).T * traj[:3,:], axis=0)
        start_idx = np.argmax(dists > self.face.dist_to_origin)
        end_idx = np.argmax(dists > self.face.dist_to_origin + self.nlayers*self.bar_length + (self.nlayers-1)*self.layer_gap)

        points = []

        last_isin = None
        for i in range(start_idx, end_idx+1):
            d = dists[i]
            ilayer = int(np.floor((d-self.face.dist_to_origin)/(self.bar_length+self.layer_gap)))
            if not self.containing_box.contains(traj[:3,i]):
                ilayer = -1
            isin = None
            if 0 <= ilayer < self.nlayers:
                for irow in range(self.nrows):
                    for icol in range(self.ncols):
                        if self.bars[ilayer][irow][icol].contains(traj[:3,i]):
                            isin = (ilayer, irow, icol)
                            break
                    if isin is not None:
                        break

            if isin != last_isin:
                if last_isin is not None:
                    exit_point = traj[:3,i-1]
                    points.append((last_isin, entry_point, exit_point))
                if isin is not None:
                    entry_point = traj[:3,i]

            last_isin = isin

        return points

    def hits_straight_line(self, isects):
        layer_hits = np.zeros((self.nlayers,self.nrows,self.ncols))
        for isect in isects:
            layer_hits[isect[0][0],isect[0][1],isect[0][2]] = 1
        nlayers = np.sum(layer_hits, axis=0)
        if np.amax(nlayers) == self.nlayers:
            return True
        return False

    @property
    def nrows(self):
        return self.__nrows
    @nrows.setter
    def nrows(self, _):
        print "Can't change nrows after initialization"
    @property
    def ncols(self):
        return self.__ncols
    @ncols.setter
    def ncols(self, _):
        print "Can't change ncols after initialization"
    @property
    def nlayers(self):
        return self.__nlayers
    @nlayers.setter
    def nlayers(self, _):
        print "Can't change nlayers after initialization"
    @property
    def bar_width(self):
        return self.__bar_width
    @bar_width.setter
    def bar_width(self, _):
        print "Can't change bar_width after initialization"
    @property
    def bar_height(self):
        return self.__bar_height
    @bar_height.setter
    def bar_height(self, _):
        print "Can't change bar_height after initialization"
    @property
    def bar_length(self):
        return self.__bar_length
    @bar_length.setter
    def bar_length(self, _):
        print "Can't change bar_length after initialization"
    @property
    def bar_gap(self):
        return self.__bar_gap
    @bar_gap.setter
    def bar_gap(self, _):
        print "Can't change bar_gap after initialization"
    @property
    def layer_gap(self):
        return self.__layer_gap
    @layer_gap.setter
    def layer_gap(self, _):
        print "Can't change layer_gap after initialization"

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

