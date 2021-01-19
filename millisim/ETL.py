import numpy as np

class ETLSensor(object):
    '''
    samllest entity of the ETL detector
    '''
    def __init__(self, dist_to_origin, x, y, width, height):
        # if width or height are None, detector plane has infinite extent
        # width corresponds to eta-hat direction (self.unit_w)
        # height corresonds to phi-hat direction (self.unit_v)

        self.dist_to_origin = float(dist_to_origin)
        #self.eta = float(eta)
        #self.phi = float(phi)
        self.width = float(width) if width is not None else None
        self.height = float(height) if height is not None else None
        self.x = x
        self.y = y

        theta = 0.0
        z = dist_to_origin

        self.center = np.array([x,y,z])
        self.norm = self.center / np.linalg.norm(self.center)

        self.unit_v = np.array([0., 1., 0.])
        self.unit_w = np.array([1., 0., 0.])#cross(self.norm, self.unit_v)
        self.hit = False
        

    # get the four corners, for drawing purposes
    def get_corners(self):
        if self.width is None or self.height is None:
            raise Exception("Can't get corners of an infinite detector!")

        c1 = self.center + self.unit_w * self.width/2 + self.unit_v * self.height/2
        c2 = self.center - self.unit_w * self.width/2 + self.unit_v * self.height/2
        c3 = self.center - self.unit_w * self.width/2 - self.unit_v * self.height/2
        c4 = self.center + self.unit_w * self.width/2 - self.unit_v * self.height/2
        
        return c1,c2,c3,c4

    def get_line_segments(self):
        c1,c2,c3,c4 = self.get_corners()
        return [(c1,c2),(c2,c3),(c3,c4),(c4,c1)]

    def draw(self, ax, is3d=False, isXZ=False, **kwargs):
        if self.width is None or self.height is None:
            raise Exception("Can't draw an infinite detector!")
        if "color" not in kwargs and "c" not in kwargs:
            kwargs["color"] = 'k'
        # NOTE: y and z axes flipped, for consistency with Drawing module (see Drawing.Draw3Dtrajs)
        for p1, p2 in self.get_line_segments():
            if is3d:
                ax.plot(xs=[p1[0],p2[0]], ys=[p1[2],p2[2]], zs=[p1[1],p2[1]], **kwargs)
            elif isXZ:
                ax.plot([p1[2],p2[2]],[p1[0],p2[0]], **kwargs)
            else:
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]], **kwargs)

    def transform_from_detcoords(self, v, w, n=None):
        if n is None:
            n = self.dist_to_origin
        return n*self.norm + v*self.unit_v + w*self.unit_w

    def find_intersection(self, traj, tvec=None):
        # find the intersection with a plane with normal norm
        # and distance to origin dist. returns None if no intersection
        # ETL with a X-Y plane is a bit simpler than the general case

        dists = traj[2, :]
        idx = np.argmax(dists > self.dist_to_origin)

        if idx == 0:
            return None

        p1 = traj[:3,idx-1]
        p2 = traj[:3,idx]
        
        delta = p2 - p1
        frac = delta/delta[2]
        intersect = frac*(self.dist_to_origin-p1[2]) + p1
        
        v = np.dot(intersect,self.unit_v)
        w = np.dot(intersect,self.unit_w)
        if ((self.x - self.width/2) < w < (self.x + self.width/2)) and ((self.y - self.height/2) < v < (self.y+self.height/2)):
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

class ETLFace(object):
    '''
    One face of the ETL detector
    '''
    def __init__(self, dist_to_origin, centers, sensor, offset_x=0, offset_y=0, mirror=False):
        self.dist_to_origin = dist_to_origin
        self.sensor = sensor
        self.centers = centers
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.sensors  = [ ETLSensor(dist_to_origin=dist_to_origin, x=x/1000., y=y/1000., width=sensor.width, height=sensor.height) for x,y in centers  ]
        self.x = np.array([x/1000. for x,y in centers])
        self.y = np.array([y/1000. for x,y in centers])
        if mirror:
            self.sensors += [ ETLSensor(dist_to_origin=dist_to_origin, x=-x/1000., y=y/1000., width=sensor.width, height=sensor.height) for x,y in centers  ]
            self.x = np.append(self.x, [-x/1000. for x,y in centers] )
            self.y = np.append(self.y, [y/1000. for x,y in centers] )
        return
        
    def draw(self, ax, is3d=False, isXZ=False, **kwargs):
        if "color" not in kwargs and "c" not in kwargs:
            kwargs["color"] = 'k'
        try:
            tmpColor = kwargs["color"]
        except KeyError:
            tmpColor = kwargs["c"]
        for sensor in self.sensors:
            if sensor.hit:
                #print "Have a sensor that was hit!"
                kwargs["color"] = 'green'
                #kwargs["fc"] = (1,0,0,0.5)
            else:
                kwargs["color"] = tmpColor
                #kwargs["fc"] = None
            # NOTE: y and z axes flipped, for consistency with Drawing module (see Drawing.Draw3Dtrajs)
            for p1, p2 in sensor.get_line_segments():
                if is3d:
                    fs = 'full' if sensor.hit else 'none'
                    ax.plot(xs=[p1[0],p2[0]], ys=[p1[2],p2[2]], zs=[p1[1],p2[1]], fillstyle=fs, **kwargs)
                elif isXZ:
                    ax.plot([p1[2],p2[2]],[p1[0],p2[0]], **kwargs)
                else:
                    ax.plot([p1[0],p2[0]],[p1[1],p2[1]], **kwargs)
        
    def find_intersection(self, traj, tvec=None):
        dists = traj[2, :]
        idx = np.argmax(dists > self.dist_to_origin)

        if idx == 0:
            return None

        p1 = traj[:3,idx-1]
        p2 = traj[:3,idx]
        
        delta = p2 - p1
        frac = delta/delta[2]
        intersect = frac*(self.dist_to_origin-p1[2]) + p1
        
        #print "intersect", intersect
        
        for sensor in self.sensors:
            v = np.dot(intersect,sensor.unit_v)
            w = np.dot(intersect,sensor.unit_w)
            if ((sensor.x - sensor.width/2) < w < (sensor.x + sensor.width/2)) and ((sensor.y - sensor.height/2) < v < (sensor.y+sensor.height/2)):
                unit = (p2-p1)/np.linalg.norm(p2-p1)
                theta = np.arccos(np.dot(unit,sensor.norm))
            
                projW = np.dot(unit,sensor.unit_w)
                projV = np.dot(unit,sensor.unit_v)
            
                thW = np.arcsin(projW / np.linalg.norm(unit-projV*sensor.unit_v))
                thV = np.arcsin(projV / np.linalg.norm(unit-projW*sensor.unit_w))

                t = None
                if tvec is not None:
                    t = tvec[idx-1] + frac * (tvec[idx] - tvec[idx-1])
            
                pInt = traj[3:,idx-1] + frac * (traj[3:,idx] - traj[3:,idx-1])

                sensor.hit = True
                
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

        return {"x_int" : intersect}
    
    def find_intersection_fast(self, x, y, debug=False):
        
        #return ((self.x-self.sensor.width/2)<x).any() & ((self.x+self.sensor.width/2)>x).any() & ((self.y-self.sensor.height/2)<y).any() & ((self.y+self.sensor.height/2)>y).any()
        for sensor in self.sensors:
            if debug: print(sensor.x, sensor.width, sensor.y, sensor.width)
            if ((sensor.x - sensor.width/2) < x < (sensor.x + sensor.width/2)) and ((sensor.y - sensor.height/2) < y < (sensor.y+sensor.height/2)):
                
                return True

        return False
