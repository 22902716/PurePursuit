import numpy as np
import math
from dataSave import dataSave

mu = 0.
sigma = 0.2


class PurePursuitPlanner:
    def __init__(self, map_name, testmode, param, wb = 0.35):
        self.wheelbase = wb
        
        self.map_name = map_name
        self.TESTMODE = testmode
        self.waypoints = np.loadtxt('./maps/'+self.map_name+'_'+"raceline"+'.csv', delimiter=",")
        self.ego_index = None
        self.Tindx = None
        self.scale = 0.
        self.saveFlag = True

        '''
        load parameter figure out how to use yaml
        '''
        if self.TESTMODE == "Benchmark" or self.TESTMODE == " ":
            self.v_gain = param.Benchmark_v_gain                #change this parameter for different tracks 
            self.lfd = param.Benchmark_lfd      
            self.Max_iter = param.Benchmark_Max_iter               
        elif self.TESTMODE == "perception_noise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
            self.v_gain = param.noise_v_gain                #change this parameter for different tracks 
            self.lfd = param.noise_lfd                     #lood forward distance constant
            self.Max_iter = param.noise_Max_iter
        elif self.TESTMODE == "v_gain":
            self.v_gain = param.gain_tune_v_gain                #change this parameter for different tracks 
            self.lfd = param.gain_tune_lfd                     #lood forward distance constant
            self.Max_iter = param.gain_tune_Max_iter
        elif self.TESTMODE == "lfd":
            self.v_gain = param.lfd_tune_v_gain                #change this parameter for different tracks 
            self.lfd = param.lfd_tune_lfd                     #lood forward distance constant
            self.Max_iter = param.lfd_tune_Max_iter
        elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_Delay_steering" or self.TESTMODE == "perception_delay":
            self.v_gain = param.delay_v_gain                #change this parameter for different tracks 
            self.lfd = param.delay_lfd                     #lood forward distance constant
            self.Max_iter = param.delay_Max_iter

        self.ds = dataSave(testmode,map_name,self.Max_iter)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        self.drawn_waypoints = []
        self.points = np.vstack((self.waypoints[:,1],self.waypoints[:,2])).T
        scaled_points = 50.*self.points
        for i in range(self.points.shape[0]):
            if len(self.drawn_waypoints) < self.points.shape[0]:
                b = e.batch.add(1, 0, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

        current_index = 0 if self.Tindx is None else self.Tindx
        b = e.batch.add(1, 0, None, ('v3f/stream', [scaled_points[current_index][0], scaled_points[current_index][1], 0.]),
                                ('c3B/stream', [0,0,0]))
        
    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        seg_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)
        # print(len(self.ss))
        if idx+1 >= len(self.ss):
            idxadd1 = 0
        else: 
            idxadd1 = idx +1
        d_ss = self.ss[idxadd1] - self.ss[idx]

        d1, d2 = math.dist(self.points[idx],self.X0[0:2]),math.dist(self.points[idxadd1],self.X0[0:2])

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:
                # negative due to floating point precision
                # if the point is very close to the trackline, then the trianlge area is increadibly small
                h = 0
                x = d_ss + d1
                # print(f"Area square is negative: {Area_square}")
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5
        return x, h
    
    def distanceCalc(self,x, y, tx, ty):     #tx = target x, ty = target y
        dx = tx - x
        dy = ty - y
        return np.hypot(dx, dy)
    
    def search_nearest_target(self, obs):
        
        self.speed_list = self.waypoints[:, 5]
        self.min_dist = np.linalg.norm(self.X0[0:2] - self.points,axis = 1)
        self.ego_index = np.argmin(self.min_dist)
        if self.Tindx is None:
            self.Tindx = self.ego_index
            self.prev_x0 = [0.0,0.0,0.0]

        
        speed = self.speed_list[self.ego_index]
        self.Lf = speed*self.v_gain + self.lfd  # update look ahead distance
        
        # search look ahead target point index
        while self.Lf > self.distanceCalc(self.X0[0],
                                            self.X0[1], 
                                            self.points[self.Tindx][0], 
                                            self.points[self.Tindx][1]):

            if self.Tindx + 1> len(self.points)-1:
                self.Tindx = 0
            else:
                self.Tindx += 1

    def plan(self, obs, laptime):
        self.X0 = self.inputStateAdust(obs)
        self.search_nearest_target(obs)
        waypoint = np.dot (np.array([np.sin(-obs['poses_theta'][0]),np.cos(-obs['poses_theta'][0])]),
                           self.points[self.Tindx]-np.array(self.X0[0:2])) 
        speed = self.speed_list[self.ego_index]
        if np.abs(waypoint) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint/self.Lf**2)
        steering_angle = np.arctan(self.wheelbase/radius)

        self.completion = round(self.ego_index/len(self.points)*100,2)
        if self.completion > 99.5: self.completion = 100 
        speed_mod,steering_angle_mod = self.outputActionAdjust(speed,steering_angle)
        _,trackErr = self.interp_pts(self.ego_index,self.min_dist)
        slip_angle = self.slipAngleCalc(obs)

        self.saveFlag = self.toggle(self.saveFlag)

        if self.saveFlag:
            self.ds.saveStates(laptime, self.X0, speed, trackErr, self.scaledRand, steering_angle_mod, slip_angle)
            
        return speed_mod, steering_angle_mod
    

    def slipAngleCalc(self,obs):
        x = [self.X0[0] -self.prev_x0[0]]
        y = [self.X0[1] - self.prev_x0[1]]
        
        velocity_dir = np.arctan2(y,x)
        slip = np.abs(velocity_dir[0] - obs['poses_theta'][0]) *360 / (2*np.pi)
        if slip > 180:
            slip = slip-360

        

        self.prev_x0 = self.X0

        return slip
    
    def toggle(self,value):
        if value:
            return False
        else:
            return True

    def inputStateAdust(self,obs):
        X0 = [obs['poses_x'][0], obs['poses_y'][0], obs['linear_vels_x'][0]]
        rand = np.random.normal(mu,sigma,1)
        self.scaledRand = rand*self.scale

        if self.TESTMODE == "perception_noise":
            X0 = [obs['poses_x'][0]+self.scaledRand[0], obs['poses_y'][0]+self.scaledRand[0], obs['linear_vels_x'][0]]

        return X0
    
    def outputActionAdjust(self,speed,steering):
        rand = np.random.normal(mu,sigma,1)
        self.scaledRand = rand*self.scale

        speed_mod = speed
        steering_mod = steering

        if self.TESTMODE == "Outputnoise_speed":
            speed_mod = speed + self.scaledRand[0]
        elif self.TESTMODE == "Outputnoise_steering":
            steering_mod = steering + self.scaledRand[0]

        return speed_mod, steering_mod

