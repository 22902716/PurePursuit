import time
from f110_gym.envs.base_classes import Integrator
import gym
import yaml
import numpy as np
import math
from argparse import Namespace
import matplotlib.pyplot as plt
import collections
#todo:

# VERBOSE = False
VERBOSE = True


np.random.seed(0) #keep result consistent between the measurement in Pure-Pursuit and MPC default 0
# Max_iter = None


class PurePursuitPlanner:
    def __init__(self, conf, wb, speedgain, map_name, TESTMODE):
        self.wheelbase = wb                 #vehicle wheelbase
        self.conf = conf          
        self.map_name = map_name
        self.load_waypoints(conf)           
        self.speedgain = speedgain          
        self.drawn_waypoints = []
        self.ego_index = None
        self.Tindx = None
        self.TESTMODE = TESTMODE

        if map_name == "CornerHallE":
        
            if self.TESTMODE == "Benchmark" or self.TESTMODE == " ":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.1                   #lood forward distance constant
                self.Max_iter = 5
            elif self.TESTMODE == "localnoise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.1                     #lood forward distance constant
                self.Max_iter = 300
            elif self.TESTMODE == "v_gain":
                self.v_gain = 0.0                 #change this parameter for different tracks 
                self.lfd = 0.1                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "lfd":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.0                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_delay_steering" or self.TESTMODE == "perception_delay":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.1                     #lood forward distance constant
                self.Max_iter = 10

        elif map_name == "esp":
        
            if self.TESTMODE == "Benchmark" or self.TESTMODE == " ":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.1                   #lood forward distance constant
                self.Max_iter = 5
            elif self.TESTMODE == "localnoise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.1                     #lood forward distance constant
                self.Max_iter = 300
            elif self.TESTMODE == "v_gain":
                self.v_gain = 0.0                 #change this parameter for different tracks 
                self.lfd = 0.1                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "lfd":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.0                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_delay_steering" or self.TESTMODE == "perception_delay":
                self.v_gain = 0.12                 #change this parameter for different tracks 
                self.lfd = 0.1                     #lood forward distance constant
                self.Max_iter = 10

        elif map_name == "gbr":

            if self.TESTMODE == "Benchmark" or self.TESTMODE == " ":
                self.v_gain = 0.07                 #change this parameter for different tracks 
                self.lfd = 0.3                     #lood forward distance constant
                self.Max_iter = 5
            elif self.TESTMODE == "localnoise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
                self.v_gain = 0.07                 #change this parameter for different tracks 
                self.lfd = 0.3                     #lood forward distance constant
                self.Max_iter = 300
            elif self.TESTMODE == "v_gain":
                self.v_gain = 0.0                 #change this parameter for different tracks 
                self.lfd = 0.3                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "lfd":
                self.v_gain = 0.07                 #change this parameter for different tracks 
                self.lfd = 0.0                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_delay_steering" or self.TESTMODE == "perception_delay":
                self.v_gain = 0.07                 #change this parameter for different tracks 
                self.lfd = 0.3                     #lood forward distance constant
                self.Max_iter = 10

        elif map_name == "mco":

            if self.TESTMODE == "Benchmark" or self.TESTMODE == " ":
                self.v_gain = 0.05                 #change this parameter for different tracks 
                self.lfd = 0.45                     #lood forward distance constant
                self.Max_iter = 5
            elif self.TESTMODE == "localnoise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
                self.v_gain = 0.05                 #change this parameter for different tracks 
                self.lfd = 0.45                     #lood forward distance constant
                self.Max_iter = 300
            elif self.TESTMODE == "v_gain":
                self.v_gain = 0.0                 #change this parameter for different tracks 
                self.lfd = 0.45                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "lfd":
                self.v_gain = 0.05                 #change this parameter for different tracks 
                self.lfd = 0.0                     #lood forward distance constant
                self.Max_iter = 50
            elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_delay_steering" or self.TESTMODE == "perception_delay":
                self.v_gain = 0.05                 #change this parameter for different tracks 
                self.lfd = 0.45                     #lood forward distance constant
                self.Max_iter = 10

            
            
            
      
    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        # linetype = "centerline"
        linetype = "raceline"
        if self.map_name == "example":
            self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        else:
            self.waypoints = np.loadtxt('./maps/'+self.map_name+'_'+linetype+'.csv', delimiter=",")

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.map_name == "example":
            self.points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        else:
            self.points = np.vstack((self.waypoints[:,1],self.waypoints[:,2])).T
        scaled_points = 50.*self.points

        for i in range(self.points.shape[0]):
            if len(self.drawn_waypoints) < self.points.shape[0]:
                b = e.batch.add(1, 0, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def mark_current_ind(self,e, color):
    
        scaled_points = 50.*self.points

        if self.Tindx is None:
            current_index = 0
        else:
            current_index = self.Tindx
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

        d1, d2 = math.dist(self.points[idx],self.poses),math.dist(self.points[idxadd1],self.poses)

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
        

    def search_nearest_target(self, observations,scaledRand):
        
        self.speed_list = self.waypoints[:, self.conf.wpt_vind]

        if self.TESTMODE == "localnoise":
            self.poses = [observations['poses_x'][0]+scaledRand[0], observations['poses_y'][0]+scaledRand[0]]
        else:
            self.poses = [observations['poses_x'][0], observations['poses_y'][0]]

        self.min_dist = np.linalg.norm(self.poses - self.points,axis = 1)
        self.ego_index = np.argmin(self.min_dist)
        if self.Tindx is None:
            self.Tindx = self.ego_index
        

        speed = self.speed_list[self.ego_index]
        self.Lf = speed*self.v_gain + self.lfd  # update look ahead distance
        
        # search look ahead target point index
        while self.Lf > self.distanceCalc(self.poses[0],
                                            self.poses[1], 
                                            self.points[self.Tindx][0], 
                                            self.points[self.Tindx][1]):

            if self.Tindx + 1> len(self.points)-1:
                self.Tindx = 0
            else:
                self.Tindx += 1

    def action(self, obs):

        waypoint = np.dot (np.array([np.sin(-obs['poses_theta'][0]),np.cos(-obs['poses_theta'][0])]),
                           self.points[self.Tindx]-np.array(self.poses)) 

        self.speed = self.speed_list[self.ego_index]
        if np.abs(waypoint) < 1e-6:
            return self.speed, 0.
        radius = 1/(2.0*waypoint/self.Lf**2)
        steering_angle = np.arctan(self.wheelbase/radius)
        self.completion = round(self.ego_index/len(self.points)*100,2)
        # print(self.speed, steering_angle, self.Tindx,self.ego_index,completion)
        return self.speed, steering_angle

def main():    
    # TESTMODE_list = ["Benchmark", "v_gain", "lfd", "localnoise", "Outputnoise_speed", "Outputnoise_steering", "control_delay_steering", "control_delay_speed", "perception_delay"]
    # TESTMODE_list = ["Benchmark"]
    TESTMODE_list = ["v_gain", "lfd"]
    # TESTMODE_list = ["control_delay_steering", "perception_delay", "control_delay_speed"]
    # TESTMODE_list = [ "Outputnoise_speed", "Outputnoise_steering"]

    
    # TESTMODE_list = ["v_gain"]
    # TESTMODE_list = ["lfd"]

    # TESTMODE_list = ["localnoise"]
    # TESTMODE_list = ["perception_delay"]

    # TESTMODE_list = ["Outputnoise_steering"]
    # TESTMODE_list = ["Outputnoise_speed"]

    # TESTMODE_list = ["control_delay_steering"]
    # TESTMODE_list = ["control_delay_speed"]



    # TESTMODE_list = [" "]

    # map_name_list = ["esp","gbr","mco"]
    map_name_list = ["CornerHallE"]

    # map_name = "example"
    # map_name = "aut"
    # map_name = "esp"
    # map_name = "gbr"
    # map_name = "mco"

    for map_name in map_name_list:
        print("new map, " + map_name)

        for TESTMODE in TESTMODE_list:
            print("new mode, " + TESTMODE)
            speedgain = 1.
            color_marker = 0

            with open('maps/config_example_map.yaml') as file:
                conf_dict = yaml.load(file, Loader=yaml.FullLoader)
            conf = Namespace(**conf_dict)

            planner = PurePursuitPlanner(conf, 0.4, speedgain, map_name, TESTMODE)

            def render_callback(env_renderer):
                e = env_renderer

                # update camera to follow car

                x = e.cars[0].vertices[::2]
                y = e.cars[0].vertices[1::2]
                top, bottom, left, right = max(y), min(y), min(x), max(x)
                e.score_label.x = left
                e.score_label.y = top - 700
                e.left = left - 800
                e.right = right + 800
                e.top = top + 800
                e.bottom = bottom - 800

                planner.render_waypoints(env_renderer)
                planner.mark_current_ind(env_renderer, color_marker)

            if map_name == "example":
                env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
                obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            else:
                env = gym.make('f110_gym:f110-v0', map='./maps/'+map_name, map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
                obs, step_reward, done, info = env.reset(np.array([[planner.waypoints[0][1], planner.waypoints[0][2], planner.waypoints[0][4]]]))


            env.add_render_callback(render_callback)

            env.render()



            #plot specific init 
            laptime = 0.0
            laptime_sim_plot = []
            Iter_count = 0
            lapCount = 0
            x_position = []
            y_position = []
            plot_speed = []
            rand_append = []
            max_rand = 0.0
            trackErr = 0.0
            Expected_speed = []
            trackErr_list = []
            collision_count = 0
            txt_collision = []
            txt_completion = []
            txt_variable1 = []
            txt_variable2 = []
            txt_laptime = []
            txt_trackErr = []
            laptime_sim_plot_perstep = []
            txt_computation_time = []
            mu = 0.
            sigma = 0.2
            rand = None
            scale = 0.0
            scaledRand = 0.0
            time_delay = 0
            steering_angle = 0
            speed = 0
            # new_obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            if map_name == "example":
                new_obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            else:
                new_obs, step_reward, done, info = env.reset(np.array([[planner.waypoints[0][1], planner.waypoints[0][2], planner.waypoints[0][4]]]))
            queue = collections.deque(maxlen=time_delay+1)
            for i in range(0,time_delay+1):
                queue.append(obs)     

            computation_time_start = time.time()

                

            while Iter_count < planner.Max_iter:
                
                    
                if (lapCount+obs['lap_counts']+collision_count) != Iter_count or  obs['collisions'] == 1:
                    computation_time = time.time() - computation_time_start
                    if TESTMODE == "Benchmark":
                        Iter_count += 1
                        average_trackErr = np.mean(trackErr_list)
                        print("Iter_count = ", Iter_count, "laptime = ", laptime, " average tracking error = ",average_trackErr)

                        txt_variable_label1 = "Lap count"
                        txt_variable_label2 = "NA"
                        txt_variable1.append(Iter_count)
                        txt_variable2.append(0)

                    elif TESTMODE == 'v_gain':
                        Iter_count += 1
                        average_trackErr = np.mean(trackErr_list)

                        if obs['collisions']:
                            print("Iter_count = ", Iter_count, "I crashed, completion Percentage is", int(planner.completion),"%. when v_gain = ",planner.v_gain,"and LF = ",planner.Lf, " average tracking error = ",average_trackErr)
                        else:
                            print("Iter_count = ", Iter_count, "laptime = ", laptime,"v_gain = ",planner.v_gain,"and LF = ",planner.Lf, " average tracking error = ",average_trackErr)
                                            
                        txt_variable_label1 = "V_gain"
                        txt_variable_label2 = "Lookahead distance"
                        txt_variable1.append(round(planner.v_gain,3))
                        txt_variable2.append(planner.Lf)
                        # print("lapcount = ",Iter_count, "crash count = ", collision_count, "obs laps", obs['lap_counts'],"lapCount = ",lapCount)
                        # planner.v_gain += 0.005
                        planner.v_gain += 0.01
                    elif TESTMODE == 'lfd':
                        Iter_count += 1
                        average_trackErr = np.mean(trackErr_list)

                        if obs['collisions']:
                            print("Iter_count = ", Iter_count, "I crashed, completion Percentage is", int(planner.completion),"%. when lfd = ",planner.lfd,"LF = ",planner.Lf, " average tracking error = ",average_trackErr)
                        else:
                            print("Iter_count = ", Iter_count, "laptime = ", laptime,"look forward distance = ",planner.lfd, "LF = ",planner.Lf, " average tracking error = ",average_trackErr)

                        txt_variable_label1 = "lfd constant"
                        txt_variable_label2 = "Lookahead distance"
                        txt_variable1.append(round(planner.lfd,2))
                        txt_variable2.append(planner.Lf)
                        # planner.lfd += 0.02
                        planner.lfd += 0.05


                    elif TESTMODE == 'localnoise':
                        counter = Iter_count % 10
                        print(counter)
                        Iter_count += 1
                        average_trackErr = np.mean(trackErr_list)
                        
                        max_rand_idx = np.argmax(rand_append)
                        max_rand = rand_append[max_rand_idx][0]
                        if obs['collisions']:
                            print("Iter_count = ", Iter_count, "I crashed, completion Percentage is", int(planner.completion),"%. when scaled noise = ",scale," max noise = ", max_rand, " average tracking error = ",average_trackErr)
                        else:
                            print("Iter_count = ", Iter_count, "laptime = ", laptime,"scaled noise = ",scale," max noise = ", max_rand, " average tracking error = ",average_trackErr)
                        
                        rand_append = []

                        txt_variable_label1 = "noise scale"
                        txt_variable_label2 = "maximum noise"
                        txt_variable1.append(round(scale,2))
                        txt_variable2.append(round(max_rand,2))

                        # scale += 0.02
                        if counter == 9:
                            scale += 0.02

                        np.random.seed(counter)


                    elif TESTMODE == 'Outputnoise_speed' or TESTMODE == "Outputnoise_steering":
                        counter = Iter_count % 10
                        Iter_count += 1
                        average_trackErr = np.mean(trackErr_list)

                        max_rand_idx = np.argmax(rand_append)
                        max_rand = rand_append[max_rand_idx][0]
                        if obs['collisions']:
                            print("Iter_count = ", Iter_count, "I crashed, completion Percentage is", int(planner.completion),"%. when scaled noise = ",scale," max noise = ", max_rand, " average tracking error = ",average_trackErr)
                        else:
                            print("Iter_count = ", Iter_count, "laptime = ", laptime,"scaled noise = ",scale," max noise = ", max_rand, " average tracking error = ",average_trackErr)
                        
                        rand_append = []

                        txt_variable_label1 = "noise scale"
                        txt_variable_label2 = "maximum noise"
                        txt_variable1.append(round(scale,2))
                        txt_variable2.append(round(max_rand,2))
                        np.random.seed(counter)

                        if counter == 9:
                            scale += 0.02
                
                    
                    elif TESTMODE == "control_delay_speed" or TESTMODE == "control_delay_steering" or TESTMODE == "perception_delay":
                        Iter_count += 1
                        average_trackErr = np.mean(trackErr_list)

                        print_time_delay = time_delay * 10
                        if obs['collisions']:
                            print("Iter_count = ", Iter_count, "I crashed, completion Percentage is", int(planner.completion),"%. when delay is = ",print_time_delay," ms", " average tracking error = ",average_trackErr)
                        else:
                            print("Iter_count = ", Iter_count, "laptime = ", laptime,"when ",TESTMODE, " at ",print_time_delay," ms", " average tracking error = ",average_trackErr)                    

                        txt_variable_label1 = "Time delay(ms)"
                        txt_variable_label2 = "NA"
                        txt_variable1.append(print_time_delay)
                        txt_variable2.append(0)
                        queue = collections.deque(maxlen=time_delay+1)
                        time_delay += 1

                    if obs['collisions']:
                        collision_count += 1
                        planner.ego_index = None
                        planner.Tindx = None       
                        lapCount += obs['lap_counts'][0]             
                        
                        txt_collision.append(0)
                        txt_laptime.append(0)
                        txt_completion.append(planner.completion)
                        txt_trackErr.append(average_trackErr)
                        txt_computation_time.append(0)

                        if map_name == "example":
                            obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
                        else:
                            obs, step_reward, done, info = env.reset(np.array([[planner.waypoints[0][1], planner.waypoints[0][2], planner.waypoints[0][4]]]))    

                    else: 
                        planner.ego_index = None
                        planner.Tindx = None   
                        txt_collision.append(1)
                        txt_laptime.append(laptime)
                        txt_completion.append(100)
                        txt_trackErr.append(average_trackErr)
                        txt_computation_time.append(computation_time)
                        if map_name == "example":
                            obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
                            lapCount += 1            
                        else:
                            obs, step_reward, done, info = env.reset(np.array([[planner.waypoints[0][1], planner.waypoints[0][2], planner.waypoints[0][4]]]))    
                            if map_name != "CornerHallE":
                                lapCount += 1      

                    x_position= np.array(x_position)
                    y_position = np.array(y_position)
                    laptime_sim_plot = np.array(laptime_sim_plot)
                    plot_speed = np.array(plot_speed)
                    Expected_speed = np.array(Expected_speed)
                    trackErr_list = np.array(trackErr_list)

                    save_arr1 = np.concatenate([laptime_sim_plot[:,None]
                                    ,x_position[:,None]
                                    ,y_position[:,None]
                                    ,plot_speed[:,None]
                                    ,Expected_speed[:,None]
                                    ,trackErr_list[:,None]
                                    ],axis = 1)      
                    np.savetxt("Imgs/"+map_name+"/"+TESTMODE+"/"+str(Iter_count)+".csv", save_arr1, delimiter=',',header="laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error",fmt="%-10f")


                    laptime = 0.0
                    x_position = []
                    y_position = []
                    plot_speed = []
                    Expected_speed = []
                    laptime_sim_plot = []
                    trackErr_list = []
                    laptime_sim_plot_perstep = []
                    computation_time_start = time.time()

            


                if TESTMODE == "control_delay_speed" or TESTMODE == "control_delay_steering":
                    planner.search_nearest_target(obs,0)
                    new_speed ,new_steering_angle= planner.action(obs)
                    speed ,steering_angle= planner.action(obs)
                    control = [new_speed, new_steering_angle]
                            
                    x_position.append(obs['poses_x'][0])
                    y_position.append(obs['poses_y'][0])
                    plot_speed.append(obs['linear_vels_x'][0])
                    laptime_sim_plot.append(laptime)
                    Expected_speed.append(planner.speed_list[planner.ego_index])

                    z = 2 #update at 10Hz. simulation runs at 100Hz
                    # print("new cycle")
                    while z > 0 : 
                        # print("steering_angle = " , steering_angle, ", new steering angle = " , new_steering_angle, ", delay = ", time_delay)
                            # steering_angle = new_steering_angle
                            # speed = new_speed
                        queue.append(control)
                        if TESTMODE == "control_delay_speed":
                            obs, _, _, _ = env.step(np.array([[steering_angle, queue[0][0]*speedgain]]))
                        elif TESTMODE == "control_delay_steering":
                            obs, _, _, _ = env.step(np.array([[queue[0][1], speed*speedgain]]))
                        z -= 1
                        _,trackErr = planner.interp_pts(planner.ego_index,planner.min_dist)
                        laptime += 0.01
                        laptime_sim_plot_perstep.append(laptime)

                    trackErr_list.append(trackErr)


                elif TESTMODE == "perception_delay":

                    planner.search_nearest_target(obs,0)
                    speed ,steering_angle= planner.action(obs)
        

                    x_position.append(obs['poses_x'][0])
                    y_position.append(obs['poses_y'][0])
                    plot_speed.append(obs['linear_vels_x'][0])
                    laptime_sim_plot.append(laptime)
                    Expected_speed.append(planner.speed_list[planner.ego_index])
                    rand_append.append(0)

                    z = 2 #update at 10Hz if z = 10. simulation runs at 100Hz
                    while z > 0 :
                        new_obs, _, _, _ = env.step(np.array([[steering_angle, speed*speedgain]]))
                        laptime += 0.01 
                        _,trackErr = planner.interp_pts(planner.ego_index,planner.min_dist)
                        laptime_sim_plot_perstep.append(laptime)

                        queue.append(new_obs)
                        obs = queue[0]
                        z -= 1  

                    trackErr_list.append(trackErr) 

                else:

                    rand = np.random.normal(mu,sigma,1)
                    scaledRand = rand*scale
                    planner.search_nearest_target(obs,scaledRand)
                    speed, steering_angle = planner.action(obs)
                            
                    x_position.append(obs['poses_x'][0])
                    y_position.append(obs['poses_y'][0])
                    plot_speed.append(obs['linear_vels_x'][0])
                    laptime_sim_plot.append(laptime)
                    Expected_speed.append(planner.speed_list[planner.ego_index])
                    rand_append.append(scaledRand)
                    
                    # print(obs['poses_x'][0], obs['poses_y'][0])
                    # print("steering angle = ", steering_angle)
                    # print("Target index = ", Tindx)
                    # print("closest index = ",ego_index)
                    # print("current speed = ", obs['linear_vels_x'][0])
                    # print("look ahead distane = ", planner.Lf)



                    z = 2 #update at 10Hz. simulation runs at 100Hz
                    while z > 0 :
                        if TESTMODE == "Outputnoise_steering":
                            rand = np.random.normal(mu,sigma,1)
                            scaledRand = rand*scale
                            obs, _, _, _ = env.step(np.array([[steering_angle+scaledRand[0], speed*speedgain]]))

                        elif TESTMODE == "Outputnoise_speed":
                            rand = np.random.normal(mu,sigma,1)
                            scaledRand = rand*scale
                            obs, _, _, _ = env.step(np.array([[steering_angle, speed*speedgain+scaledRand[0]]]))
                        else: 
                            obs, _, _, _ = env.step(np.array([[steering_angle, speed*speedgain]]))
                            # time.sleep(0.01)
                        z -= 1
                        _,trackErr = planner.interp_pts(planner.ego_index,planner.min_dist)

                        laptime += 0.01  
                        laptime_sim_plot_perstep.append(laptime)
                    trackErr_list.append(trackErr)

                    # time.sleep(0.1)

                if map_name == "CornerHallE" and planner.completion >= 98:
                    obs, step_reward, done, info = env.reset(np.array([[planner.waypoints[0][1], planner.waypoints[0][2], planner.waypoints[0][4]]]))    
                    lapCount += 1
                env.render(mode='human')
                
            
            txt_collision = np.array(txt_collision)
            txt_completion = np.array(txt_completion)
            txt_variable1 = np.array(txt_variable1)
            txt_variable2 = np.array(txt_variable2)
            txt_laptime = np.array(txt_laptime)
            txt_trackErr = np.array(txt_trackErr)
            txt_computation_time = np.array(txt_computation_time)
            
            save_arr = np.concatenate([txt_collision[:,None]
                                    ,txt_laptime[:,None]
                                    ,txt_completion[:,None]
                                    ,txt_variable1[:,None]
                                    ,txt_variable2[:,None]
                                    ,txt_trackErr[:,None]
                                    ,txt_computation_time[:,None]
                                    ],axis = 1)
            if VERBOSE:
                # plt.figure()
                # plt.title(label = "Completion(%) vs lap iteration")
                # plt.scatter(txt_variable1,txt_completion)
                # plt.savefig(f"Imgs/{map_name}/{TESTMODE}/success_rate.svg")
                # plt.figure()
                # plt.title(label = "Lap time vs Look forward distance gain")
                # plt.scatter(txt_variable1,txt_laptime)
                # plt.savefig(f"Imgs/{map_name}/{TESTMODE}/laptime.svg")
                np.savetxt("csv/"+map_name+ '/'+map_name +'_'+ TESTMODE+'.csv', save_arr, delimiter=',',header="lap success, laptime(s), completion(%),"+ txt_variable_label1 + ", " + txt_variable_label2 + " , average tracking error, computation time",fmt="%-10f")
                print(TESTMODE + " csv file saved")
            print("TEST COMPLETE")
            # print("Never gonna give you upppppp")

            
if __name__ == '__main__':
    main()

    