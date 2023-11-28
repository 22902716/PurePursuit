from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import matplotlib.pyplot as plt
import numpy as np
from PurePursuit import PurePursuitPlanner
import time
import collections as co


SAVELAPDATA = True
# SAVELAPDATA = False


np.random.seed(0) #keep result consistent between the measurement in Pure-Pursuit and MPC default 0
UPDATE_PERIOD = 2

def main():
    map_name_list = ["gbr","esp","mco"]
    # map_name_list = ["mco"]

    '''Tuning'''
    # testmode_list = ["v_gain","lfd"]

    '''Experiments'''
    testmode_list = ["perception_delay"]

    # testmode_list = ["Benchmark","perception_noise","Outputnoise_speed","Outputnoise_steering","control_delay_speed","control_Delay_steering","perception_delay"]
    # testmode_list = ["Benchmark","perception_noise","Outputnoise_speed","Outputnoise_steering"]
    # testmode_list = ["control_delay_speed","control_Delay_steering","perception_delay"]

    
    for map_name in map_name_list:
        print("new map, " + map_name)
        for TESTMODE in testmode_list:
            print("new mode, " + TESTMODE)
            with open(f'param/{map_name}.yaml') as file:
                param_dict = yaml.load(file, Loader=yaml.FullLoader)
            param = Namespace(**param_dict)

            planner = PurePursuitPlanner(map_name,TESTMODE,param)
            
            
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

            time_delay = 0
            iter_count = 0
            lapCount = 0
            collision_count = 0
            laptime = 0.0

            env = gym.make('f110_gym:f110-v0', map='./maps/'+map_name, map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
            obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))
            new_obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))

            env.add_render_callback(render_callback)
            env.render()
            computation_time_start = time.time()

            new_speed,new_steering_angle = planner.plan(obs,laptime)
            control = [new_speed, new_steering_angle]
            control_queue, obs_queue = initqueue(obs,control,time_delay)

            while iter_count < planner.Max_iter:


                if (lapCount+obs['lap_counts']+collision_count) != iter_count or  obs['collisions'] or new_obs['collisions']:
                    computation_time = time.time() - computation_time_start
                    lap_success = 1
                    planner.scale = iter_count // 10 * 0.02
                    iter_count += 1

                    if obs['collisions'] or new_obs['collisions']:
                        print("Iter_count = ", iter_count, "I crashed, completion Percentage is", int(planner.completion),"%")
                        lap_success = 0
                        collision_count += 1
                        lapCount += obs['lap_counts'][0]
                        planner.ego_index = None
                        planner.Tindx = None    
                        obs, _, _, _ = env.reset(np.array([[0, 0, 0]]))
                    else:
                        print("Iter_count = ", iter_count, "laptime = ", laptime)                      

                    if TESTMODE == "Benchmark":
                        var1 = 0
                        var2 = 0
                    if TESTMODE == "perception_noise":
                        var1 = planner.scale
                        var2 = max(planner.ds.txt_x0[:,5])
                    if TESTMODE == "Outputnoise_speed":
                        var1 = planner.scale
                        var2 = max(planner.ds.txt_x0[:,5])
                    if TESTMODE == "Outputnoise_steering":
                        var1 = planner.scale
                        var2 = max(planner.ds.txt_x0[:,5])
                    if TESTMODE == "control_delay_speed" or TESTMODE == "control_Delay_steering" or TESTMODE == "perception_delay":
                        var1 = time_delay*10
                        var2 = 0
                    if TESTMODE == "v_gain":
                        var1 = planner.v_gain
                        var2 = planner.lfd + planner.v_gain*speed
                        planner.v_gain += 0.01
                    if TESTMODE == "lfd":
                        var1 = planner.lfd
                        var2 = planner.lfd + planner.v_gain*speed
                        planner.lfd += 0.05

                    aveTrackErr = np.mean(planner.ds.txt_x0[:,5])

                    if SAVELAPDATA:
                        planner.ds.savefile(iter_count)

                    planner.ds.lapInfo(iter_count,lap_success,laptime,planner.completion,var1,var2,aveTrackErr,computation_time)
                    laptime = 0.0
                    computation_time_start = time.time()
                    if TESTMODE == "control_delay_speed" or TESTMODE == "control_Delay_steering" or TESTMODE == "perception_delay":
                        time_delay += 1
                        new_speed,new_steering_angle = planner.plan(obs,laptime)
                        control = [new_speed, new_steering_angle]
                        control_queue, obs_queue = initqueue(obs,control,time_delay)

                if TESTMODE == "perception_delay":
                    speed,steering_angle = planner.plan(obs_queue[0],laptime)
                else:
                    speed,steering_angle = planner.plan(obs,laptime)
                    new_speed,new_steering_angle = planner.plan(obs,laptime)
                    control = [new_speed, new_steering_angle]

                z = UPDATE_PERIOD
                while z > 0:
                    control_queue.append(control)
                    if TESTMODE == "control_delay_speed":
                        obs, _, _, _ = env.step(np.array([[steering_angle, control_queue[0][0]]]))
                    elif TESTMODE == "control_Delay_steering":
                        obs, _, _, _ = env.step(np.array([[control_queue[0][1], speed]]))
                    else:
                        if TESTMODE == "perception_delay":
                            new_obs, _, _, _ = env.step(np.array([[steering_angle,speed]]))
                            obs_queue.append(new_obs)
                        else:
                            obs, _, _, _ = env.step(np.array([[steering_angle, speed]]))
                    z -= 1
                    laptime += 0.01
                env.render(mode='human_fast') #'human_fast'(without delay) or 'human' (with 0.001 delay)
            planner.ds.saveLapInfo()

def initqueue(obs, control, time_delay):
    control_queue = co.deque(maxlen = time_delay+1)
    for i in range(0, time_delay+1):
        control_queue.append(control)

    obs_queue = co.deque(maxlen = time_delay+1)
    for i in range(0, time_delay+1):
        obs_queue.append(obs)

    return control_queue, obs_queue

if __name__ == '__main__':
    main()