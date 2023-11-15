from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import matplotlib.pyplot as plt
import numpy as np
from PurePursuit import PurePursuitPlanner

np.random.seed(0) #keep result consistent between the measurement in Pure-Pursuit and MPC default 0

def main():
    map_name_list = ["gbr","esp","mco"]

    '''Tuning'''
    # testmode_list = ["v_gain","lfd"]

    '''Experiments'''
    testmode_list = ["Benchmark","localnoise","Outputnoise_speed","Outputnoise_steering","control_delay_speed","control_Delay_steering","perception_delay"]
    
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

            iter_count = 0

            env = gym.make('f110_gym:f110-v0', map='./maps/'+map_name, map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
            # obs, step_reward, done, info = env.reset(np.array([[planner.X0[0], planner.X0[1], planner.X0[2]]]))
            obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))

            env.add_render_callback(render_callback)
            env.render()

            while iter_count < planner.Max_iter:
                planner.scale = iter_count * 0.02
                
                speed,steering_angle = planner.plan(obs)
                z = 2
                while z > 0:
                    obs, _, done, _ = env.step(np.array([[steering_angle, speed]]))
                    z-=1
                env.render(mode='human_fast') #'human_fast'(without delay) or 'human' (with 0.001 delay)


if __name__ == '__main__':
    main()