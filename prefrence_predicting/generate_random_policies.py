import gym
import numpy as np
from collections import defaultdict
# import plotting
from grid_world import GridWorldEnv
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from value_iteration import iterative_policy_evaluation,learn_successor_feature,value_iteration,follow_policy,policy_improvement,build_pi

def validate_custome_reward_func():
    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    env.set_custom_reward_function([-1,50,-50,1,-1,-2])
    for x in range (len(env.reward_function)):
        for y in range(len(env.reward_function[0])):
            for a_i in range(4):
                print (env.reward_function[x][y][a_i])
                print (env.prev_reward_function[x][y][a_i])
                print ((x,y,a_i))
                print ("\n")
                assert env.reward_function[x][y][a_i] == env.prev_reward_function[x][y][a_i]


def get_random_reward_vector():
    space = [-1,50,-50,1,-1,-2]
    vector = []
    for i in range(6):
        s = random.choice(space)
        space.remove(s)
        vector.append(s)
    return np.array(vector)

def generate_random_policy():
    # vec = get_random_reward_vector()
    vec = np.array([-1,-50,50,-1,-1,-1])

    GAMMA = 1
    V,Qs = value_iteration(rew_vec = vec,GAMMA=GAMMA)
    follow_policy(Qs, 1000,viz_policy=True)

    psi = learn_successor_feature(Qs,V,GAMMA,rew_vec = vec)
    print (psi[0][0])
    print (np.dot(psi[0][0],vec))
    print (V[0][0])


# # generate_random_policy()
V,Qs = value_iteration(GAMMA=1)
# psi = learn_successor_feature(Qs,V,1)
# print (psi)

pi = build_pi(Qs)
V = iterative_policy_evaluation(pi, GAMMA=1)

print (V)
