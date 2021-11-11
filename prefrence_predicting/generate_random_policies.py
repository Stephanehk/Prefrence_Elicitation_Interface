import gym
import numpy as np
from collections import defaultdict
# import plotting
from grid_world import GridWorldEnv
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from value_iteration import learn_successor_feature_iter, iterative_policy_evaluation,learn_successor_feature,value_iteration,follow_policy,policy_improvement,build_pi

def get_random_reward_vector():
    space = [-1,50,-50,1,-1,-2]
    vector = []
    for i in range(6):
        s = random.choice(space)
        space.remove(s)
        vector.append(s)
    return np.array(vector)

def generate_random_policy(GAMMA):
    vec = get_random_reward_vector()
    # vec = np.array([-1,50,-50,1,-1,-2])
    V,Qs = value_iteration(rew_vec = vec,GAMMA=GAMMA)
    # follow_policy(Qs, 1000,viz_policy=True)
    pi = build_pi(Qs)
    # psi_og = learn_successor_feature(Qs,V,GAMMA,rew_vec = vec)
    succ_feat = learn_successor_feature_iter(pi,GAMMA,rew_vec = vec)
    return succ_feat, pi

def generate_all_policies(n_policies,GAMMA):
    succ_feats = []
    pis = []
    for i in range (n_policies):
        succ_feat, pi = generate_random_policy(GAMMA)
        succ_feats.append(succ_feat)
        pis.append(pi)
    return succ_feats, pis

def calc_value(w, state, succ_feats):
    max_ = 0
    x,y = state
    for i in range(len(succ_feats)):
        max_ = max(np.dot(succ_feats[i][x][y],w),max_)
    return max_

# succ_feats, pis = generate_all_policies(100)
# print ("============================")
# v_approx = calc_value([-1,50,-50,1,-1,-2], (0,0), succ_feats)
# print (v_approx)

# 
# rew_vect = np.array([ -1.3642614, 14.295705, -17.196255, 0.65846264, -0.12430768, -1.5144913 ])
# V,Q = value_iteration(rew_vec =rew_vect,GAMMA=0.999)
#
# avg_return = follow_policy(Q,100,viz_policy=False)
# pi = build_pi(Q)
# V_under_gt = iterative_policy_evaluation(pi, GAMMA=0.999)
# avg_return = np.sum(V_under_gt)/92
# print (avg_return)
