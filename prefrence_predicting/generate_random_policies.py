import gym
import numpy as np
from collections import defaultdict
# import plotting
from grid_world import GridWorldEnv
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random_policy_data
from value_iteration import learn_successor_feature_iter, iterative_policy_evaluation,learn_successor_feature,value_iteration,follow_policy,policy_improvement,build_pi

changed_gt_rew_vec = False

vec = np.array([-1,50,-50,1,-1,-2])
V,Qs = value_iteration(rew_vec = vec,GAMMA=0.999)
pi = build_pi(Qs)
gt_succ_feat,_ = learn_successor_feature_iter(pi,0.999,rew_vec = vec)


def get_random_reward_vector(gt_rew_vec):
    if gt_rew_vec is None:
        space = [-1,50,-50,1,-1,-2]
    else:
        space = [50,-50,1,-1,-2,0,-10,10,5]
    vector = []
    for i in range(6):
        s = random.choice(space)
        # space.remove(s)
        vector.append(s)
    return np.array(vector)

def generate_random_policy(GAMMA,env=None,gt_rew_vec=None):
    vec = get_random_reward_vector(gt_rew_vec)
    # vec = np.array([-1,50,-50,1,-1,-2])
    V,Qs = value_iteration(rew_vec = vec,GAMMA=GAMMA,env=env)
    # follow_policy(Qs, 1000,viz_policy=True)
    pi = build_pi(Qs,env=env)
    # psi_og = learn_successor_feature(Qs,V,GAMMA,rew_vec = vec)
    succ_feat, _= learn_successor_feature_iter(pi,GAMMA,rew_vec = vec,env=env)
    return succ_feat, pi

def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def generate_all_policies(n_policies,GAMMA,env=None,gt_rew_vec=None):
    succ_feats = []
    pis = []
    i = 0
    n_duplicates = 0 #makes sure we do not try and generate more unique policies than exist
    while i < n_policies and n_duplicates < 100:
        i+=1
        succ_feat, pi = generate_random_policy(GAMMA,env,gt_rew_vec)
        if is_arr_in_list(succ_feat, succ_feats):
            i-=1
            n_duplicates+= 1
        else:
            # print ("generated policy: " + str(len(pis)))
            succ_feats.append(succ_feat)
            pis.append(pi)
    return succ_feats, pis


def calc_value(state,gt_rew_vec=None,env=None):
    if gt_rew_vec is None or env is None:
        w = [-1,50,-50,1,-1,-2]
        x,y = state
        return np.dot(random_policy_data.gt_succ_feat[x][y],w)
    else:
        w = gt_rew_vec
        x,y = state
        if not random_policy_data.changed_gt_rew_vec:
            V,Qs = value_iteration(rew_vec = np.array(gt_rew_vec),GAMMA=0.999,env=env)
            pi = build_pi(Qs,env=env)
            gt_succ_feat,_ = learn_successor_feature_iter(pi,0.999,rew_vec = vec,env=env)
            np.save("gt_succ_feat_2.npy",gt_succ_feat)
            # global changed_gt_rew_vec
            random_policy_data.changed_gt_rew_vec = True
            return np.dot(gt_succ_feat[x][y],w)
        else:
            gt_succ_feat = np.load("gt_succ_feat_2.npy")
            return np.dot(gt_succ_feat[x][y],w)


# succ_feats, pis = generate_all_policies(100,0.9)
# succ_feats, pis = generate_all_policies(1000,0.999)
# np.save("1000_succ_feats.npy",succ_feats)
# np.save("1000_pis.npy",pis)
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
