import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pickle
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import torch
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import random
import openpyxl
import timeit
import copy
from datetime import datetime
import os

from grid_world import GridWorldEnv
from load_training_data import get_all_statistics,find_end_state,get_all_statistics_aug_human, get_N_length_dataset, get_state_feature, get_action_feature
from value_iteration import value_iteration, follow_policy, learn_successor_feature,get_gt_avg_return,build_pi,iterative_policy_evaluation,learn_successor_feature_iter,build_random_policy
from generate_random_policies import generate_all_policies, calc_value
from prefrence_stats_analysis import LogisticRegression
from generate_random_mdp import generate_MDP,is_in_blocked_area,contains_cords
import random_policy_data

# from generate_random_gridworlds import eval_under_grid_worlds

keep_ties = True
n_prob_samples = 1
n_prob_iters = 30

GAMMA=0.999
include_dif_traj_lengths = True
# mode = "deterministic_user_data"
# mode = "user_data"
# mode = "sigmoid"
mode = "deterministic"
# gt_V,_ = value_iteration(GAMMA=0.999)
LR = 2
N_ITERS = 30000
# optimizer_add = "line_search"
# optimizer_add = "mini_batch"
# optimizer_add = "cyclic"
optimizer_add = "none"

use_random_MDPs = False
use_extended_SF = False

# synth_pref_type = "pr_model_none_data"

prefrence_model = "pr" #how we generate prefs
prefrence_assum = "pr" #how we learn prefs
if prefrence_assum == "er":
    # print("generating policies...")
    # succ_feats, pis = generate_all_policies(100,GAMMA)
    # np.save(str(GAMMA) + "_100_succ_feats.npy",succ_feats)
    # np.save(str(GAMMA) + "_100_pis.npy",pis)
    # print("finished")
    # # assert False
    #
    # succ_feats = np.load("100_succ_feats.npy",allow_pickle=True)
    # pis = np.load("100_pis.npy",allow_pickle=True)
    #
    # succ_feats = np.load("1000_succ_feats.npy",allow_pickle=True)
    # pis = np.load("1000_pis.npy",allow_pickle=True)

    # succ_feats = np.load("succ_feats_no_gt.npy",allow_pickle=True)
    # pis = np.load("pis_no_gt.npy",allow_pickle=True)
    # succ_feats = None
    # pis = None

    # succ_feats_gt = np.load("succ_feats.npy",allow_pickle=True)
    # pis_gt = np.load("pis.npy",allow_pickle=True)
    pass

def find_reward_features(traj,env,use_extended_SF=False,GAMMA=1,traj_length=3):
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    if use_extended_SF:
        phi = np.zeros(6+(4 * env.width * env.height))
        phi_dis = np.zeros(6+(4 * env.width * env.height))
    else:
        phi = np.zeros(6)
        phi_dis = np.zeros(6)

    for i in range (1,traj_length+1):
        # print ("===========================")
        # print (traj_ts_x,traj_ts_y)
        # print ("===========================")
        #check if we are at terminal state
        if env.board[traj_ts_x, traj_ts_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
            continue
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(env.board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(env.board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], env.board):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
            # print ("state feature: ")
            # print (get_state_feature(traj_ts_x,traj_ts_y,env))
            dis_state_sf = (GAMMA**(i-1))*get_state_feature(traj_ts_x,traj_ts_y,env)
            state_sf = get_state_feature(traj_ts_x,traj_ts_y,env)
        else:
            #check if we are at terminal state
            if env.board[traj_ts_x, traj_ts_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
                # print ("in terminal")
                dis_state_sf = [0,0,0,0,0,0]
                state_sf = [0,0,0,0,0,0]
            else:
                dis_state_sf = (GAMMA**(i-1))*(get_state_feature(traj_ts_x,traj_ts_y,env)*[1,0,0,0,0,1])
                state_sf = (get_state_feature(traj_ts_x,traj_ts_y,env)*[1,0,0,0,0,1])

        if use_extended_SF:

            #find action index
            for a_i, action_ in enumerate(actions):
                if action_[0] == traj[i][0] and action_[1] == traj[i][1]:
                    action_index = a_i


            if env.board[prev_x, prev_y] == 1 or env.board[prev_x, prev_y] == 3 or env.board[prev_x, prev_y] == 7 or env.board[prev_x, prev_y] == 9:
                dis_action_sf = np.zeros(env.height*env.width*4)
            else:
                dis_action_sf =(GAMMA**(i-1))*get_action_feature(prev_x, prev_y, action_index,env=env)
            dis_state_sf = list(dis_state_sf)
            dis_state_sf.extend(dis_action_sf)

            if env.board[prev_x, prev_y] == 1 or env.board[prev_x, prev_y] == 3 or env.board[prev_x, prev_y] == 7 or env.board[prev_x, prev_y] == 9:
                action_sf = np.zeros(env.height*env.width*4)
            else:
                action_sf =get_action_feature(prev_x, prev_y, action_index,env=env)

            # print ("action_sf: " + str(action_sf))
            state_sf = list(state_sf)
            state_sf.extend(action_sf)

        phi_dis += dis_state_sf
        phi+= state_sf


        prev_x = traj_ts_x
        prev_y = traj_ts_y
    # print ("--done--\n")
    return phi_dis,phi

def augment_data(X,Y,ytype="scalar"):
    aX = []
    ay = []
    for x,y in zip(X,Y):
        aX.append(x)
        ay.append(y)

        neg_x = [x[1],x[0]]
        aX.append(neg_x)
        if ytype == "scalar":
            ay.append(1-y)
        else:
            ay.append([y[1],y[0]])
    return np.array(aX), np.array(ay)

def get_gt_er (x,t1_ss=None,t1_es=None,t2_ss=None,t2_es=None,gt_rew_vec=None,env=None):
    #calculates gt expected return
    if gt_rew_vec is not None:
        w = gt_rew_vec
    else:
        w = [-1,50,-50,1,-1,-2]
    if (torch.is_tensor(x)):
        x = x.detach().numpy()


    if t1_ss == None:
        t1_ss = [int(x[0][6]), int(x[0][7])]
        t1_es = [int(x[0][8]), int(x[0][9])]

        t2_ss = [int(x[1][6]), int(x[1][7])]
        t2_es = [int(x[1][8]), int(x[1][9])]


    x = np.array(x)
    r = np.dot(x[:,0:6],w)

    r1_er = r[0] + calc_value(t1_es,gt_rew_vec,env) - calc_value(t1_ss,gt_rew_vec,env)
    r2_er = r[1] + calc_value(t2_es,gt_rew_vec,env) - calc_value(t2_ss,gt_rew_vec,env)
    return r1_er, r2_er

def clean_y(X,R,Y,sess):
    formatted_y = []
    out_X = []

    synth_formatted_y = []
    synth_y_dist = []
    synth_out_X = []
    n_unique_trajs = 0

    for x,r,y,ses in zip(X,R,Y,sess):
        x = [list(x[0]),list(x[1])]

        #change x to include start end state for each trajectory
        if prefrence_assum == "er":
            x[0] = list(x[0])
            x[0].extend([ses[0][0][0],ses[0][0][1], ses[0][1][0],ses[0][1][1]])

            x[1] = list(x[1])
            x[1].extend([ses[1][0][0],ses[1][0][1], ses[1][1][0],ses[1][1][1]])
            x = [x[0],x[1]]

        if y != None and not is_subset(x,out_X):
            n_unique_trajs += 1

        # if is_subset(x,out_X):
        #     n_unique_trajs += 1


        if y == 0:
            formatted_y.append(np.array([1,0]))
            out_X.append(x)
        elif y == 1:
            formatted_y.append(np.array([0,1]))
            out_X.append(x)
        elif y == 0.5:
            formatted_y.append(np.array([0.5,0.5]))
            out_X.append(x)
        elif y == None:
            if mode == "deterministic_user_data":
                #means there is still some synthetic data mixed in
                assert False

            # if is_subset(x,synth_out_X):
            #     continue

            if not "deterministic" in mode:
                for n_samp in range(n_prob_samples):
                    if prefrence_assum == "pr":
                        r1_prob = sigmoid(r[0]-r[1])
                        r2_prob = sigmoid(r[1]-r[0])
                    elif prefrence_assum == "er":
                        r1_er, r2_er = get_gt_er (x)
                        r1_prob = sigmoid(r1_er-r2_er)
                        r2_prob = sigmoid(r2_er-r1_er)

                    num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                    if num == 1:
                        pref = [1,0]
                    elif num == 0:
                        pref = [0,1]
                    synth_formatted_y.append(np.array(pref))
                    synth_y_dist.append([r1_prob, r2_prob])
                    synth_out_X.append(x)
            else:
                if prefrence_assum == "er":
                    r1_er, r2_er = get_gt_er (x)
                    synth_formatted_y.append(np.array(get_pref([r1_er, r2_er])))
                else:
                    synth_formatted_y.append(np.array(get_pref(r)))
                synth_out_X.append(x)
        else:
            formatted_y.append(np.array(y))
            out_X.append(x)


    loss_coef = (len(out_X)/n_unique_trajs)/n_prob_samples
    loss_coef = min(1,loss_coef)

    if mode == "deterministic_user_data":
        assert loss_coef == 1

    # print ("loss coef: " + str(loss_coef))

    return out_X,formatted_y,synth_out_X,synth_formatted_y,synth_y_dist, loss_coef

def format_y(Y,ytype="scalar"):
    formatted_y = []
    if ytype=="scalar":
        for y in Y:
            if y == 0:
                formatted_y.append(np.array([1,0]))
            elif y == 1:
                formatted_y.append(np.array([0,1]))
            elif y == 0.5:
                formatted_y.append(np.array([0.5,0.5]))
    else:
        formatted_y = Y
        # for y in Y:
        #     formatted_y.append([y])

    return torch.tensor(formatted_y,dtype=torch.float)

def format_X(X):
    return torch.tensor(X,dtype=torch.float)

def sigmoid(val):
    return 1 / (1 + math.exp(-val))

def is_slice_in_list(s,l):
    len_s = len(s) #so we don't recompute length of s on every iteration
    return any(s == l[i:len_s+i] for i in range(len(l) - len_s+1))

def is_subset(sub, set):
    for pair in set:
        if sub[0] == pair[0] and sub[1] == pair[1]:
            return True
    return False
#
x2er = {}
x2r = {}

def generate_synthetic_prefs(pr_X,rewards,sess,mode,gt_rew_vec=[-1,50,-50,1,-1,-2],env=None):
    synth_y = []
    non_redundent_pr_X = []
    expected_returns = []

    # if synth_pref_type.lower() != "none":
    #     pref_model = torch.load("preference_models/" + synth_pref_type)
    n_removed = 0
    # pr_X = np.array(pr_X)
    # if use_extended_SF:
    #     pr_X = pr_X[:,0:6]

    for r,x,ses in zip(rewards,pr_X,sess):
        if prefrence_model == "pr" and prefrence_assum == "pr":
            x_f = [list(x[0][0:6]),list(x[1][0:6])]
            x_orig = [list(x[0]), list(x[1])]
        #change x to include start end state for each trajectory
        if prefrence_model == "er" and prefrence_assum == "er":
            if use_extended_SF:
                assert False
            x[0] = list(x[0])
            x[0].extend([ses[0][0][0],ses[0][0][1], ses[0][1][0],ses[0][1][1]])

            x[1] = list(x[1])
            x[1].extend([ses[1][0][0],ses[1][0][1], ses[1][1][0],ses[1][1][1]])
            x_f = [x[0],x[1]]
            x_orig = [list(x[0]), list(x[1])]
        if prefrence_model == "pr" and prefrence_assum == "er":
            if use_extended_SF:
                assert False
            x[0] = list(x[0])
            x[0].extend([0,0,0,0])
            x[1] = list(x[1])
            x[1].extend([0,0,0,0])
            x_f = [x[0],x[1]]
            x_orig = [list(x[0]), list(x[1])]
        if prefrence_model == "er" and prefrence_assum == "pr":
            x_f = [list(x[0]),list(x[1])]
            x_orig = [list(x[0]), list(x[1])]


        #adds discounting to prefrences
        # t1_n_trans = x[0][0] + x[0][1] + x[0][2] + x[0][5]
        # t2_n_trans = x[1][0] + x[1][1] + x[1][2] + x[1][5]
        # r[0] *=(np.power(GAMMA,t1_n_trans))
        # r[1] *=(np.power(GAMMA,t2_n_trans))
        #remove duplicates
        # if is_subset(x_f,non_redundent_pr_X):
        #     n_removed+=1
        #     continue

        t1_ss = [int(ses[0][0][0]), int(ses[0][0][1])]
        t1_es = [int(ses[0][1][0]), int(ses[0][1][1])]

        t2_ss = [int(ses[1][0][0]), int(ses[1][0][1])]
        t2_es = [int(ses[1][1][0]), int(ses[1][1][1])]

        diff_r = r[1] - r[0]
        diff_vs0 = calc_value(t2_ss,gt_rew_vec,env) - calc_value(t1_ss,gt_rew_vec,env)
        diff_vst = calc_value(t2_es,gt_rew_vec,env) - calc_value(t1_es,gt_rew_vec,env)

        if mode == "sigmoid":

            if prefrence_model == "er":
                r1_er, r2_er = get_gt_er (x,t1_ss,t1_es,t2_ss,t2_es,gt_rew_vec,env)

            if prefrence_model == "pr" and not keep_ties and r[1] == r[0]:
                continue

            if prefrence_model == "er" and not keep_ties and r1_er == r2_er:
                continue

            for n_samp in range(n_prob_samples):

                if prefrence_model == "pr":
                    r1_prob = sigmoid(r[0]-r[1])
                    r2_prob = sigmoid(r[1]-r[0])

                elif prefrence_model == "er":
                    r1_prob = sigmoid(r1_er-r2_er)
                    r2_prob = sigmoid(r2_er-r1_er)

                num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                if num == 1:
                    pref = [1,0]
                elif num == 0:
                    pref = [0,1]
                synth_y.append(pref)
                non_redundent_pr_X.append(x_orig)
        else:
            if prefrence_model == "er":
                r1_er, r2_er = get_gt_er (x,t1_ss,t1_es,t2_ss,t2_es,gt_rew_vec,env)
                pref = get_pref([r1_er, r2_er])
            else:
                pref = get_pref(r)

            if pref == [0.5,0.5] and not keep_ties:
                continue

            if prefrence_model == "er":
                expected_returns.append([r1_er, r2_er])
            synth_y.append(pref)
            non_redundent_pr_X.append(x_orig)
    print ("removed " + str(n_removed) + " duplicate segment pairs")
    return non_redundent_pr_X, synth_y,expected_returns

def reward_pred_loss(output, target):
    batch_size = output.size()[0]
    output = torch.squeeze(output)
    output = torch.log(output)
    res = torch.mul(output,target)
    return -torch.sum(res)

def mixed_synth_reward_pred_loss(output, target,loss_coef):
    batch_size = output.size()[0]
    output = torch.squeeze(output)
    output = torch.log(output)
    res = torch.mul(output,target)
    return -torch.multiply(torch.sum(res),loss_coef)
    # return -torch.sum(res)/batch_size


class RewardFunctionPR(torch.nn.Module):
    def __init__(self,GAMMA,use_extended_SF=False, n_features=6):
        super(RewardFunctionPR, self).__init__()
        self.n_features = n_features
        self.GAMMA = GAMMA
        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
        # if use_extended_SF:
        #     self.linear1 = torch.nn.Linear(self.n_features + 400, 1,bias=False) #TODO: HARDCODING THIS IN FOR NOW
        # else:
        #     self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False)
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False)


    def forward(self, phi):
        pr = torch.squeeze(self.linear1(phi))
        # left = pr[:,0:1]
        # right = pr[:,1:2]
        left_pred = torch.sigmoid(torch.subtract(pr[:,0:1],pr[:,1:2]))
        right_pred = torch.sigmoid(torch.subtract(pr[:,1:2],pr[:,0:1]))
        phi_logit = torch.stack([left_pred,right_pred],axis=1)
        return phi_logit

class RewardFunctionER(torch.nn.Module):
    def __init__(self,GAMMA,succ_feats,preference_weights, n_features=6):
        super(RewardFunctionER, self).__init__()
        self.n_features = n_features
        self.GAMMA = GAMMA
        # self.succ_feats = torch.tensor([succ_feats[0]],dtype=torch.double)
        self.succ_feats = torch.tensor(succ_feats,dtype=torch.double)
        # self.succ_feats_gt = torch.tensor(succ_feats_gt,dtype=torch.double)
        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False).double()

        self.softmax = torch.nn.Softmax(dim=1)

        if preference_weights is not None:
            self.rw = preference_weights[0][0]
            self.v_stw = preference_weights[0][1]
            self.v_s0w = preference_weights[0][2]
        else:
            self.rw = 1
            self.v_stw = 1
            self.v_s0w = 1

        self.T = 0.001

        #For debugging
        # self.smax_w_1 = []
        # self.smax_w_2 = []
        # self.softmax_temp = torch.nn.Softmax(dim=0)


    def get_vals(self,cords):
        selected_succ_feats =[self.succ_feats[:,x,y].double() for x,y in cords.long()]
        selected_succ_feats = torch.stack(selected_succ_feats)

        vs = self.linear1(selected_succ_feats)
        v_pi_approx = torch.sum(torch.mul(self.softmax(vs/self.T),vs),dim = 1)

        v_pi_approx = torch.squeeze(v_pi_approx)
        return v_pi_approx

    def get_er(self,phi):
        #only used for evaluation - calculates change in expected returns
        with torch.no_grad():
            pr = torch.squeeze(self.linear1(phi[:,:,0:6].double()))
            ss_x = torch.squeeze(phi[:,:,6:7])
            ss_y = torch.squeeze(phi[:,:,7:8])
            ss_cord_pairs = torch.stack([ss_x,ss_y], dim=1)

            es_x = torch.squeeze(phi[:,:,8:9])
            es_y = torch.squeeze(phi[:,:,9:10])
            es_cord_pairs = torch.stack([es_x,es_y], dim=1)

            #build list of succ fears for start/end states
            v_ss = self.get_vals(ss_cord_pairs)
            v_es = self.get_vals(es_cord_pairs)

            left_pr = pr[:,0:1]
            right_pr = pr[:,1:2]

            left_vf_ss = v_ss[:,0:1]
            right_vf_ss = v_ss[:,1:2]

            left_vf_es = v_es[:,0:1]
            right_vf_es = v_es[:,1:2]

            left_delta_v = torch.subtract(left_vf_es, left_vf_ss)
            right_delta_v = torch.subtract(right_vf_es, right_vf_ss)

            left_delta_er = torch.add(left_pr, left_delta_v)
            right_delta_er = torch.add(right_pr, right_delta_v)
            er = torch.stack([left_delta_er,right_delta_er],axis=1)
            return er

    def forward(self, phi):

        # forward_start = timeit.default_timer()
        pr = torch.squeeze(self.linear1(phi[:,:,0:6].double()))
        ss_x = torch.squeeze(phi[:,:,6:7])
        ss_y = torch.squeeze(phi[:,:,7:8])
        ss_cord_pairs = torch.stack([ss_x,ss_y], dim=1)


        es_x = torch.squeeze(phi[:,:,8:9])
        es_y = torch.squeeze(phi[:,:,9:10])
        es_cord_pairs = torch.stack([es_x,es_y], dim=1)


        #build list of succ fears for start/end states
        v_ss = self.get_vals(ss_cord_pairs)
        v_es = self.get_vals(es_cord_pairs)

        left_pr = pr[:,0:1]
        right_pr = pr[:,1:2]

        left_vf_ss = v_ss[:,0:1]
        right_vf_ss = v_ss[:,1:2]

        left_vf_es = v_es[:,0:1]
        right_vf_es = v_es[:,1:2]


        #apply weights learned from logistic regression (if it exists)
        left_pr = torch.multiply(left_pr, self.rw)
        right_pr = torch.multiply(right_pr, self.rw)

        left_vf_ss = torch.multiply(left_vf_ss, self.v_s0w)
        right_vf_ss = torch.multiply(right_vf_ss, self.v_s0w)

        left_vf_es = torch.multiply(left_vf_es, self.v_stw)
        right_vf_es = torch.multiply(right_vf_es, self.v_stw)

        #calculate change in expected return
        left_delta_v = torch.subtract(left_vf_es, left_vf_ss)
        right_delta_v = torch.subtract(right_vf_es, right_vf_ss)

        left_delta_er = torch.add(left_pr, left_delta_v)
        right_delta_er = torch.add(right_pr, right_delta_v)

        left_pred = torch.sigmoid(torch.subtract(left_delta_er, right_delta_er))
        right_pred = torch.sigmoid(torch.subtract(right_delta_er, left_delta_er))

        phi_logit = torch.stack([left_pred,right_pred],axis=1)

        # forward_end = timeit.default_timer()
        # print ("Forward time: " + str(forward_end - forward_start))
        # left_pred = torch.sigmoid(torch.subtract(left_pr,right_pr))
        # right_pred = torch.sigmoid(torch.subtract(right_pr,left_pr))
        # phi_logit = torch.stack([left_pred,right_pred],axis=1)
        return phi_logit

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_pref(arr,include_eps = True):
    #pref = get_pref([sigmoid(r1_er-r2_er), sigmoid(r2_er-r1_er)])
    # if prefrence_assum == "pr" and include_eps and abs(arr[0] - arr[1]) < 0.1:
    #     return [0.5,0.5]
    #
    # elif prefrence_assum == "er" and include_eps and (sigmoid(arr[0] - arr[1]) - sigmoid(arr[1] - arr[0])) < 0.1:
    #     return [0.5,0.5]
    #and abs(arr[0] - arr[1]) < 0.1

    if include_eps and abs(arr[0] - arr[1]) < 1:
        return [0.5,0.5]

    if (arr[0] > arr[1]):
        res = [1,0]
    elif (arr[1] > arr[0]):
        res = [0,1]
    else:
        res = [0.5,0.5]
    return res

def stringify(arr):
    return [str(a) for a in arr]


def save_synth_data_table(X,Y,Y_dist):
    X_str = stringify(X)
    Y_str = stringify(Y)
    Y_dist_str = stringify(Y_dist)
    df = pd.DataFrame(np.array([X_str, Y_dist_str, Y_str]).T,columns=["[psi_1, psi2]", "prefrence distribution", "sampled prefrence"])
    return df
    # df.to_csv("2021_12_11_selected_synthetic_data.csv", sep='\t')

def save_er_synth_data_table(X,Y,ER):
    X = np.array(X)
    PSI = X[:,:,0:6]
    STATES = X[:,:,6:10]

    PSI_str = stringify(PSI)
    STATES_str = stringify(STATES)
    Y_str = stringify(Y)
    ER_str = stringify(ER)

    df = pd.DataFrame(np.array([PSI_str, STATES_str, Y_str,ER_str]).T,columns=["[psi_1, psi2]", "[ssx,ssy, esx,esy]", "sampled prefrence","ER"])
    df.to_csv("er_synth_data.csv", sep='\t')
    return df

def generate_results_table(X,Y,model):
    logits = torch.squeeze(model(X))
    logits_str = stringify(logits)
    prs_str = stringify(np.dot(X[:,:,0:6], np.array([-1,50,-50,1,-1,-2]).T))
    states_str = stringify(X[:,:,6:10])
    X_str = stringify(X)
    Y_str = stringify(Y)

    df = pd.DataFrame(np.array([X_str, prs_str, states_str, logits_str, Y_str]).T,columns=["[psi_1, psi2]", "ground truth partial returns","[ss_x, ss_y, es_x, es_y]", "outputted logits", "ground truth prefrence"])

    df.to_csv("er_synth_reward_pred_debug.csv", sep='\t')

def model_eval(X,Y,w,model,is_testing=False,gt_rew_vec= [-1,50,-50,1,-1,-2],env=None,trajs=None):
    # w = np.array([ 2.2874,  0.4489,  0.0879,  0.0262, -0.0747,  0.9900]).T
    if gt_rew_vec is None:
        gt_rew_vec= [-1,50,-50,1,-1,-2]

    n_correct = 0
    total= 0
    index = 0
    model_out = model(torch.tensor(X))

    if prefrence_assum == "er":
        ERS = model.get_er(X).numpy()
        left_ers = ERS[:,0:1]
        right_ers = ERS[:,1:2]

    for x,y in zip(X,Y):
        y_f = get_pref(y)

        r_gt = np.dot(x[:,0:6], np.array(gt_rew_vec).T)

        if prefrence_assum == "pr":
            r =np.dot(x,w)
            # y_hat = [sigmoid(r[0]-r[1]),sigmoid(r[1]-r[0])]
            # y_gt = [sigmoid(r_gt[0]-r_gt[1]),sigmoid(r_gt[1]-r_gt[0])]
            y_hat = [r[0],r[1]]
            y_gt = [r_gt[0],r_gt[1]]
        elif prefrence_assum == "er":
            y_hat = [left_ers[index], right_ers[index]]

            # TODO: we are comparing logits to er values which itself might be incorrect
            # y_hat = ym.squeeze().detach().numpy()

            r1_er_gt, r2_er_gt = get_gt_er(x,gt_rew_vec=gt_rew_vec,env=env)
            # y_gt = [sigmoid(r1_er_gt-r2_er_gt), sigmoid(r2_er_gt-r1_er_gt)]
            y_gt = [r1_er_gt, r2_er_gt]

        res=get_pref(y_hat)

        # print (y_gt)
        # print (x2er.get(tuple(x)))

        y_gt_pref = get_pref(y_gt)
        # print (y_f)
        # print (y_gt)
        # print ("\n")
        # assert y_f[0] == y_gt_pref[0] and y_f[1] == y_gt_pref[1]
        total +=1

        if res[0] == y_f[0] and res[1] == y_f[1]:
            n_correct += 1
        # else:
        #     print (x)
        #     print ("\n")
        #     if trajs is not None:
        #         print ("***********************")
        #         print (trajs[index])
        #         print (x)
        #         print ("\n")

        # else:
        #     print (x)
        #     print (y_hat)
        #     print (y_gt)
        #     print (y_f)
        #     print ("\n")
        index+=1

    return (n_correct/total)

def disp_mmv(arr,title,axis):
    print ("Mean " + title + ": " + str(np.mean(arr,axis=axis)))
    print ("Median " + title + ": " + str(np.median(arr,axis=axis)))
    print (title + " Variance: " + str(np.var(arr,axis=axis)))

def run_single_set(model, optimizer, X_train, y_train, saX_type, sX_train, sy_train, loss_coef):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(X_train)
    y_pred = torch.clamp(y_pred,min=1e-35,max=None)#prevents prob pred of 0

    # Compute Loss
    # print (y_pred.shape)

    loss = reward_pred_loss(y_pred, y_train)
    batch_size = y_pred.size()[0]

    if saX_type == np.ndarray and "user" in mode:
        sy_pred = model(sX_train)
        sy_pred = torch.clamp(sy_pred,min=1e-35,max=None)#prevents prob pred of 0
        sloss = mixed_synth_reward_pred_loss(sy_pred, sy_train,loss_coef)
        s_batch_size = sy_pred.size()[0]
        loss += sloss
        loss /= (loss_coef*s_batch_size + batch_size)
    else:
        loss /= (batch_size)
    return loss, optimizer, model

def train(aX, ay, saX = None, say = None, loss_coef = None, plot_loss=True,preference_weights=None,gt_rew_vec=None,env=None,trajs=None):
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = train_test_split(aX, ay,test_size=2,random_state= 0,shuffle=True)

    # assert False
    X_train = format_X(X_train)
    y_train = format_y(y_train,"arr")

    X_test = format_X(X_test)
    y_test = format_y(y_test,"arr")

    if type(saX) == np.ndarray and "user" in mode:
        sX_train, sX_test, sy_train, sy_test = train_test_split(saX, say,test_size=.2,random_state= 0,shuffle=True)
        sX_train = format_X(sX_train)
        sy_train = format_y(sy_train,"arr")

        sX_test = format_X(sX_test)
        sy_test = format_y(sy_test,"arr")
    else:
        sX_train = None
        sy_train = None


    if prefrence_assum == "pr":
        if use_extended_SF:
            if env is not None:
                n_feats = 4*env.width*env.height + 6
            else:
                n_feats = 406
        else:
            n_feats = 6
        model = RewardFunctionPR(GAMMA,use_extended_SF=use_extended_SF,n_features=n_feats)
    elif prefrence_assum == "er":
        model = RewardFunctionER(GAMMA,succ_feats,preference_weights)

    if optimizer_add == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    train_accuracies = []
    test_accuracies = []

    learning_rates = [0.3,0.1,0.01]
    lr_i = 0
    spike_losses = []
    prev_loss = None

    loss = None
    # cycle = 0
    if optimizer_add == "line_search":
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        path = current_time + "_run"
        os.makedirs(path)

        torch.save({'epoch': 0,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss}, path + "/prev_model_params")

    best_loss = float("inf")
    best_loss_i = None
    batch_size = 64


    best_total_loss = float("inf") #best loss over entire training cycle

    for param in model.parameters():
        best_weights = param.detach().numpy()[0]



    if type(saX) == np.ndarray and "user" in mode:
        synth_data_r = sX_train.size()[0]/(X_train.size()[0] + sX_train.size()[0])
        synth_batch_size = int ((batch_size*synth_data_r)/(1-synth_data_r))

    for epoch in range(N_ITERS):

        if optimizer_add == "mini_batch":
            permutation = torch.randperm(X_train.size()[0])

            if type(saX) == np.ndarray and "user" in mode:
                permutation_synth = torch.randperm(sX_train.size()[0])

            total_batch_loss = 0
            for i in range(0,X_train.size()[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                if type(saX) == np.ndarray and "user" in mode:
                    indices_synth = permutation_synth[i:i+synth_batch_size]
                    batch_sx, batch_sy = sX_train[indices_synth], sy_train[indices_synth]
                else:
                    batch_sx = None
                    batch_sy = None

                loss, optimizer, model = run_single_set(model, optimizer, batch_x, batch_y, type(saX), batch_sx, batch_sy, loss_coef)

                if loss.detach().numpy() < best_total_loss:
                    best_total_loss = loss.detach().numpy()
                    for param in model.parameters():
                        best_weights = param.detach().numpy()[0].copy()


                # Backward pass
                loss.backward()
                optimizer.step()

                total_batch_loss += loss.detach()

            print(total_batch_loss/batch_size)
            losses.append(total_batch_loss/batch_size)


            continue

        elif optimizer_add == "line_search":
            if lr_i > 2:
                #choose best learning rate and move on

                checkpoint = torch.load(path + "/prev_model_params_" + str(best_loss_i))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # iter = checkpoint['epoch']
                loss = checkpoint['loss']

                lr_i = 0
                best_loss = float("inf")
                best_loss_i = None


                # print (loss)
                losses.append(loss)

                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss}, path + "/prev_model_params")
            else:
                #continue line search
                checkpoint = torch.load(path + "/prev_model_params")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # iter = checkpoint['epoch']
                loss = checkpoint['loss']

            for g in optimizer.param_groups:
                g['lr'] = learning_rates[lr_i]
        elif optimizer_add == "cyclic":
            for g in optimizer.param_groups:
                g['lr'] = learning_rates[lr_i]
            lr_i+=1
            lr_i = lr_i % len(learning_rates)

        loss, optimizer, model = run_single_set(model, optimizer, X_train, y_train, type(saX), sX_train, sy_train, loss_coef)

        if loss.detach().numpy() < best_total_loss:
            best_total_loss = loss.detach().numpy()
            for param in model.parameters():
                best_weights = param.detach().numpy()[0].copy()
            # print ("Updating: bl " + str(best_total_loss) + " best bw " + str(best_weights))

        # Backward pass
        loss.backward()
        optimizer.step()

        if optimizer_add != "line_search":
            # print (loss)
            losses.append(loss)

        if optimizer_add == "line_search":
            torch.save({'epoch': 0,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss}, path + "/prev_model_params_" + str(lr_i))

            next_loss, _, _ = run_single_set(model, optimizer, X_train, y_train, type(saX), sX_train, sy_train, loss_coef)
            if next_loss < best_loss:
                best_loss = next_loss
                best_loss_i = lr_i

            lr_i+=1

        prev_loss = loss

        # tracks accuracy during training (this significantly slows things down)
        # for param in model.parameters():
        #     train_accuracy = model_eval(X_train,y_train,param.detach().numpy()[0].T)
        #     test_accuracy = model_eval(X_test,y_test,param.detach().numpy()[0].T)
        #     train_accuracies.append(1-train_accuracy)
        #     test_accuracies.append(1-test_accuracy)


    # fig1, ax1 = plt.subplots()
    # ax1.plot(losses, color = "b")
    if plot_loss:
        fig2, ax2 = plt.subplots()
        ax2.plot(losses,color = "b")
        # ax2.plot(train_accuracies,color = "r")
        # ax2.plot(test_accuracies, color = "g")
        plt.plot([0,len(losses)], [min(losses), min(losses)], marker = 'o',color = "black")

        plt.show()
    #
    # print ("best_weights: " + str(best_weights))
    # print ("best_loss: " + str(best_total_loss))

    if prefrence_assum == "er":
        for param in model.parameters():
            param.data = nn.parameter.Parameter(torch.tensor(best_weights).double())


    train_loss = reward_pred_loss(model(X_train), y_train).detach().numpy()
    # test_loss = reward_pred_loss(model(X_test), y_test).detach().numpy()

    train_batch_size = y_train.size()[0]
    # test_batch_size = y_test.size()[0]

    if type(saX) == np.ndarray and "user" in mode:
        train_loss += mixed_synth_reward_pred_loss(model(sX_train), sy_train, loss_coef).detach().numpy()
        test_loss += mixed_synth_reward_pred_loss(model(sX_test), sy_test, loss_coef).detach().numpy()

        train_loss /= (train_batch_size + loss_coef*sy_train.size()[0])
        test_loss /= (test_batch_size + loss_coef*sy_test.size()[0])
    else:
        train_loss /= (train_batch_size)
        # test_loss /= (test_batch_size)

    print ("training loss: " + str(train_loss))
    # print ("testing loss: " + str(test_loss))

    # generate_results_table(X_test,y_test,model)
    #
    for param in model.parameters():
        reward_vector = param.detach().numpy()

    training_acc = model_eval(X_train,y_train,reward_vector.T,model,is_testing=True, gt_rew_vec=gt_rew_vec,env=env,trajs=trajs)

    # testing_acc = model_eval(X_test,y_test,reward_vector.T,model,is_testing=True, gt_rew_vec=gt_rew_vec,env=env)
    # if type(saX) == np.ndarray and "user" in mode:
    #     testing_acc += model_eval(sX_train,sy_train,reward_vector.T,model,is_testing=True, gt_rew_vec=gt_rew_vec,env=env)
    #     testing_acc /=2
    #
    #     training_acc += model_eval(sX_test,sy_test,reward_vector.T,model,is_testing=True, gt_rew_vec=gt_rew_vec,env=env)
    #     training_acc /=2

    if len(reward_vector) == 1:
        reward_vector = reward_vector[0]
    print ("Training accuracy: " + str(training_acc))
    # print ("Testing accuracy: " + str(testing_acc))
    print ("Learned reward weights:")
    print (reward_vector)

    # f = open("BACKUP_DEBUG_OUT.txt", "w")
    # f.write("Testing accuracy: " + str(testing_acc) + "\n")
    # f.write("Training accuracy: " + str(training_acc) + "\n")
    # f.write(str(reward_vector) + "\n")
    # f.close()

    # if LINE_SEARCH_ON:
    #     os.remove(path)

    test_loss = float("inf")
    testing_acc = float("inf")

    return reward_vector,losses,train_loss, test_loss, training_acc, testing_acc



def train_from_checkpoint(aX, ay, new_LR, saX = None, say = None, loss_coef = None, plot_loss=True):
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = train_test_split(aX, ay,test_size=.2,random_state= 0,shuffle=True)

    # assert False
    X_train = format_X(X_train)
    y_train = format_y(y_train,"arr")

    X_test = format_X(X_test)
    y_test = format_y(y_test,"arr")

    if type(saX) == np.ndarray and "user" in mode:
        sX_train, sX_test, sy_train, sy_test = train_test_split(saX, say,test_size=.2,random_state= 0,shuffle=True)
        sX_train = format_X(sX_train)
        sy_train = format_y(sy_train,"arr")

        sX_test = format_X(sX_test)
        sy_test = format_y(sy_test,"arr")


    if prefrence_assum == "pr":
        model = RewardFunctionPR(use_extended_SF=use_extended_SF)
    elif prefrence_assum == "er":
        model = RewardFunctionER(succ_feats)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)


    checkpoint = torch.load("model_params_1")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter = checkpoint['epoch']
    loss = checkpoint['loss']

    for g in optimizer.param_groups:
        g['lr'] = new_LR

    for epoch in range(2):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train)
        y_pred = torch.clamp(y_pred,min=1e-35,max=None)#prevents prob pred of 0

        # Compute Loss
        loss = reward_pred_loss(y_pred, y_train)
        batch_size = y_pred.size()[0]

        if type(saX) == np.ndarray and "user" in mode:
            sy_pred = model(sX_train)
            sy_pred = torch.clamp(sy_pred,min=1e-35,max=None)#prevents prob pred of 0

            sloss = mixed_synth_reward_pred_loss(sy_pred, sy_train,loss_coef)
            s_batch_size = sy_pred.size()[0]
            loss += sloss
            loss /= (loss_coef*s_batch_size + batch_size)
        else:
            loss /= (batch_size)

        loss.backward()
        optimizer.step()

        # print (loss)
    return loss


def train_from_human_data(X_copy, r_copy, y_copy, ses_copy, preference_weights):
    X_copy, y_copy, X_copy_sytnh, y_copy_synth,_,loss_coef = clean_y(X_copy, r_copy,y_copy,ses_copy)
    aX, ay = augment_data(X_copy,y_copy,"arr")

    # print ("finding reward vector...")
    rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay, None, None, loss_coef, plot_loss=False, preference_weights=preference_weights)

    # print ("performing value iteration...")
    V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

    # print ("following policy...")
    # avg_return = follow_policy(Q,100,viz_policy=False)
    pi = build_pi(Q)
    V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
    avg_return = np.sum(V_under_gt)/92 #number of possible start states
    print ("average return following learned policy: ")
    print (avg_return)
    print ("--------------------------------------------------------------------\n")

#
# env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
# random_pi = build_random_policy(env=env)
# V_under_random_pi = iterative_policy_evaluation(random_pi,rew_vec = np.array([-1,50,-50,1,-1,-2]), GAMMA=GAMMA,env=env)
# random_avg_return = np.sum(V_under_random_pi)/env.n_starts
# print ("random policies avg return: ")
# print (random_avg_return)
#
# assert False



all_reward_vecs = []
all_train_losses = []
all_test_losses = []
all_training_acc= []
all_testing_acc = []
all_total_training_losses = []
all_avg_returns = []
#
# trajpairs = []
# phis = []

if use_random_MDPs:

    num_near_opt = 0
    num_above_random = 0
    all_avg_returns = []
    all_scaled_returns = []
    for trial in range(0,30):
        # all_trajs = None
        # all_X = np.load("random_MDPs/MDP_" + str(trial) +"all_X.npy").tolist()
        # all_r = np.load("random_MDPs/MDP_" + str(trial) +"all_r.npy")
        # print (all_X.shape)
        # trial = "test1_"
        all_trajs = np.load("random_MDPs/MDP_" + str(trial) +"all_trajs.npy").tolist()
        all_ses = np.load("random_MDPs/MDP_" + str(trial) +"all_new_ses.npy").tolist()
        gt_rew_vec = np.load("random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy")
        succ_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_feats.npy")
        with open("random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
            env = pickle.load(rf)

        print (env.board)
        print ("============")

        if len(all_trajs)>3000:
            print ("SUBSAMPLING TRAJ PAIRS")
            idx = np.random.choice(np.arange(len(all_trajs)), 3000, replace=False)
            all_trajs = np.array(all_trajs)[idx]
            all_ses = np.array(all_ses)[idx]

        all_X = []
        all_r = []

        # print (env.board)

        for i_, traj_pair in enumerate(all_trajs):
            phi_dis1,phi1 = find_reward_features(traj_pair[0],env,use_extended_SF=use_extended_SF,GAMMA=GAMMA)
            phi_dis2,phi2 = find_reward_features(traj_pair[1],env,use_extended_SF=use_extended_SF,GAMMA=GAMMA)

            # print (traj_pair[0])
            # print (all_ses[i_][0])
            # print ("\n")

            all_r.append([np.dot(gt_rew_vec,phi_dis1[0:6]), np.dot(gt_rew_vec,phi_dis2[0:6])])
            all_X.append([phi_dis1, phi_dis2])

        # print ("processed")
        pr_X,synth_max_y,expected_returns = generate_synthetic_prefs(all_X,all_r,all_ses,mode,gt_rew_vec=np.array(gt_rew_vec),env=env)
        aX, ay = augment_data(pr_X,synth_max_y,"arr")
        rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay,plot_loss=False,gt_rew_vec=np.array(gt_rew_vec),env=env,trajs=all_trajs)


        print ("# of synthetic prefrences: " + str(len(pr_X)))
        print ("Ground truth reward vector: " + str(gt_rew_vec))
        V,Q = value_iteration(rew_vec =rew_vect,GAMMA=0.999,env=env)
        pi = build_pi(Q,env=env)

        V_under_gt = iterative_policy_evaluation(pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)

        avg_return = np.sum(V_under_gt)/env.n_starts
        all_avg_returns.append(avg_return)
        print ("average return following learned policy: ")
        print (avg_return)

        gt_avg_return = get_gt_avg_return(gt_rew_vec=gt_rew_vec, env=env, GAMMA=0.999)

        #build random policy
        random_pi = build_random_policy(env=env)
        V_under_random_pi = iterative_policy_evaluation(random_pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)
        random_avg_return = np.sum(V_under_random_pi)/env.n_starts
        print ("random policies avg return: ")
        print (random_avg_return)

        #scale everything: f(z) = (z-x) / (y-x)
        scaled_return = (avg_return - random_avg_return)/(gt_avg_return - random_avg_return)
        all_scaled_returns.append(scaled_return)

        print ("scaled return following learned policy: " + str(scaled_return))

        random_policy_data.changed_gt_rew_vec = False
        if (scaled_return >= 0.9):
            num_near_opt+=1
        if (scaled_return >= 0):
            num_above_random+=1

        # assert False

        print ("=================================================================================\n")

    np.save("random_MDPs/main_avg_return_" + str(mode) + "_" + str(prefrence_model) + "_" + str(prefrence_assum) + str(use_extended_SF) + "_" +  str(GAMMA) + ".npy",all_avg_returns)
    print ("% of MDPs where near optimal performance was achieved: " + str(100*(num_near_opt/30)) + "%")
    print ("% of MDPs where better than random performance was achieved: " + str(100*(num_above_random/30)) + "%")



    #
    # num_near_opt = 0
    # num_above_random = 0
    # all_avg_returns = []
    # all_scaled_returns = []
    # for trial in range(30,60):
    #
    #     # all_X = np.load("random_MDPs/MDP_" + str(trial) +"all_X.npy").tolist()
    #     # print (all_X.shape)
    #
    #     all_trajs = np.load("random_MDPs/MDP_" + str(trial) +"all_trajs.npy")
    #     all_r = np.load("random_MDPs/MDP_" + str(trial) +"all_r.npy")
    #     all_ses = np.load("random_MDPs/MDP_" + str(trial) +"all_ses.npy")
    #     gt_rew_vec = np.load("random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy")
    #     succ_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_feats.npy")
    #     with open("random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
    #         env = pickle.load(rf)
    #
    #     # apply gamma to state features for reward learning
    #     all_X = []
    #     for traj_pair in all_trajs.tolist():
    #         phi_dis1,phi1 = find_reward_features(traj_pair[0],env,use_extended_SF=use_extended_SF,GAMMA=1)
    #         phi_dis2,phi2 = find_reward_features(traj_pair[1],env,use_extended_SF=use_extended_SF,GAMMA=1)
    #
    #         all_X.append([phi_dis1, phi_dis2])
    #
    #     pr_X,synth_max_y,expected_returns = generate_synthetic_prefs(all_X,all_r.tolist(),all_ses.tolist(),mode,gt_rew_vec=np.array(gt_rew_vec),env=env)
    #
    #     aX, ay = augment_data(pr_X,synth_max_y,"arr")
    #
    #     # print ("finding reward vector...")
    #     rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay,plot_loss=False,gt_rew_vec=np.array(gt_rew_vec),env=env)
    #     print ("# of synthetic prefrences: " + str(len(pr_X)))
    #
    #     print ("Ground truth reward vector: " + str(gt_rew_vec))
    #     # print ("performing value iteration...")
    #     V,Q = value_iteration(rew_vec =rew_vect,GAMMA=0.999,env=env)
    #     # print ("following policy...")
    #     pi = build_pi(Q,env=env)
    #     V_under_gt = iterative_policy_evaluation(pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)
    #     avg_return = np.sum(V_under_gt)/env.n_starts
    #     print ("average return following learned policy: ")
    #     print (avg_return)
    #     all_avg_returns.append(avg_return)
    #
    #
    #     gt_avg_return = get_gt_avg_return(gt_rew_vec=gt_rew_vec, env=env, GAMMA=0.999)
    #
    #     #build random policy
    #     random_pi = build_random_policy(env=env)
    #     V_under_random_pi = iterative_policy_evaluation(random_pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)
    #     random_avg_return = np.sum(V_under_random_pi)/env.n_starts
    #     print ("random policies avg return: ")
    #     print (random_avg_return)
    #
    #     #scale everything: f(z) = (z-x) / (y-x)
    #     scaled_return = (avg_return - random_avg_return)/(gt_avg_return - random_avg_return)
    #     all_scaled_returns.append(scaled_return)
    #
    #     print ("scaled return following learned policy: " + str(scaled_return))
    #
    #     random_policy_data.changed_gt_rew_vec = False
    #
    #     if (scaled_return >= 0.9):
    #         num_near_opt+=1
    #     if (scaled_return >= 0):
    #         num_above_random+=1
    #
    #     print ("=================================================================================\n")
    #
    # np.save("random_MDPs/avg_return_" + str(mode) + "_" + str(prefrence_model) + "_" + str(prefrence_assum) + str(use_extended_SF) + "_" +  str(GAMMA) + ".npy",all_avg_returns)
    # print ("% of MDPs where near optimal performance was achieved: " + str(100*(num_near_opt/30)) + "%")
    # print ("% of MDPs where better than random performance was achieved: " + str(100*(num_above_random/30)) + "%")

elif mode == "deterministic_user_data":
    vf_X, vf_r, vf_y, vf_ses, pr_X, pr_r, pr_y, pr_ses, none_X, none_r, none_y, none_ses = get_all_statistics_aug_human()


    X_copy = none_X.copy()
    r_copy = none_r.copy()
    y_copy = none_y.copy()
    ses_copy = none_ses.copy()

    # X_copy = pr_X.copy()
    # r_copy = pr_r.copy()
    # y_copy = pr_y.copy()
    # ses_copy = pr_ses.copy()

    # X_copy = vf_X.copy()
    # r_copy = vf_r.copy()
    # y_copy = vf_y.copy()
    # ses_copy = vf_ses.copy()


    print ("Number of human prefs used: " + str(len(X_copy)))


    X_copy, y_copy, X_copy_sytnh, y_copy_synth,_,loss_coef = clean_y(X_copy, r_copy,y_copy,ses_copy)

    # X_copy.extend(X_copy_sytnh)
    # y_copy.extend(y_copy_synth)

    # aX, ay = augment_data(pr_X,pr_y,"scalar")
    aX, ay = augment_data(X_copy,y_copy,"arr")
    # aX_synth, ay_synth = augment_data(X_copy_sytnh,y_copy_synth,"arr")

    print ("finding reward vector...")
    rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay, None, None, loss_coef, plot_loss=False)

    print ("performing value iteration...")
    V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

    print ("following policy...")
    # avg_return = follow_policy(Q,100,viz_policy=False)
    pi = build_pi(Q)
    V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
    avg_return = np.sum(V_under_gt)/92 #number of possible start states
    print ("average return following learned policy: ")
    print (avg_return)


elif mode == "user_data":
    vf_X, vf_r, vf_y, vf_ses, pr_X, pr_r, pr_y, pr_ses, none_X, none_r, none_y, none_ses = get_all_statistics(include_dif_traj_lengths=include_dif_traj_lengths)


    # X_user = none_X.copy()
    # r_user = none_r.copy()
    # y_user = none_y.copy()
    # ses_copy = none_ses.copy()

    X_user = pr_X.copy()
    r_user = pr_r.copy()
    y_user = pr_y.copy()
    ses_copy = pr_ses.copy()

    # X_user = vf_X.copy()
    # r_user = vf_r.copy()
    # y_user = vf_y.copy()
    # ses_copy = vf_ses.copy()

    # X_copy, y_copy, X_copy_sytnh, y_copy_synth = clean_y(X_copy,r_copy,y_copy)
    dfs = []
    for prob_iter in range(n_prob_iters):
        print ("==========================Trial " + str(prob_iter)+" ==========================")

        # X_copy = none_X.copy()
        # r_copy = none_r.copy()
        # y_copy = none_y.copy()

        X_copy, y_copy, X_copy_sytnh, y_copy_synth,y_dist_copy,loss_coef = clean_y(X_user,r_user,y_user,ses_copy)

        df = save_synth_data_table(X_copy_sytnh,y_copy_synth,y_dist_copy)
        dfs.append(df)
        # continue
        print (len(X_copy) + len(X_copy_sytnh))

        aX, ay = augment_data(X_copy,y_copy,"arr")
        aX_synth, ay_synth = augment_data(X_copy_sytnh,y_copy_synth,"arr")

        # aX = X_copy
        # ay = y_copy

        # X_copy_sytnh.extend(aX)
        # y_copy_synth.extend(ay)
        # aX_synth = np.array(X_copy_sytnh)
        # ay_synth = np.array(y_copy_synth)

        print ("finding reward vector...")
        rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay, aX_synth, ay_synth,loss_coef, plot_loss=False)


        all_reward_vecs.append(rew_vect)
        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_training_acc.append(training_acc)
        all_testing_acc.append(testing_acc)
        all_total_training_losses.append(all_losses)

        print ("performing value iteration...")
        V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

        print ("following policy...")
        # avg_return = follow_policy(Q,100,viz_policy=False)
        pi = build_pi(Q)
        V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
        avg_return = np.sum(V_under_gt)/92
        all_avg_returns.append(avg_return)
        print ("============================================================")

    # with pd.ExcelWriter('2021_12_11_selected_synthetic_data,n=100.xlsx') as writer:
    #     df_iter = 0
    #     for df in dfs:
    #         df.head(100).to_excel(writer, sheet_name="Iter_" + str(df_iter) + "_synthetic data")
    #         df_iter+=1
    print ("\n\n")
    print ("Across all " + str(n_prob_iters) + " trials,")

    disp_mmv(all_testing_acc, "Testing Accuracy",None)
    disp_mmv(all_training_acc, "Training Accuracy",None)

    print (all_avg_returns)
    disp_mmv(all_avg_returns, "Average Return",None)
    disp_mmv(all_train_losses, "Training Loss",None)
    disp_mmv(all_test_losses, "Testing Loss",None)
    disp_mmv(all_reward_vecs, "Reward Vector",0)
    np.save("n=100_all_reward_vecs.npy",all_reward_vecs)

    plt.plot(np.array(all_total_training_losses).T)
    plt.show()
elif mode == "sigmoid":
    vf_X, vf_r, vf_y, vf_ses, pr_X, pr_r, pr_y, pr_ses, none_X, none_r, none_y, none_ses = get_all_statistics(include_dif_traj_lengths=include_dif_traj_lengths,use_extended_SF=use_extended_SF,GAMMA=GAMMA) #gamma used for labeling

    pr_X.extend(vf_X)
    pr_r.extend(vf_r)
    pr_ses.extend(vf_ses)

    pr_X.extend(none_X)
    pr_r.extend(none_r)
    pr_ses.extend(none_ses)


    # pr_X, pr_r, pr_ses = get_N_length_dataset(21)


    pr_X_copy = pr_X.copy()
    pr_r_copy = pr_r.copy()


    for prob_iter in range(n_prob_iters):
        pr_X,synth_max_y,_ = generate_synthetic_prefs(pr_X_copy,pr_r_copy,pr_ses,mode)
        print (len(pr_X))
        # aX, ay = augment_data(pr_X,pr_y,"scalar")
        aX, ay = augment_data(pr_X,synth_max_y,"arr")


        # learning_rates = []
        # # # #20 spots between 0.1 and 0.01
        # for i in range(21):
        #     learning_rates.append(10-i*0.45)
        # for i in range(21):
        #     learning_rates.append(1-i*0.045)
        # for i in range(21):
        #     learning_rates.append(0.1-i*0.0045)
        #
        # spike_losses = []
        #
        # for n_lr in learning_rates:
        #     l = train_from_checkpoint(aX, ay, n_lr, plot_loss=True)
        #     spike_losses.append(l)
        # np.save("learning_rates6.npy", learning_rates)
        # np.save("spike_losses6.npy", spike_losses)
        # assert False


        print ("==========================Trial " + str(prob_iter)+" ==========================")
        print ("finding reward vector...")
        rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay,plot_loss=False)
        all_reward_vecs.append(rew_vect)
        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_training_acc.append(training_acc)
        all_testing_acc.append(testing_acc)
        all_total_training_losses.append(all_losses)

        print (rew_vect)

        print ("performing value iteration...")
        V,Q = value_iteration(rew_vec =rew_vect,GAMMA=0)

        print ("following policy...")
        # avg_return = follow_policy(Q,100,viz_policy=False)
        pi = build_pi(Q)
        V_under_gt = iterative_policy_evaluation(pi, GAMMA=0.999)
        avg_return = np.sum(V_under_gt)/92
        all_avg_returns.append(avg_return)
        print ("============================================================")

    print ("\n\n")
    print ("GAMMA: " + str(GAMMA))
    print ("Across all " + str(n_prob_iters) + " trials,")

    disp_mmv(all_testing_acc, "Testing Accuracy",None)
    disp_mmv(all_training_acc, "Training Accuracy",None)
    disp_mmv(all_avg_returns, "Average Return",None)
    get_gt_avg_return(GAMMA=0.999)
    disp_mmv(all_train_losses, "Training Loss",None)
    disp_mmv(all_test_losses, "Testing Loss",None)
    disp_mmv(all_reward_vecs, "Reward Vector",0)


    plt.plot(np.array(all_total_training_losses).T)
    plt.show()
else:
    vf_X, vf_r, vf_y, vf_ses, pr_X, pr_r, pr_y, pr_ses, none_X, none_r, none_y, none_ses = get_all_statistics(include_dif_traj_lengths=include_dif_traj_lengths,use_extended_SF=use_extended_SF,GAMMA=GAMMA)
    pr_X.extend(vf_X)
    pr_r.extend(vf_r)
    pr_ses.extend(vf_ses)

    pr_X.extend(none_X)
    pr_r.extend(none_r)
    pr_ses.extend(none_ses)



    #
    # print (len(none_X))
    # print (len(pr_X))
    # print (len(vf_X))

    # for traj_length in range (3,24,3):
    #     print ("\n==========================================")
    #     print ("TRAJ LENGTH: " + str(traj_length))
    #
    #     pr_X, pr_r, pr_ses = get_N_length_dataset(traj_length)


    pr_X,synth_max_y,expected_returns = generate_synthetic_prefs(pr_X,pr_r,pr_ses,mode)

    print ("# of synthetic prefrences: " + str(len(pr_X)))
    # save_er_synth_data_table(pr_X,synth_max_y,expected_returns)

    # aX, ay = augment_data(pr_X,pr_y,"scalar")
    aX, ay = augment_data(pr_X,synth_max_y,"arr")


    print ("finding reward vector...")


    # learning_rates = []
    # # # #20 spots between 0.1 and 0.01
    # for i in range(21):
    #     learning_rates.append(10-i*0.45)
    # for i in range(21):
    #     learning_rates.append(1-i*0.045)
    # for i in range(21):
    #     learning_rates.append(0.1-i*0.0045)
    #
    # spike_losses = []
    #
    # for n_lr in learning_rates:
    #     l = train_from_checkpoint(aX, ay, n_lr, plot_loss=True)
    #     spike_losses.append(l)
    # np.save("learning_rates5.npy", learning_rates)
    # np.save("spike_losses5.npy", spike_losses)
    # assert False

    rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay,plot_loss=False)
    print ("performing value iteration...")
    V,Q = value_iteration(rew_vec =rew_vect,GAMMA=0.999)

    # print ("following policy...")
    # follow_policy(Q,100,viz_policy=True)
    print ("following policy...")
    pi = build_pi(Q)
    V_under_gt = iterative_policy_evaluation(pi, GAMMA=0.999)
    avg_return = np.sum(V_under_gt)/92
    print ("average return following learned policy: ")
    print (avg_return)

    get_gt_avg_return(GAMMA=0.999)
