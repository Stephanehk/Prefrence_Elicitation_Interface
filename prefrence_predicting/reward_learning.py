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

from grid_world import GridWorldEnv
from load_training_data import get_all_statistics,find_end_state
from value_iteration import value_iteration, follow_policy, learn_successor_feature,get_gt_avg_return,build_pi,iterative_policy_evaluation
from generate_random_policies import generate_all_policies, calc_value

keep_ties = False
n_prob_samples = 1
n_prob_iters = 30
GAMMA=0.999
include_dif_traj_lengths = False
# mode = "deterministic_user_data"
# mode = "user_data"
mode = "sigmoid"
gt_V,_ = value_iteration(GAMMA=0.999)
LR = 1
N_ITERS = 10000
# mode = "deterministic"

prefrence_assum = "er"
if prefrence_assum == "er":
    # print("generating policies...")
    # succ_feats, pis = generate_all_policies(100,GAMMA)
    # np.save("100_succ_feats.npy",succ_feats)
    # np.save("100_pis.npy",pis)
    # print("finished")
    # assert False
    succ_feats = np.load("100_succ_feats.npy",allow_pickle=True)
    pis = np.load("100_pis.npy",allow_pickle=True)
    succ_feats_gt = np.load("succ_feats.npy",allow_pickle=True)
    pis_gt = np.load("pis.npy",allow_pickle=True)

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

def get_gt_er (x):
    #calculates gt expected return
    w = [-1,50,-50,1,-1,-2]
    if (torch.is_tensor(x)):
        x = x.detach().numpy()


    t1_ss = [int(x[0][6]), int(x[0][7])]
    t1_es = [int(x[0][8]), int(x[0][9])]

    t2_ss = [int(x[1][6]), int(x[1][7])]
    t2_es = [int(x[1][8]), int(x[1][9])]


    x = np.array(x)
    r = np.dot(x[:,0:6],w)

    r1_er = r[0] + calc_value(t1_es) - calc_value(t1_ss)
    r2_er = r[1] + calc_value(t2_es) - calc_value(t2_ss)
    return r1_er, r2_er

def clean_y(X,R,Y):
    formatted_y = []
    out_X = []

    synth_formatted_y = []
    synth_y_dist = []
    synth_out_X = []
    n_unique_trajs = 0

    for x,r,y in zip(X,R,Y):
        x = [list(x[0]),list(x[1])]

        if y != None and not is_subset(x,out_X):
            n_unique_trajs += 1

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
            if is_subset(x,synth_out_X):
                continue

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
                    #NOT DEFINED
                    assert False
                else:
                    synth_formatted_y.append(np.array(get_pref(r,False)))
                synth_out_X.append(x)
        else:
            formatted_y.append(np.array(y))
            out_X.append(x)

    loss_coef = (len(out_X)/n_unique_trajs)/n_prob_samples
    loss_coef = min(1,loss_coef)
    print ("loss coef: " + str(loss_coef))

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

def generate_synthetic_prefs(pr_X,rewards,sess,mode):
    synth_y = []
    non_redundent_pr_X = []
    expected_returns = []
    for r,x,ses in zip(rewards,pr_X,sess):

        if prefrence_assum == "pr":
            x_f = [list(x[0]),list(x[1])]
        #change x to include start end state for each trajectory
        if prefrence_assum == "er":
            x[0] = list(x[0])
            x[0].extend([ses[0][0][0],ses[0][0][1], ses[0][1][0],ses[0][1][1]])

            x[1] = list(x[1])
            x[1].extend([ses[1][0][0],ses[1][0][1], ses[1][1][0],ses[1][1][1]])
            x_f = [x[0],x[1]]

        #adds discounting to prefrences
        # t1_n_trans = x[0][0] + x[0][1] + x[0][2] + x[0][5]
        # t2_n_trans = x[1][0] + x[1][1] + x[1][2] + x[1][5]
        # r[0] *=(np.power(GAMMA,t1_n_trans))
        # r[1] *=(np.power(GAMMA,t2_n_trans))
        #remove duplicates
        if is_subset(x_f,non_redundent_pr_X):
            continue

        if mode == "sigmoid":

            if prefrence_assum == "er":
                r1_er, r2_er = get_gt_er(x)
                if not keep_ties and r1_er == r2_er:
                    continue

            if prefrence_assum == "pr" and not keep_ties and r[1] == r[0]:
                continue

            for n_samp in range(n_prob_samples):
                if prefrence_assum == "pr":
                    r1_prob = sigmoid(r[0]-r[1])
                    r2_prob = sigmoid(r[1]-r[0])

                elif prefrence_assum == "er":
                    r1_prob = sigmoid(r1_er-r2_er)
                    r2_prob = sigmoid(r2_er-r1_er)

                num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                if num == 1:
                    pref = [1,0]
                elif num == 0:
                    pref = [0,1]
                synth_y.append(pref)
                non_redundent_pr_X.append(x_f)
        else:
            if prefrence_assum == "er":
                r1_er, r2_er = get_gt_er (x)
                # pref = get_pref([sigmoid(r1_er-r2_er), sigmoid(r2_er-r1_er)])
                pref = get_pref([r1_er, r2_er])

                # x2er[tuple(x[0])] = r1_er
                # x2er[tuple(x[1])] = r2_er
                # x2r[tuple(x[0])] = np.dot(x[0][0:6],[-1,50,-50,1,-1,-2])
                # x2r[tuple(x[1])] = np.dot(x[1][0:6],[-1,50,-50,1,-1,-2])
            else:
                pref = get_pref(r)

            # if pref != [0.5,0.5]:
            #     continue

            if pref == [0.5,0.5] and not keep_ties:
                continue

            if prefrence_assum == "er":
                expected_returns.append([r1_er, r2_er])
            synth_y.append(pref)
            non_redundent_pr_X.append(x_f)
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
    def __init__(self,n_features=6):
        super(RewardFunctionPR, self).__init__()
        self.n_features = n_features
        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
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
    def __init__(self,succ_feats,n_features=6):
        super(RewardFunctionER, self).__init__()
        self.n_features = n_features
        # self.succ_feats = torch.tensor([succ_feats[0]],dtype=torch.double)
        self.succ_feats = torch.tensor(succ_feats,dtype=torch.double)
        self.succ_feats_gt = torch.tensor(succ_feats_gt,dtype=torch.double)

        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False).double()

        self.softmax = torch.nn.Softmax(dim=1)
        self.softmax_temp = torch.nn.Softmax(dim=0)

    # def calc_value(self, state):
    #     x,y = state
    #     return torch.max(torch.tensor([torch.squeeze(self.linear1(succ_feats[i][x][y])) for i in range(len(self.succ_feats))]))
    def get_vals_old(self,d_xs,d_ys):
        v_pi_approx = []
        #
        for x,y in zip(d_xs,d_ys):
            traj_vs = []
            for traj in range (len(x)):
                ss_vs = []
                for i in range(len(self.succ_feats)):
                    # print (int(x[traj]))
                    # print (int(y[traj]))
                    succ_phi = self.succ_feats[i][int(x[traj])][int(y[traj])]
                    succ_phi = succ_phi.double()
                    # v = self.linear1(succ_phi)
                    v = torch.dot(succ_phi,torch.tensor([-1,50,-50,1,-1,-2],dtype=torch.double))
                    ss_vs.append(v)
                ss_vs = torch.tensor(ss_vs)
                max_ = torch.sum(torch.mul(self.softmax_temp(ss_vs),ss_vs))

                traj_vs.append(max_)
            # print (traj_vs)
            v_pi_approx.append(torch.tensor(traj_vs))
        return torch.stack(v_pi_approx)

        # v_pi_approx = []
        # #
        # for x,y in zip(d_xs,d_ys):
        #     traj_vs = []
        #     for traj in range (len(x)):
        #         succ_phi = self.succ_feats_gt[0][int(x[traj])][int(y[traj])]
        #         succ_phi = succ_phi.double()
        #         v = self.linear1(succ_phi)
        #
        #         # v = torch.dot(succ_phi,torch.tensor([-1,50,-50,1,-1,-2],dtype=torch.double))
        #         # assert (np.round(v.detach().numpy(),3) == np.round(gt_V[int(x[traj])][int(y[traj])],3))
        #         traj_vs.append(v)
        #     # print (traj_vs)
        #     v_pi_approx.append(torch.tensor(traj_vs))
        # return torch.stack(v_pi_approx)

    def get_vals(self,cords):

        with torch.no_grad():
            selected_succ_feats =[self.succ_feats[:,x,y].double() for x,y in cords.long()]
            selected_succ_feats = torch.stack(selected_succ_feats)
            vs = self.linear1(selected_succ_feats)
            # return torch.squeeze(vs)
            v_pi_approx = torch.sum(torch.mul(self.softmax(vs),vs),dim = 1)
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
        #[ssx1, ssx2] u [ssy1, ssy2] => [[ssx1, ssy1], [ssx2, ssy2]]

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

        #
        # v_ss = self.get_vals_old(ss_x, ss_y)
        # v_es = self.get_vals_old(es_x, es_y)
        # #
        # print(v_es_)
        # print(v_es)
        # print ("\n")

        # print (v_es[0], v_es_old[0])

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

        left_pred = torch.sigmoid(torch.subtract(left_delta_er, right_delta_er))
        right_pred = torch.sigmoid(torch.subtract(right_delta_er, left_delta_er))

        phi_logit = torch.stack([left_pred,right_pred],axis=1)
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

    if include_eps and abs(arr[0] - arr[1]) < 0.1:
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

def model_eval(X,Y,w,model,is_testing=False):
    # w = np.array([ 2.2874,  0.4489,  0.0879,  0.0262, -0.0747,  0.9900]).T
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

        r_gt = np.dot(x[:,0:6], np.array([-1,50,-50,1,-1,-2]).T)

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

            r1_er_gt, r2_er_gt = get_gt_er(x)
            # y_gt = [sigmoid(r1_er_gt-r2_er_gt), sigmoid(r2_er_gt-r1_er_gt)]
            y_gt = [r1_er_gt, r2_er_gt]

        res=get_pref(y_hat)

        # print (y_gt)
        # print (x2er.get(tuple(x)))

        y_gt_pref = get_pref(y_gt)
        # print (y_f)
        # print (y_gt)
        # print ("\n")
        assert y_f[0] == y_gt_pref[0] and y_f[1] == y_gt_pref[1]
        total +=1

        if res[0] == y_f[0] == y_gt_pref[0] and res[1] == y_f[1] == y_gt_pref[1]:
            n_correct += 1
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


def train(aX, ay, saX = None, say = None, loss_coef = None, plot_loss=True):
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = train_test_split(aX, ay,test_size=.2,random_state= 0,shuffle=True)

    # X_train = list(X_train)
    # X_train.append([[2, 0, 1, 0, 0, 0, 2, 1, 2, 2], [2, 0, 0, 0, 0, 1, 3, 7, 1, 6]])
    # X_train = np.array(X_train)
    #
    # y_train = list(y_train)
    # y_train.append([0, 1])
    # y_train = np.array(y_train)
    #
    # print (len(X_train))
    # X_train = np.array(aX)
    # y_train = np.array(ay)
    # print (len(X_train))


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
        model = RewardFunctionPR()
    elif prefrence_assum == "er":
        model = RewardFunctionER(succ_feats)

    # testing_acc = model_eval(X_train,y_train,None,model,is_testing=True)
    # assert False
    # model_eval(X_train,y_train,np.array([-1,50,-50,1,-1,-2]).T,model,is_testing=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(N_ITERS):
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
        # print (loss)
        losses.append(loss.detach())
        # Backward pass
        loss.backward()
        optimizer.step()

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

    train_loss = reward_pred_loss(model(X_train), y_train).detach().numpy()
    test_loss = reward_pred_loss(model(X_test), y_test).detach().numpy()

    train_batch_size = y_train.size()[0]
    test_batch_size = y_test.size()[0]

    if type(saX) == np.ndarray and "user" in mode:
        train_loss += mixed_synth_reward_pred_loss(model(sX_train), sy_train, loss_coef).detach().numpy()
        test_loss += mixed_synth_reward_pred_loss(model(sX_test), sy_test, loss_coef).detach().numpy()

        train_loss /= (train_batch_size + loss_coef*sy_train.size()[0])
        test_loss /= (test_batch_size + loss_coef*sy_test.size()[0])
    else:
        train_loss /= (train_batch_size)
        test_loss /= (test_batch_size)

    print ("training loss: " + str(train_loss))
    print ("testing loss: " + str(test_loss))

    generate_results_table(X_test,y_test,model)

    for param in model.parameters():
        reward_vector = param.detach().numpy()[0]

    testing_acc = model_eval(X_train,y_train,reward_vector.T,model,is_testing=True)
    training_acc = model_eval(X_test,y_test,reward_vector.T,model,is_testing=True)
    if type(saX) == np.ndarray and "user" in mode:
        testing_acc += model_eval(sX_train,sy_train,reward_vector.T,model,is_testing=True)
        testing_acc /=2

        training_acc += model_eval(sX_test,sy_test,reward_vector.T,model,is_testing=True)
        training_acc /=2

    print ("Training accuracy: " + str(training_acc))
    print ("Testing accuracy: " + str(testing_acc))
    print (reward_vector)

    f = open("BACKUP_DEBUG_OUT.txt", "w")
    f.write("Testing accuracy: " + str(testing_acc) + "\n")
    f.write("Training accuracy: " + str(training_acc) + "\n")
    f.write(str(reward_vector) + "\n")
    f.close()
    print ("=================================================================================\n")
    return reward_vector,losses,train_loss, test_loss, training_acc, testing_acc


vf_X, vf_r, vf_y, vf_ses, pr_X, pr_r, pr_y, pr_ses, none_X, none_r, none_y, none_ses = get_all_statistics(include_dif_traj_lengths=include_dif_traj_lengths)

#
# n_same = 0
# total = 0
#
# pr_X, pr_y, _, _ = clean_y(pr_X,pr_r,pr_y)
# none_X, none_y, _, _ = clean_y(none_X,none_r,none_y)
#
# for prx,pry in zip(pr_X,pr_y):
#     for nonex,noney in zip(none_X,none_y):
#         if np.array_equal(prx, nonex) and np.array_equal(pry, noney):
#             # print ((prx,nonex))
#             # print ((pry, noney))
#             # print ("\n")
#             n_same +=1
#             break
#         # else:
#         #     print ((prx,nonex))
#         #     print ((pry, noney))
#         #     print ("\n")
#
#     total +=1
#
# print (n_same/total)
# print (n_same)
# print (len(pr_X))
# print (len(none_X))
#
#
#
# assert False

# pr_X.extend(vf_X)
# pr_r.extend(vf_r)
# pr_X.extend(none_X)
# pr_X.extend(none_r)

# pr_X.append([[0,0,0,0,3,3],[2,0,1,2,0,0]])
# synth_max_y.append([1,0])
#
# pr_X.append([[3,0,0,3,0,0],[0,1,0,0,2,2]])
# synth_max_y.append([0,1])
all_reward_vecs = []
all_train_losses = []
all_test_losses = []
all_training_acc= []
all_testing_acc = []
all_total_training_losses = []
all_avg_returns = []

if mode == "deterministic_user_data":
    X_copy = none_X.copy()
    r_copy = none_r.copy()
    y_copy = none_y.copy()

    X_copy, y_copy, X_copy_sytnh, y_copy_synth,_,_ = clean_y(X_copy, r_copy,y_copy)


    # aX, ay = augment_data(pr_X,pr_y,"scalar")
    aX, ay = augment_data(X_copy,y_copy,"arr")
    aX_synth, ay_synth = augment_data(X_copy_sytnh,y_copy_synth,"arr")


    print ("finding reward vector...")
    rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay, aX_synth, ay_synth, plot_loss=False)

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
    # X_user = none_X.copy()
    # r_user = none_r.copy()
    # y_user = none_y.copy()

    X_user = pr_X.copy()
    r_user = pr_r.copy()
    y_user = pr_y.copy()

    # X_user = vf_X.copy()
    # r_user = vf_r.copy()
    # y_user = vf_y.copy()

    # X_copy, y_copy, X_copy_sytnh, y_copy_synth = clean_y(X_copy,r_copy,y_copy)
    dfs = []
    for prob_iter in range(n_prob_iters):
        print ("==========================Trial " + str(prob_iter)+" ==========================")

        # X_copy = none_X.copy()
        # r_copy = none_r.copy()
        # y_copy = none_y.copy()

        X_copy, y_copy, X_copy_sytnh, y_copy_synth,y_dist_copy,loss_coef = clean_y(X_user,r_user,y_user)

        df = save_synth_data_table(X_copy_sytnh,y_copy_synth,y_dist_copy)
        dfs.append(df)
        # continue
        print (len(X_copy) + len(X_copy_sytnh))

        # aX, ay = augment_data(X_copy,y_copy,"arr")
        # aX_synth, ay_synth = augment_data(X_copy_sytnh,y_copy_synth,"arr")
        #
        aX = X_copy
        ay = y_copy

        # X_copy_sytnh.extend(aX)
        # y_copy_synth.extend(ay)
        aX_synth = np.array(X_copy_sytnh)
        ay_synth = np.array(y_copy_synth)

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
    pr_X.extend(vf_X)
    pr_r.extend(vf_r)
    pr_ses.extend(vf_ses)

    pr_X.extend(none_X)
    pr_r.extend(none_r)
    pr_ses.extend(none_ses)

    pr_X_copy = pr_X.copy()
    pr_r_copy = pr_r.copy()

    for prob_iter in range(n_prob_iters):
        pr_X,synth_max_y,_ = generate_synthetic_prefs(pr_X_copy,pr_r_copy,pr_ses,mode)
        print (len(pr_X))
        # aX, ay = augment_data(pr_X,pr_y,"scalar")
        aX, ay = augment_data(pr_X,synth_max_y,"arr")


        print ("==========================Trial " + str(prob_iter)+" ==========================")
        print ("finding reward vector...")
        rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay,plot_loss=False)
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

    print ("\n\n")
    print ("Across all " + str(n_prob_iters) + " trials,")

    disp_mmv(all_testing_acc, "Testing Accuracy",None)
    disp_mmv(all_training_acc, "Training Accuracy",None)
    disp_mmv(all_avg_returns, "Average Return",None)
    disp_mmv(all_train_losses, "Training Loss",None)
    disp_mmv(all_test_losses, "Testing Loss",None)
    disp_mmv(all_reward_vecs, "Reward Vector",0)

    plt.plot(np.array(all_total_training_losses).T)
    plt.show()
else:
    pr_X.extend(vf_X)
    pr_r.extend(vf_r)
    pr_ses.extend(vf_ses)

    pr_X.extend(none_X)
    pr_r.extend(none_r)
    pr_ses.extend(none_ses)

    pr_X,synth_max_y,expected_returns = generate_synthetic_prefs(pr_X,pr_r,pr_ses,mode)

    # save_er_synth_data_table(pr_X,synth_max_y,expected_returns)

    # aX, ay = augment_data(pr_X,pr_y,"scalar")
    aX, ay = augment_data(pr_X,synth_max_y,"arr")


    print ("finding reward vector...")
    rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay,plot_loss=True)

    print ("performing value iteration...")
    V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

    # print ("following policy...")
    # follow_policy(Q,100,viz_policy=True)
    print ("following policy...")
    pi = build_pi(Q)
    V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
    avg_return = np.sum(V_under_gt)/92
    print ("average return following learned policy: ")
    print (avg_return)

    get_gt_avg_return(GAMMA=GAMMA)
