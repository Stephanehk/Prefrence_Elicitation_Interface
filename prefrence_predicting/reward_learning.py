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

from grid_world import GridWorldEnv
from load_training_data import get_all_statistics,find_end_state
from value_iteration import value_iteration, follow_policy, learn_successor_feature,get_gt_avg_return,build_pi,iterative_policy_evaluation
from generate_random_policies import generate_all_policies, calc_value

keep_ties = True
n_prob_samples = 10
n_prob_iters = 30
GAMMA=0.999
include_dif_traj_lengths = True
# mode = "deterministic_user_data"
# mode = "user_data"
# mode = "sigmoid"
mode = "deterministic"

prefrence_assum = "er"
if prefrence_assum == "er":
    print("generating policies...")
    succ_feats, pis = generate_all_policies(100,GAMMA)
    print("finished")

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

def get_er (x,w = [-1,50,-50,1,-1,-2]):

    # x[0] = list(x[0])
    # x[0].extend([ses[0][0][0],ses[0][0][1], ses[0][1][0],ses[0][1][1]])
    #
    # x[1] = list(x[1])
    # x[1].extend([ses[1][0][0],ses[1][0][1], ses[1][1][0],ses[1][1][1]])
    # x_f = [x[0],x[1]]

    t1_ss = [x[0][6], x[0][7]]
    t1_es = [x[0][8], x[0][9]]

    t2_ss = [x[1][6], x[1][7]]
    t2_es = [x[1][8], x[1][9]]

    # t1_ss = ses[0][0]
    # t1_es = ses[0][1]
    #
    # t2_ss = ses[1][0]
    # t2_es = ses[1][1]

    r = np.dot(x,w)

    r1_er = r[0] + calc_value(w,t1_es,succ_feats) - calc_value(w,t1_ss,succ_feats)
    r2_er = r[1] + calc_value(w,t2_es,succ_feats) - calc_value(w,t2_ss,succ_feats)

    return r1_er, r2_er

def clean_y(X,R,Y):
    formatted_y = []
    out_X = []

    synth_formatted_y = []
    synth_out_X = []
    for x,r,y in zip(X,R,Y):
        x = [list(x[0]),list(x[1])]
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
                        r1_er, r2_er = get_er (x)
                        r1_prob = sigmoid(r1_er-r2_er)
                        r2_prob = sigmoid(r2_er-r1_er)
                    # if abs(r1_prob - r2_prob) < 0.1:
                    #     continue

                    num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                    if num == 1:
                        pref = [1,0]
                    elif num == 0:
                        pref = [0,1]
                    synth_formatted_y.append(np.array(pref))
                    synth_out_X.append(x)
            else:
                synth_formatted_y.append(np.array(get_pref(r,False)))
                synth_out_X.append(x)
        else:
            formatted_y.append(np.array(y))
            out_X.append(x)
    return out_X,formatted_y,synth_out_X,synth_formatted_y

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

def generate_synthetic_prefs(pr_X,rewards,sess,mode):
    synth_y = []
    non_redundent_pr_X = []
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
        t1_n_trans = x[0][0] + x[0][1] + x[0][2] + x[0][5]
        t2_n_trans = x[1][0] + x[1][1] + x[1][2] + x[1][5]
        # r[0] *=(np.power(GAMMA,t1_n_trans))
        # r[1] *=(np.power(GAMMA,t2_n_trans))
        #remove duplicates
        if is_subset(x_f,non_redundent_pr_X):
            continue

        if mode == "sigmoid":
            if not keep_ties and r[1] == r[0]:
                continue

            for n_samp in range(n_prob_samples):
                if prefrence_assum == "pr":
                    r1_prob = sigmoid(r[0]-r[1])
                    r2_prob = sigmoid(r[1]-r[0])
                elif prefrence_assum == "er":
                    r1_er, r2_er = get_er(x)
                    r1_prob = sigmoid(r1_er-r2_er)
                    r2_prob = sigmoid(r2_er-r1_er)

                # if abs(r1_prob - r2_prob) < 0.1:
                #     continue
                num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                if num == 1:
                    pref = [1,0]
                elif num == 0:
                    pref = [0,1]
                synth_y.append(pref)
                non_redundent_pr_X.append(x_f)
        else:
            pref = get_pref(r,False)
            if pref == [0.5,0.5] and not keep_ties:
                continue
            synth_y.append(pref)
            non_redundent_pr_X.append(x_f)
    return non_redundent_pr_X, synth_y

def reward_pred_loss(output, target):
    batch_size = output.size()[0]
    output = torch.squeeze(output)
    output = torch.log(output)
    res = torch.mul(output,target)
    return -torch.sum(res)/batch_size

def mixed_synth_reward_pred_loss(output, target,x_length):
    batch_size = output.size()[0]
    output = torch.squeeze(output)
    output = torch.log(output)
    res = torch.mul(output,target)

    # print (len(output)/x_length)
    # print (torch.sum(res)/batch_size)

    return -torch.mul(torch.sum(res)/batch_size, x_length/len(output))

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
        self.succ_feats = torch.tensor(succ_feats,dtype=torch.double)
        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False).double()

    # def calc_value(self, state):
    #     x,y = state
    #     return torch.max(torch.tensor([torch.squeeze(self.linear1(succ_feats[i][x][y])) for i in range(len(self.succ_feats))]))
    def get_vals(self,xs,ys):
        v_pi_approx = []

        for x,y in zip(xs,ys):
            traj_vs = []
            for traj in range (len(x)):
                ss_vs = []
                for i in range(len(self.succ_feats)):
                    # print (int(x[traj]))
                    # print (int(y[traj]))
                    succ_phi = self.succ_feats[i][int(x[traj])][int(y[traj])]
                    succ_phi = succ_phi.double()
                    v = self.linear1(succ_phi)
                    ss_vs.append(v)
                traj_vs.append(torch.max(torch.tensor(ss_vs)))
            v_pi_approx.append(torch.tensor(traj_vs))
        v_pi_approx = torch.stack(v_pi_approx)
        return v_pi_approx

    def forward(self, phi):

        pr = torch.squeeze(self.linear1(phi[:,:,0:6].double()))
        ss_x = torch.squeeze(phi[:,:,6:7])
        ss_x = torch.squeeze(phi[:,:,7:8])

        es_x = torch.squeeze(phi[:,:,8:9])
        es_y = torch.squeeze(phi[:,:,9:10])

        # print (phi[:,:,6:7])
        #
        # print (ss_x)

        #build list of succ fears for start states
        v_ss = self.get_vals(ss_x,ss_x)
        # v_es = self.get_vals(es_x,es_x)

        left_pr = pr[:,0:1]
        right_pr = pr[:,1:2]

        left_pred = torch.sigmoid(torch.subtract(left_pr, right_pr))
        right_pred = torch.sigmoid(torch.subtract(right_pr, left_pr))

        phi_logit = torch.stack([left_pred,right_pred],axis=1)
        return phi_logit

def get_pref(arr,include_eps = True):
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

def generate_results_table(X,Y,model):
    logits = torch.squeeze(model(X))
    logits_str = stringify(logits)
    prs_str = stringify(np.dot(X, np.array([-1,50,-50,1,-1,-2]).T))
    prefs_str = stringify(np.array([get_pref(logit) for logit in logits]))
    X_str = stringify(X)
    Y_str = stringify(Y)

    df = pd.DataFrame(np.array([X_str, prs_str, logits_str, Y_str, prefs_str]).T,columns=["[psi_1, psi2]", "ground truth partial returns", "outputted logits", "ground truth prefrence", "predicted prefrence"])

    df.to_csv("2021_21_9_synth_reward_pred_debug.csv", sep='\t')

def model_eval(X,Y,w,is_testing=False):
    # w = np.array([ 2.2874,  0.4489,  0.0879,  0.0262, -0.0747,  0.9900]).T
    n_correct = 0
    total= 0

    for (x,y) in zip(X,Y):
        if not keep_ties and y[0] == 0.5:
            continue

        r = np.dot(x,w)

        r_gt = np.dot(x, np.array([-1,50,-50,1,-1,-2]).T)

        t1_n_trans = x[0][0] + x[0][1] + x[0][2] + x[0][5]
        t2_n_trans = x[1][0] + x[1][1] + x[1][2] + x[1][5]
        # r[0] *=(np.power(GAMMA,t1_n_trans))
        # r_gt[0] *=(np.power(GAMMA,t1_n_trans))
        #
        # r[1] *=(np.power(GAMMA,t2_n_trans))
        # r_gt[1] *=(np.power(GAMMA,t2_n_trans))
        if prefrence_assum == "pr":
            y_hat = [sigmoid(r[0]-r[1]),sigmoid(r[1]-r[0])]
            y_gt = [sigmoid(r_gt[0]-r_gt[1]),sigmoid(r_gt[1]-r_gt[0])]
        elif prefrence_assum == "er":
            r1_er, r2_er = get_er(x,w=w)
            y_hat = [sigmoid(r1_er-r2_er), sigmoid(r2_er-r1_er)]

            r1_er_gt, r2_er_gt = get_er(x)
            y_gt = [sigmoid(r1_er_gt-r2_er_gt), sigmoid(r2_er_gt-r1_er_gt)]

        res=get_pref(y_hat)

        y_f = get_pref(y)

        y_gt = get_pref(y_gt)


        total +=1
        if res[0] == y_f[0] == y_gt[0] and res[1] == y_f[1] == y_gt[1]:
            n_correct += 1

    return (n_correct/total)

def disp_mmv(arr,title,axis):
    print ("Mean " + title + ": " + str(np.mean(arr,axis=axis)))
    print ("Median " + title + ": " + str(np.median(arr,axis=axis)))
    print (title + " Variance: " + str(np.var(arr,axis=axis)))


def train(aX, ay, saX = None, say = None, plot_loss=True):
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = train_test_split(aX, ay,test_size=.2,random_state= 0,shuffle=True)
    #
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

    optimizer = torch.optim.SGD(model.parameters(), lr=2)#crank up lr
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(30000):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train)
        y_pred = torch.clamp(y_pred,min=1e-35,max=None)#prevents prob pred of 0

        # Compute Loss
        loss = reward_pred_loss(y_pred, y_train)
        print (loss)

        if type(saX) == np.ndarray and "user" in mode:
            sy_pred = model(sX_train)
            sy_pred = torch.clamp(sy_pred,min=1e-35,max=None)#prevents prob pred of 0

            sloss = mixed_synth_reward_pred_loss(sy_pred, sy_train,len(X_train))
            loss += sloss

        # if torch.isnan(loss):

            # for param in model.parameters():
            #     reward_vector = param.detach().numpy()[0]
            #     print (reward_vector)
            # print (sy_train)
            # print (sy_pred)
            # print (torch.log(torch.squeeze(sy_pred)))
            # assert False

        losses.append(loss.detach())
        # Backward pass
        loss.backward()
        optimizer.step()

        # tracks accuracy during training (this significantly slows things down)
        # for param in model.parameters():
        #     train_accuracy = model_eval(X_train,y_train,param.detach().numpy()[0].T)
        #     test_accuracy = model_eval(X_test,y_test,param.detach().numpy()[0].T)
        #
        #     train_accuracies.append(1-train_accuracy)
        #
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

    if type(saX) == np.ndarray and "user" in mode:
        train_loss += mixed_synth_reward_pred_loss(model(sX_train), sy_train,len(X_train)).detach().numpy()
        test_loss += mixed_synth_reward_pred_loss(model(sX_test), sy_test,len(X_test)).detach().numpy()
    print ("training loss: " + str(train_loss))
    print ("testing loss: " + str(test_loss))

    # generate_results_table(X_test,y_test,model)

    for param in model.parameters():
        reward_vector = param.detach().numpy()[0]

    testing_acc = model_eval(X_train,y_train,reward_vector.T,is_testing=True)
    training_acc = model_eval(X_test,y_test,reward_vector.T,is_testing=True)
    if type(saX) == np.ndarray and "user" in mode:
        testing_acc += model_eval(sX_train,sy_train,reward_vector.T,is_testing=True)
        testing_acc /=2

        training_acc += model_eval(sX_test,sy_test,reward_vector.T,is_testing=True)
        training_acc /=2

    print ("Training accuracy: " + str(testing_acc))
    print ("Testing accuracy: " + str(training_acc))
    print (reward_vector)
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

    X_copy, y_copy, X_copy_sytnh, y_copy_synth = clean_y(X_copy, r_copy,y_copy)


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

    # X_user = pr_X.copy()
    # r_user = pr_r.copy()
    # y_user = pr_y.copy()

    X_user = vf_X.copy()
    r_user = vf_r.copy()
    y_user = vf_y.copy()

    # X_copy, y_copy, X_copy_sytnh, y_copy_synth = clean_y(X_copy,r_copy,y_copy)

    for prob_iter in range(n_prob_iters):
        # X_copy = none_X.copy()
        # r_copy = none_r.copy()
        # y_copy = none_y.copy()

        X_copy, y_copy, X_copy_sytnh, y_copy_synth = clean_y(X_user,r_user,y_user)

        print (len(X_copy) + len(X_copy_sytnh))
        # aX, ay = augment_data(pr_X,pr_y,"scalar")
        aX, ay = augment_data(X_copy,y_copy,"arr")
        aX_synth, ay_synth = augment_data(X_copy_sytnh,y_copy_synth,"arr")

        print ("==========================Trial " + str(prob_iter)+" ==========================")
        print ("finding reward vector...")
        rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX_synth, ay_synth, None, ay_synth, plot_loss=False)


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

    print (all_avg_returns)
    disp_mmv(all_avg_returns, "Average Return",None)
    disp_mmv(all_train_losses, "Training Loss",None)
    disp_mmv(all_test_losses, "Testing Loss",None)
    disp_mmv(all_reward_vecs, "Reward Vector",0)

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
        pr_X,synth_max_y = generate_synthetic_prefs(pr_X_copy,pr_r_copy,pr_ses,mode)
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

    pr_X,synth_max_y = generate_synthetic_prefs(pr_X,pr_r,pr_ses,mode)

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
