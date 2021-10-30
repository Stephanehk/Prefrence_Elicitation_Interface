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
from load_training_data import get_all_statistics
from value_iteration import value_iteration, follow_policy, learn_successor_feature,get_gt_avg_return,build_pi,iterative_policy_evaluation

keep_ties = True
n_prob_samples = 100
n_prob_iters = 30
GAMMA=0.999
include_dif_traj_lengths = True
# mode = "deterministic_user_data"
mode = "user_data"
# mode = "sigmoid"
# mode = "deterministic"

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

def clean_y(X,R,Y):
    formatted_y = []
    out_X = []
    for x,r,y in zip(X,R,Y):
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
            if not "deterministic" in mode:
                for n_samp in range(n_prob_samples):
                    np.random.seed(n_samp)
                    r1_prob = sigmoid(r[0]-r[1])
                    r2_prob = sigmoid(r[1]-r[0])
                    # if abs(r1_prob - r2_prob) < 0.1:
                    #     continue
                    num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                    if num == 1:
                        pref = [1,0]
                    elif num == 0:
                        pref = [0,1]
                    formatted_y.append(np.array(pref))
                    out_X.append(x)
            else:
                if r[0] > r[1]:
                    pref = [1,0]
                elif r[1] > r[0]:
                    pref = [0,1]
                elif r[1] == r[0]:
                    pref = [0.5, 0.5]
                formatted_y.append(np.array(pref))
                out_X.append(x)
    return out_X,formatted_y



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


def generate_synthetic_prefs(pr_X,rewards,mode):
    synth_y = []
    non_redundent_pr_X = []
    for r,x in zip(rewards,pr_X):

        x_f = [list(x[0]),list(x[1])]

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
                r1_prob = sigmoid(r[0]-r[1])
                r2_prob = sigmoid(r[1]-r[0])
                # if abs(r1_prob - r2_prob) < 0.1:
                #     continue
                num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                if num == 1:
                    pref = [1,0]
                elif num == 0:
                    pref = [0,1]
                synth_y.append(pref)
                non_redundent_pr_X.append(x_f)
        # elif mode == "max":
        else:
            if r[0] > r[1]:
                pref = [1,0]
            elif r[1] > r[0]:
                pref = [0,1]
            elif r[1] == r[0]:
                if not keep_ties:
                    continue
                pref = [0.5,0.5]

            synth_y.append(pref)
            non_redundent_pr_X.append(x_f)
    return non_redundent_pr_X, synth_y


def validate_synth_data(X,y):
    for i in range(len(X)):
        r1 = np.dot(X[i][0], [-1,50,-50,1,-1,-2])
        r2 = np.dot(X[i][1], [-1,50,-50,1,-1,-2])

        X[i][0][1] = 0
        X[i][0][2] = 0
        X[i][1][1] = 0
        X[i][1][2] = 0
        r1_no_term = np.dot(X[i][0], [-1,50,-50,1,-1,-2])
        r2_no_term = np.dot(X[i][1], [-1,50,-50,1,-1,-2])



        pref_prob = [sigmoid(r1-r2),sigmoid(r2-r1)]
        pref_prob_no_term = [sigmoid(r1_no_term-r2_no_term),sigmoid(r2_no_term-r1_no_term)]
        if pref_prob[0] > pref_prob[1]:
            recovered_pref = [1,0]
        elif pref_prob[1] > pref_prob[0]:
            recovered_pref = [0,1]
        else:
            recovered_pref = [0.5,0.5]

        if pref_prob_no_term[0] > pref_prob_no_term[1]:
            recovered_pref_no_term = [1,0]
        elif pref_prob_no_term[1] > pref_prob_no_term[0]:
            recovered_pref_no_term = [0,1]
        else:
            recovered_pref_no_term = [0.5,0.5]

        assert (recovered_pref[0] == y[i][0] == recovered_pref_no_term[0] and recovered_pref[1] == y[i][1] == recovered_pref_no_term[1])

def reward_pred_loss(output, target):
    batch_size = output.size()[0]
    output = torch.squeeze(output)
    output = torch.log(output)
    res = torch.mul(output,target)
    return -torch.sum(res)/batch_size

class RewardFunction(torch.nn.Module):
    def __init__(self,n_features=6):
        super(RewardFunction, self).__init__()
        self.n_features = n_features
        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False)

    def forward(self, phi):
        out = torch.squeeze(self.linear1(phi))
        left = out[:,0:1]
        right = out[:,1:2]
        left_pred = torch.sigmoid(torch.subtract(out[:,0:1],out[:,1:2]))
        right_pred = torch.sigmoid(torch.subtract(out[:,1:2],out[:,0:1]))
        phi_logit = torch.stack([left_pred,right_pred],axis=1)
        # phi_logit = torch.sigmoid(self.linear1(phi)) #problem here, should not apply sig function to individual
        return phi_logit

def get_pref(arr):
    if abs(arr[0] - arr[1]) < 0.1:
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
    # for (x,y) in zip(X_test,y_test):
    #     # if y == 0.5:
    #     #     continue
    #     total +=1
    #     y_hat = np.matmul(x,w)
    #     # print (y_hat)
    #     if (y_hat[0] > y_hat[1]):
    #         res = [1,0]
    #     elif (y_hat[1] > y_hat[0]):
    #         res = [0,1]
    #     else:
    #         res = [0.5,0.5]
    #     if res[0] == y[0] and res[1] == y[1]:
    #         n_correct += 1
    # print (n_correct/total)
    # #
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

        y_hat = [sigmoid(r[0]-r[1]),sigmoid(r[1]-r[0])]
        y_gt = [sigmoid(r_gt[0]-r_gt[1]),sigmoid(r_gt[1]-r_gt[0])]
        # print (r)
        # print (y)
        # print ("\n")
        res=get_pref(y_hat)

        y_f = get_pref(y)

        y_gt = get_pref(y_gt)


        total +=1
        if res[0] == y_f[0] == y_gt[0] and res[1] == y_f[1] == y_gt[1]:
            n_correct += 1
        # if is_testing:
        #     print (y_hat)
        #     print (res)
        #     print (y_f)
        #     print (y_gt)
        #     print ("\n")
    return (n_correct/total)

def disp_mmv(arr,title,axis):
    print ("Mean " + title + ": " + str(np.mean(arr,axis=axis)))
    print ("Median " + title + ": " + str(np.median(arr,axis=axis)))
    print (title + " Variance: " + str(np.var(arr,axis=axis)))


def train(aX, ay,plot_loss=True):
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = train_test_split(aX, ay,test_size=.2,random_state= 0,shuffle=True)
    #
    X_train = format_X(X_train)
    y_train = format_y(y_train,"arr")

    X_test = format_X(X_test)
    y_test = format_y(y_test,"arr")

    model = RewardFunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=2)#crank up lr
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(30000):

        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train)
        # Compute Loss
        # loss = F.cross_entropy(y_pred, y_train)
        loss = reward_pred_loss(y_pred, y_train)
        # print (loss)
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
    print ("training loss: " + str(train_loss))
    print ("testing loss: " + str(test_loss))

    # generate_results_table(X_test,y_test,model)

    for param in model.parameters():
        reward_vector = param.detach().numpy()[0]

    testing_acc = model_eval(X_train,y_train,reward_vector.T,is_testing=True)
    training_acc = model_eval(X_test,y_test,reward_vector.T,is_testing=True)
    print ("Training accuracy: " + str(testing_acc))
    print ("Testing accuracy: " + str(training_acc))
    print (reward_vector)
    print ("=================================================================================\n")
    return reward_vector,losses,train_loss, test_loss, training_acc, testing_acc



vf_X, vf_r, vf_y, pr_X, pr_r, pr_y, none_X, none_r, none_y = get_all_statistics(include_dif_traj_lengths=include_dif_traj_lengths)

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

    _, y_copy = clean_y(r_copy,y_copy)

    print (len(pr_X))
    # aX, ay = augment_data(pr_X,pr_y,"scalar")
    aX, ay = augment_data(X_copy,y_copy,"arr")

    print ("finding reward vector...")
    rew_vect,all_losses,train_loss,test_loss,training_acc, testing_acc = train(aX, ay,plot_loss=False)

    print ("performing value iteration...")
    V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

    print ("following policy...")
    # avg_return = follow_policy(Q,100,viz_policy=False)
    pi = build_pi(Q)
    V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
    avg_return = np.sum(V_under_gt)/(len(V_under_gt)*len(V_under_gt[0]))
    print ("average return following learned policy: ")
    print (avg_return)

elif mode == "user_data":
    # X_copy = none_X.copy()
    # r_copy = none_r.copy()
    # y_copy = none_y.copy()

    X_copy = pr_X.copy()
    r_copy = pr_r.copy()
    y_copy = pr_y.copy()

    X_copy, y_copy = clean_y(X_copy,r_copy,y_copy)

    for prob_iter in range(n_prob_iters):
        print (len(X_copy))
        # aX, ay = augment_data(pr_X,pr_y,"scalar")
        aX, ay = augment_data(X_copy,y_copy,"arr")


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
        avg_return = np.sum(V_under_gt)/(len(V_under_gt)*len(V_under_gt[0]))
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


elif mode == "sigmoid":
    pr_X.extend(vf_X)
    pr_r.extend(vf_r)
    pr_X.extend(none_X)
    pr_X.extend(none_r)

    pr_X_copy = pr_X.copy()
    pr_r_copy = pr_r.copy()

    for prob_iter in range(n_prob_iters):
        pr_X,synth_max_y = generate_synthetic_prefs(pr_X_copy,pr_r_copy,mode)
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
        avg_return = np.sum(V_under_gt)/(len(V_under_gt)*len(V_under_gt[0]))
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
    pr_X.extend(none_X)
    pr_X.extend(none_r)

    pr_X,synth_max_y = generate_synthetic_prefs(pr_X,pr_r,mode)
    print (len(pr_X))
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
    avg_return = np.sum(V_under_gt)/(len(V_under_gt)*len(V_under_gt[0]))
    print ("average return following learned policy: ")
    print (avg_return)

    get_gt_avg_return(GAMMA=GAMMA)
