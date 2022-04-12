import pickle
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
from scipy import stats
from mord import LogisticAT
from sklearn import metrics
from sklearn.model_selection import train_test_split
import json
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn import preprocessing
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import random
import math

class OrdinalClassifier():

    def __init__(self,random_state,fit_intercept=True):
        self.random_state = random_state
        self.coef_ = None
        self.left_regressor = LogisticRegression(random_state=self.random_state,fit_intercept=fit_intercept)
        self.right_regressor = LogisticRegression(random_state=self.random_state,fit_intercept=fit_intercept)

    def fit(self,X,y):
        left_partitioned_y = []
        right_partitioned_y = []
        for var in y:
            if var > 0:
                left_partitioned_y.append(1)
            else:
                left_partitioned_y.append(0)

            if var == 1:
                right_partitioned_y.append(1)
            else:
                right_partitioned_y.append(0)


        self.left_regressor.fit(X,left_partitioned_y)
        self.right_regressor.fit(X,right_partitioned_y)
        self.coef_ = [self.left_regressor.coef_, self.right_regressor.coef_]

    def get_params(self):
        #Note: assumes that left_regressor and right_regressor have the same parameters
        return self.left_regressor.get_params()

    def set_coefs(self,b1,b2,b3):
        self.left_regressor.coef_ = np.array([[b1,b2,b3]])
        self.right_regressor.coef_ = np.array([[b1,b2,b3]])

    def learn_coefs(self,X,y):
        self.fit(X,y) #just to get all our class variables

    def predict(self,X):
        y_pred = []
        left_partion = self.left_regressor.predict_proba(X)
        right_partion = self.right_regressor.predict_proba(X)
        for left_p, right_p in zip(left_partion,right_partion):
            left_prob = 1-left_p[1]
            right_prob = right_p[1]
            same_prob = left_p[1] - right_prob

            if left_prob >= right_prob and left_prob >= same_prob:
                pred = 0
            elif right_prob >= left_prob and right_prob >= same_prob:
                pred = 1
            elif same_prob >= left_prob and same_prob >= right_prob:
                pred = 0.5
            pred_probs = [left_prob,same_prob,right_prob]
            y_pred.append(pred_probs)
        return y_pred

with open('/Users/stephanehatgiskessell/Desktop/Kivy_stuff/MTURK_interface/2021_08_collected_data/2021_08_18_woi_questions.data', 'rb') as f:
    questions = pickle.load(f)
with open('/Users/stephanehatgiskessell/Desktop/Kivy_stuff/MTURK_interface/2021_08_collected_data/2021_08_18_woi_answers.data', 'rb') as f:
    answers = pickle.load(f)
# #
# with open('../2021_12_28_woi_questions.data', 'rb') as f:
#     questions_aug = pickle.load(f)
# with open('../2021_12_28_woi_answers.data', 'rb') as f:
#     answers_aug = pickle.load(f)
# # # #
with open('../2021_1_10_no_pen_0_questions.data', 'rb') as f:
    questions_aug = pickle.load(f)
with open('../2021_1_10_no_pen_0_answers.data', 'rb') as f:
    answers_aug = pickle.load(f)

multi_len_data = "../saved_data/augmented_trajs_multilength.json"
t_nt_1_data = "../saved_data/augmented_trajs_t_nt_quad1.json"
t_nt_2_data = "../saved_data/augmented_trajs_t_nt_quad2.json"

dsdt_data = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/saved_data/2021_07_29_dsdt_chosen.json"
dsst_data = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/saved_data/2021_07_29_dsst_chosen.json"
ssst_data = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/saved_data/2021_07_29_ssst_chosen.json"
sss_data = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/saved_data/2021_07_29_sss_chosen.json"
board = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/boards/2021-07-29_sparseboard2-notrap_board.json"
board_vf = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/boards/2021-07-29_sparseboard2-notrap_value_function.json"
board_rf = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/boards/2021-07-29_sparseboard2-notrap_rewards_function.json"



with open(board_vf, 'r') as j:
    board_vf = json.loads(j.read())

with open(dsdt_data, 'r') as j:
    dsdt_data = json.loads(j.read())
with open(dsst_data, 'r') as j:
    dsst_data = json.loads(j.read())
with open(ssst_data, 'r') as j:
    ssst_data = json.loads(j.read())
with open(sss_data, 'r') as j:
    sss_data = json.loads(j.read())

with open(multi_len_data, 'r') as j:
    multi_len_data = json.loads(j.read())
with open(t_nt_1_data, 'r') as j:
    t_nt_1_data = json.loads(j.read())
with open(t_nt_2_data, 'r') as j:
    t_nt_2_data = json.loads(j.read())

with open(board, 'r') as j:
    board = json.loads(j.read())
with open(board_rf, 'r') as j:
    board_rf = json.loads(j.read())

def add2dict(pt,a,dict):
    if pt not in dict:
        dict[pt] = [a]
    else:
        arr = dict.get(pt)
        arr.append(a)
        dict[pt] = arr
    return dict

def find_action_index(action):
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    i = 0
    for a in actions:
        if a[0] == action[0] and a[1] == action[1]:
            return i
        i+=1
    return False

def is_in_blocked_area(x,y):
    val = board[x][y]
    if val == 2 or val == 8:
        return True
    else:
        return False
def get_state_feature(x,y):
    reward_feature = np.zeros(6)
    if board[x][y] == 0:
        reward_feature[0] = 1
    elif board[x][y] == 1:
        #flag
        # reward_feature[0] = 1
        reward_feature[1] = 1
    elif board[x][y] == 2:
        #house
        # reward_feature[0] = 1
        pass
    elif board[x][y] == 3:
        #sheep
        # reward_feature[0] = 1
        reward_feature[2] = 1
    elif board[x][y] == 4:
        #coin
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[3] = 1
    elif board[x][y] == 5:
        #road block
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[4] = 1
    elif board[x][y] == 6:
        #mud area
        # reward_feature[0] = 1
        reward_feature[5] = 1
    elif board[x][y] == 7:
        #mud area + flag
        reward_feature[1] = 1
    elif board[x][y] == 8:
        #mud area + house
        pass
    elif board[x][y] == 9:
        #mud area + sheep
        reward_feature[2] = 1
    elif board[x][y] == 10:
        #mud area + coin
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[3] = 1
    elif board[x][y] == 11:
        #mud area + roadblock
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[4] = 1
    return reward_feature


def find_reward_features(traj,traj_length=3):
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y

    phi = np.zeros(6)

    for i in range (1,traj_length+1):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < 10 and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < 10 and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1]):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
            phi += get_state_feature(traj_ts_x,traj_ts_y)
        else:
            #only keep the gas/mud area score
            phi += (get_state_feature(traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])

        prev_x = traj_ts_x
        prev_y = traj_ts_y

    return phi

def find_end_state(traj,traj_length=3):
    in_gated =False
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y

    for i in range (1,traj_length+1):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < 10 and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < 10 and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1]):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]

            a = [traj_ts_x - prev_x, traj_ts_y - prev_y]
        else:
            a = [traj_ts_x + traj[i][0] - prev_x, traj_ts_y + traj[i][1] - prev_y]

        r = board_rf[prev_x][prev_y][find_action_index(a)]

        partial_return += r
        prev_x = traj_ts_x
        prev_y = traj_ts_y

    return traj_ts_x, traj_ts_y,partial_return

def get_all_statistics(questions,answers):
    pr_dsdt_res = {}
    pr_dsst_res = {}
    pr_ssst_res = {}
    pr_sss_res = {}

    vf_dsdt_res = {}
    vf_dsst_res = {}
    vf_ssst_res = {}
    vf_sss_res = {}

    none_dsdt_res = {}
    none_dsst_res = {}
    none_ssst_res = {}
    none_sss_res = {}

    prefrences = []
    delta_rs = []
    delta_v_sts = []
    delta_v_s0s = []
    delta_cis = []

    prefrences_dict = {}
    delta_rs_dict = {}
    delta_v_sts_dict = {}
    delta_v_s0s_dict = {}
    delta_cis_dict = {}


    for i in range(len(questions)):
        assignment_qs = questions[i]
        assignment_as = answers[i]
        sample_n = assignment_as[0]
        disp_id = None
        cords_id = [0,0]
        for q,a in zip(assignment_qs, assignment_as):
            if q == "observationType":

                if a == "0":
                    disp_id = "pr"
                elif a == "1":
                    disp_id = "vf"
                elif a == "2":
                    disp_id = "none"
                else:
                    print (a)
                    print ("disp id error")
                # print (disp_id)
                cords_id[1] = int(a)
                continue
            if q == "sampleNumber":
                # print (a)
                cords_id[0] = int(a)
                continue

            sample_dict_path = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/MTURK_interface/2021_07_29_data_samples/"  + disp_id + "_sample" + str(sample_n) + "/" + "sample" + str(sample_n) + "_dict.pkl"

            with open(sample_dict_path, 'rb') as f:
                sample_dict = pickle.load(f)
            num = int(q.replace("query",""))
            point = sample_dict.get(num)
            quad = point.get("quadrant")

            split_name = point.get("name").split("/")[-1].split("_")
            if (split_name[0] == "vf" or split_name[0] == "none"):
                pt = split_name[1]
                index = split_name[2]
            else:
                pt = split_name[0]
                index = split_name[1]


            if quad == "dsdt":
                traj_pairs = dsdt_data.get(pt)
            if quad == "dsst":
                traj_pairs = dsst_data.get(pt)
            if quad == "ssst":
                traj_pairs = ssst_data.get(pt)
            if quad == "sss":
                traj_pairs = sss_data.get(pt)


            pt_ = pt.replace("(","")
            pt_ = pt_.replace(")","")
            pt_ = pt_.split(",")
            x = float(pt_[0])
            y = float(pt_[1])


            poi = traj_pairs[int(index)]
            traj1 = poi[0]
            traj1_ts_x,traj1_ts_y, pr1 =find_end_state(traj1)
            traj1_v_s0 = board_vf[traj1[0][0]][traj1[0][1]]
            traj1_v_st = board_vf[traj1_ts_x][traj1_ts_y]

            traj2 = poi[1]
            traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2)
            traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
            traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]

            #make sure that our calculated pr/sv are the same as what the trajectory pair is marked as
            assert ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x)
            assert (pr2 - pr1 == y)

            disp_type = point.get("disp_type")
            dom_val = point.get("dom_val")

            delta_r = pr2 - pr1
            delta_v_s0 = traj2_v_s0 - traj1_v_s0
            delta_v_st = traj2_v_st - traj1_v_st

            if dom_val == "R":
                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"

            if a == "left":
                encoded_a = 0
            elif a == "right":
                encoded_a = 1
            elif a == "same":
                encoded_a = 0.5
            else:
                encoded_a = None

            if encoded_a != None:
                prefrences_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, encoded_a, prefrences_dict)
                delta_rs_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, delta_r, delta_rs_dict)
                delta_v_sts_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, delta_v_st, delta_v_sts_dict)
                delta_v_s0s_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, delta_v_s0, delta_v_s0s_dict)
                delta_cis_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, x, delta_cis_dict)
                prefrences.append(encoded_a)
                delta_rs.append(delta_r)
                delta_v_sts.append(delta_v_st)
                delta_v_s0s.append(delta_v_s0)
                delta_cis.append(x)


                if quad == "dsdt":
                    if disp_type == 0:
                        pr_dsdt_res = add2dict(pt,encoded_a,pr_dsdt_res)
                    elif disp_type == 1:
                        vf_dsdt_res = add2dict(pt,encoded_a,vf_dsdt_res)
                    elif disp_type == 3:
                        none_dsdt_res = add2dict(pt,encoded_a,none_dsdt_res)
                    else:
                        print ("KEY ERROR")

                elif quad == "dsst":
                    if disp_type == 0:
                        pr_dsst_res = add2dict(pt,encoded_a,pr_dsst_res)
                    elif disp_type == 1:
                        vf_dsst_res = add2dict(pt,encoded_a,vf_dsst_res)
                    elif disp_type == 3:
                        none_dsst_res = add2dict(pt,encoded_a,none_dsst_res)
                    else:
                        print ("KEY ERROR")
                elif quad == "ssst":
                    if disp_type == 0:
                        pr_ssst_res = add2dict(pt,encoded_a,pr_ssst_res)
                    elif disp_type == 1:
                        vf_ssst_res = add2dict(pt,encoded_a,vf_ssst_res)
                    elif disp_type == 3:
                        none_ssst_res = add2dict(pt,encoded_a,none_ssst_res)
                    else:
                        print ("KEY ERROR")
                elif quad == "sss":
                    if disp_type == 0:
                        pr_sss_res = add2dict(pt,encoded_a,pr_sss_res)
                    elif disp_type == 1:
                        vf_sss_res = add2dict(pt,encoded_a,vf_sss_res)
                    elif disp_type == 3:
                        none_sss_res = add2dict(pt,encoded_a,none_sss_res)
                    else:
                        print ("KEY ERROR")


    return [delta_rs_dict, delta_v_sts_dict, delta_v_s0s_dict],delta_cis_dict, prefrences_dict, prefrences, delta_rs, delta_v_sts, delta_v_s0s, delta_cis,pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res, vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res, none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res


def get_worker_pref_data(questions,answers,sample_folder,quad2data):
    pr_r = []
    pr_vs0 = []
    pr_vst = []
    pr_pref = []
    pr_res = {}

    vf_r = []
    vf_vs0 = []
    vf_vst = []
    vf_pref = []
    vf_res = {}

    none_r = []
    none_vs0 = []
    none_vst = []
    none_pref = []
    none_res = {}

    n_incorrect = 0
    n_correct = 0
    total_ = 0
    incorrect = 0
    total = 0

    n_lefts_pref = 0
    n_rights_pref = 0
    n_none_users = 0

    for i in range(len(questions)):
        assignment_qs = questions[i]
        assignment_as = answers[i]
        sample_n = assignment_as[0]
        disp_id = None
        cords_id = [0,0]
        for q,a in zip(assignment_qs, assignment_as):
            if a == "dis":
                continue
            if q == "observationType":
                if a == "0":
                    disp_id = "pr"
                elif a == "1":
                    disp_id = "vf"
                    n_none_users+=1
                elif a == "2":
                    disp_id = "none"

                else:
                    print (a)
                    print ("disp id error")
                # print (disp_id)
                cords_id[1] = int(a)
                continue
            if q == "sampleNumber":
                # print (a)
                cords_id[0] = int(a)
                continue

            sample_dict_path = "/Users/stephanehatgiskessell/Desktop/Kivy_stuff/MTURK_interface/" + sample_folder + "/"  + disp_id + "_sample" + str(sample_n) + "/" + "sample" + str(sample_n) + "_dict.pkl"

            with open(sample_dict_path, 'rb') as f:
                sample_dict = pickle.load(f)
            num = int(q.replace("query",""))
            point = sample_dict.get(num)
            quad = point.get("quadrant")
            total_+=1


            split_name = point.get("name").split("/")[-1].split("_")
            if (split_name[0] == "vf" or split_name[0] == "none"):
                pt = split_name[1]
                index = split_name[2]
            else:
                pt = split_name[0]
                index = split_name[1]

            quad = quad.replace("_formatted_imgs","") #bug from some augmented human data formatting
            traj_pairs = quad2data[quad].get(pt)

            # if quad == "aug-mul-len" or quad == "aug-t-nt-1" or quad == "aug-t-nt-2":

            # if quad == "aug-mul-len":
            #     continue

            #problem is with aug-t-nt-1

            pt_ = pt.replace("(","")
            pt_ = pt_.replace(")","")
            pt_ = pt_.split(",")
            x = float(pt_[0])
            y = float(pt_[1])



            poi = traj_pairs[int(index)]
            traj1 = poi[0]
            traj1_ts_x,traj1_ts_y, pr1 =find_end_state(traj1,len(traj1)-1)
            traj1_v_s0 = board_vf[traj1[0][0]][traj1[0][1]]
            traj1_v_st = board_vf[traj1_ts_x][traj1_ts_y]
            phi1 = find_reward_features(traj1,len(traj1)-1)

            traj2 = poi[1]
            traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2,len(traj2)-1)
            traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
            traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]
            phi2 = find_reward_features(traj2,len(traj2)-1)

            disp_type = point.get("disp_type")
            dom_val = point.get("dom_val")

            # if not (quad == "aug-mul-len" and (phi1[2] == 1 or phi2[2] == 1)):
            #     continue
            # if (quad != "aug-mul-len"):
            #     continue

            if a == "left":
                n_lefts_pref+=1
            if a == "right":
                n_rights_pref+=1

            if dom_val == "R":
                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"

            #make sure that our calculated pr/sv are the same as what the trajectory pair is marked as
            #sometimes things get flipped (bug in how augmented data was formatted after flipping), so flip back
            if not ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x):

                traj1 = poi[1]
                traj1_ts_x,traj1_ts_y, pr1 =find_end_state(traj1,len(traj1)-1)
                traj1_v_s0 = board_vf[traj1[0][0]][traj1[0][1]]
                traj1_v_st = board_vf[traj1_ts_x][traj1_ts_y]
                phi1 = find_reward_features(traj1,len(traj1)-1)

                traj2 = poi[0]
                traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2,len(traj2)-1)
                traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
                traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]
                phi2 = find_reward_features(traj2,len(traj2)-1)

                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"


            assert ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x)
            assert (pr2 - pr1 == y)

            er1 = pr1 + traj1_v_st - traj1_v_s0
            er2 = pr2 + traj2_v_st - traj2_v_s0

            if er1 > er2 and a != "left" or er2 > er1 and a != "right":
                incorrect+=1
            total+=1

            if a == "left":
                encoded_a = 0
                # encoded_a = [1,0]
            elif a == "right":
                encoded_a = 1
                # encoded_a = [0,1]
            elif a == "same":
                encoded_a = 0.5
                # encoded_a = [0.5,0.5]
            else:
                # print (a)
                encoded_a = None
            traj1_ses = [(traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y)]
            traj2_ses = [(traj2[0][0],traj2[0][1]),(traj2_ts_x,traj2_ts_y)]

            if disp_id == "vf":
                vf_r.append(pr2 - pr1)
                vf_vs0.append(traj2_v_s0 - traj1_v_s0)
                vf_vst.append(traj2_v_st - traj1_v_st)
                vf_pref.append(encoded_a)
                vf_res = add2dict(pt,encoded_a,vf_res)


            elif disp_id == "pr":
                pr_r.append(pr2 - pr1)
                pr_vs0.append(traj2_v_s0 - traj1_v_s0)
                pr_vst.append(traj2_v_st - traj1_v_st)
                pr_pref.append(encoded_a)
                pr_res = add2dict(pt,encoded_a,pr_res)


            elif disp_id == "none":
                none_r.append(pr2 - pr1)
                none_vs0.append(traj2_v_s0 - traj1_v_s0)
                none_vst.append(traj2_v_st - traj1_v_st)
                none_pref.append(encoded_a)
                none_res = add2dict(pt,encoded_a,none_res)


    print ("# of times left was preffered: " + str(n_lefts_pref))
    print ("# of times right was preffered: " + str(n_rights_pref))
    print (n_none_users)
    print ("\n")
    return vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref, vf_res, pr_res, none_res


def get_all_statistics_aug_human():
    quad2data1 = {"dsdt":dsdt_data,"dsst":dsst_data,"ssst":ssst_data, "sss":sss_data}
    sample_folder1 = "2021_07_29_data_samples"


    quad2data2 = {"aug-mul-len":multi_len_data, "aug-t-nt-1":t_nt_1_data, "aug-t-nt-2":t_nt_2_data}
    sample_folder2 = "2021_12_22_data_samples"


    vf_r = []
    vf_vs0 = []
    vf_vst = []
    vf_pref = []
    pr_r = []
    pr_vs0 = []
    pr_vst = []
    pr_pref = []
    none_r = []
    none_vs0 = []
    none_vst = []
    none_pref = []
    vf_res = {}
    pr_res = {}
    none_res = {}

    vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,  vf_res, pr_res, none_res = get_worker_pref_data(questions,answers,sample_folder1,quad2data1)
    vf_r2, vf_vs02, vf_vst2, vf_pref2, pr_r2, pr_vs02, pr_vst2, pr_pref2, none_r2, none_vs02, none_vst2, none_pref2,  vf_res2, pr_res2, none_res2= get_worker_pref_data(questions_aug,answers_aug,sample_folder2,quad2data2)


    vf_r.extend(vf_r2)
    vf_vs0.extend(vf_vs02)
    vf_vst.extend(vf_vst2)
    vf_pref.extend(vf_pref2)
    pr_r.extend(pr_r2)
    pr_vs0.extend(pr_vs02)
    pr_vst.extend(pr_vst2)
    pr_pref.extend(pr_pref2)
    none_r.extend(none_r2)
    none_vs0.extend(none_vs02)
    none_vst.extend(none_vst2)
    none_pref.extend(none_pref2)

    for k in vf_res2:
        if k in vf_res:
            vf_res[k].extend(vf_res2[k])
        else:
            vf_res.update({k:vf_res2[k]})

    for k in pr_res2:
        if k in pr_res:
            pr_res[k].extend(pr_res2[k])
        else:
            pr_res.update({k:pr_res2[k]})


    for k in none_res2:
        if k in none_res:
            none_res[k].extend(none_res2[k])
        else:
            none_res.update({k:none_res2[k]})

    return vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,  vf_res, pr_res, none_res



def multinomial_logistic_regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=.2, stratify=y,
                                                       random_state= 0)

    #https://home.ttic.edu/~nati/Publications/RennieSrebroIJCAI05.pdf
    # regressor = LogisticAT(alpha=1.0, verbose=0)

    regressor = LogisticRegression(random_state=0,multi_class="multinomial",class_weight="balanced")
    # regressor.fit(X_train, y_train)
    # pred = regressor.predict(X_test)


    print ("model coefficients:")
    regressor.fit(X_train,y_train)
    print (regressor.coef_)
    y_pred = regressor.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print ("confusion matrix:")
    print (cnf_matrix)
    #evaluating the score of the model
    scores = cross_val_score(regressor, X, y, cv=5,scoring='neg_log_loss')
    print("%0.2f mean negative log loss with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

def get_class_prob(y):
    n_lefts = 0
    n_rights = 0
    n_sames = 0
    for var in y:
        if var == 0:
            n_lefts +=1
        elif var == 0.5:
            n_sames +=1
        elif var == 1:
            n_rights +=1
    return [n_lefts/len(y), n_rights/len(y), n_sames/len(y)]

# def augment_data(X,Y):
#     aX = []
#     ay = []
#     for x,y in zip(X,Y):
#         aX.append(x)
#         ay.append(y)
#         neg_x = []
#         for val in x:
#             if val != 0:
#                 neg_x.append(-1*val)
#             else:
#                 neg_x.append(val)
#
#         aX.append(neg_x)
#         if y != 0:
#             ay.append(-1*y)
#         else:
#             ay.append(y)
#     return np.array(aX), np.array(ay)
def augment_data(X,Y):
    aX = []
    ay = []
    for x,y in zip(X,Y):
        aX.append(x)
        ay.append(y)
        neg_x = []
        for val in x:
            if val != 0:
                neg_x.append(-1*val)
            else:
                neg_x.append(val)

        aX.append(neg_x)
        if y == 0:
            ay.append(1)
        elif y == 1:
            ay.append(0)
        else:
            ay.append(0.5)
    return np.array(aX), np.array(ay)

def ordinal_logistic_regression_2(X,y,name):
    #https://github.com/Shopify/bevel
    from linear_ordinal_regression import OrderedLogit
    # ol.print_summary()
    kf = KFold(n_splits=10,random_state = 0,shuffle=True)
    total_log_loss = 0
    total_random_log_loss = 0
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        ol = OrderedLogit()
        ol.fit(X_train, y_train)

        class_probs = get_class_prob(y_train)

        X_test = X[test_index]
        y_test = y[test_index]
        y_pred = ol.predict_probabilities(X_test)
        log_loss = metrics.log_loss(y_test, y_pred)
        total_log_loss+=log_loss


        probabilities = [class_probs for _ in range(len(y_test))]
        random_log_loss = metrics.log_loss(y_test, probabilities)
        total_random_log_loss+=random_log_loss
    print (name)
    print("%0.2f mean log loss" % (total_log_loss/10))
    print('Baseline: Log Loss=%.3f' % (total_random_log_loss/10))

def ordinal_logistic_regression(X,y,name):
    kf = KFold(n_splits=10,random_state = 0,shuffle=True)
    total_log_loss = 0
    total_random_log_loss = 0
    condition_1_log_loss = 0
    condition_2_log_loss = 0
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        regressor = OrdinalClassifier(0)
        regressor.fit(X_train,y_train)

        class_probs = get_class_prob(y_train)

        X_test = X[test_index]
        y_test = y[test_index]
        y_pred = regressor.predict(X_test)
        log_loss = metrics.log_loss(y_test, y_pred)
        total_log_loss+=log_loss


        probabilities = [class_probs for _ in range(len(y_test))]
        random_log_loss = metrics.log_loss(y_test, probabilities)
        total_random_log_loss+=random_log_loss

        #Where all weights are 1 (and any bias weight is 0), resulting in change in expected return.
        regressor_c1 = OrdinalClassifier(0,fit_intercept=False)
        regressor_c1.fit(X_train,y_train) #only used to generate class vars
        regressor_c1.set_coefs(1,0,0)
        y_pred_c1 = regressor_c1.predict(X_test)
        log_loss_c1 = metrics.log_loss(y_test, y_pred_c1)
        condition_1_log_loss += log_loss_c1
        #Where only the weight on partial return is 1, and all others are 0.
        regressor_c2 = OrdinalClassifier(0,fit_intercept=False)
        regressor_c2.fit(X_train,y_train) #only used to generate class vars
        regressor_c2.set_coefs(1,1,-1)
        y_pred_c2 = regressor_c2.predict(X_test)
        log_loss_c2 = metrics.log_loss(y_test, y_pred_c2)
        condition_2_log_loss += log_loss_c2

    print (name)
    print("%0.2f mean log loss" % (total_log_loss/10))
    print('Baseline Log Loss=%.3f' % (total_random_log_loss/10))
    print('All coefficients are 1 Log Loss=%.3f' % (condition_1_log_loss/10))
    print('Only partial return coefficient is 1 Log Loss=%.3f' % (condition_2_log_loss/10))

def spearmens_test(independent,dependent,name):
    res = stats.spearmanr(independent, dependent)
    print (name)
    if res.pvalue >=0:
        print("r = %0.2f, p = %0.4f" % (res.correlation, res.pvalue))
    else:
        print ("This test is not run because there is no variation in available data points along this axis.")

def execute_quad_spearmans(quad,quad_name,disp_type,multi_quads=False):
    if multi_quads:
        xs = []
        ys = []
        all_actions = []
        for q in quad:
            for cord in q.keys():
                actions = q.get(cord)
                cord_ = cord.replace("(","")
                cord_ = cord_.replace(")","")
                cord_ = cord_.split(",")
                x = float(cord_[0])
                y = float(cord_[1])

                xs.extend([x for i in range(len(actions))])
                ys.extend([y for i in range(len(actions))])
                all_actions.extend(actions)

        # spearmens_test(xs,ys,"-- " + quad_name + " -- \nchange in state value and partial return: ")

        spearmens_test(xs,all_actions, "-- " + disp_type + " all points --" + "\nchange in state value: ")
        spearmens_test(ys,all_actions,  "-- " + disp_type + " all points --" + "\npartial return: ")
        print("\n")
    else:
        xs = []
        ys = []
        all_actions = []
        for cord in quad.keys():
            actions = quad.get(cord)
            cord_ = cord.replace("(","")
            cord_ = cord_.replace(")","")
            cord_ = cord_.split(",")
            x = float(cord_[0])
            y = float(cord_[1])

            xs.extend([x for i in range(len(actions))])
            ys.extend([y for i in range(len(actions))])
            all_actions.extend(actions)

        # spearmens_test(xs,ys,"-- " + quad_name + " -- \nchange in state value and partial return: ")
        if quad_name != "":
            prefix1 = "-- " + quad_name + " -- \nchange in state value: "
            prefix2 =  "-- " + quad_name + " -- \npartial return: "
        else:
            prefix1 = "change in state value: "
            prefix2 =  "partial return: "

        spearmens_test(xs,all_actions,prefix1)
        spearmens_test(ys,all_actions,prefix2)
        print("\n")

def format_y(Y):
    formatted_y= []
    n_lefts = 0
    n_rights = 0
    n_sames = 0

    for y in Y:
        if y == 0:
            n_lefts +=1
            formatted_y.append([0])
        elif y == 0.5:
            n_sames +=1
            formatted_y.append([0.5])
        elif y ==1:
            n_rights +=1
            formatted_y.append([1])
        else:
            print ("ERROR IN INPUT")
            assert False
    return torch.tensor(formatted_y,dtype=torch.float), [n_lefts/len(Y), n_rights/len(Y), n_sames/len(Y)]

def format_X_pr (X):
    formatted_X= []
    for x in X:
        formatted_X.append([x[0]])
    return torch.tensor(formatted_X,dtype=torch.float)

def format_X_er (X):
    formatted_X= []
    for x in X:
        formatted_X.append([x[0]+ x[1] - x[2]])# + x[1] - x[2]
    return torch.tensor(formatted_X,dtype=torch.float)

def format_X_full (X):
    return torch.tensor(X,dtype=torch.float)

def prefrence_pred_loss(output, target):
    batch_size = output.size()[0]
    output = torch.squeeze(torch.stack((output, torch.sub(1,output)),axis=2))
    output = torch.clamp(output,min=1e-35,max=None)
    output = torch.log(output)
    target = torch.squeeze(torch.stack((target, torch.sub(1,target)),axis=2))
    res = torch.mul(output,target)
    return -torch.sum(res)/batch_size


# class LogisticRegression(torch.nn.Module):
#      def __init__(self, input_size, bias):
#         super(LogisticRegression, self).__init__()
#         self.linear1 = torch.nn.Linear(input_size, 1,bias=bias)
#         #set initial weights to 0 for readability
#         with torch.no_grad():
#             self.linear1.weight = torch.nn.Parameter(torch.tensor([[0 for i in range(input_size)]],dtype=torch.float))
#      def forward(self, x):
#         y_pred = torch.sigmoid(self.linear1(x))
#         return y_pred

def sigmoid(x):
  print("logistic(c): " + str(1 / (1 + math.exp(-x))))

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_size, bias):
        super(LogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 1,bias=bias)
        self.c = torch.nn.Parameter(torch.tensor([0.0]))
        #set initial weights to 0 for readability
        with torch.no_grad():
            self.linear1.weight = torch.nn.Parameter(torch.tensor([[0 for i in range(input_size)]],dtype=torch.float))
     def forward(self, x):
        y_pred = torch.sigmoid(self.linear1(x))
        sig_c = torch.sigmoid(self.c)
        scaled_pred = torch.add(torch.mul(torch.sub(1,sig_c), y_pred),torch.divide(sig_c,2))
        return scaled_pred

def print_model_params(aX, ay,name,input_type,randomized):
    # X_train, X_test, y_train, y_test = train_test_split(aX, ay,
    #                                                    test_size=.2, stratify=ay,
    #                                                    random_state= 0)
    if input_type == "pr":
        X_train =format_X_pr(aX)
        input_size = 1
        bias = False
    elif input_type == "er":
        X_train =format_X_er(aX)
        input_size = 1
        bias = False
    elif input_type == "full":
        X_train =format_X_full(aX)
        input_size = 3
        bias = False
    y_train,_ = format_y(ay)


    model = LogisticRegression(input_size,bias)
    # criterion = torch.nn.BCELoss()
    # criterion = PrefrenceBCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    for epoch in range(3000):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train)
        # Compute Loss

        loss = prefrence_pred_loss(y_pred, y_train)
        # Backward pass
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        # print (name + "\n")
        if not randomized:
            for param_name, param in model.named_parameters():
                print (param_name)
                print (param)
            print ("\n")

def train_and_eval(aX, ay,name,input_type, randomized=False):
    torch.manual_seed(0)
    # print (len(aX))
    aX = np.array(aX)
    ay = np.array(ay)

    kf = KFold(n_splits=10,random_state = 0,shuffle=True)
    total_testing_log_loss = 0
    total_training_log_loss = 0
    total_random_log_loss = 0
    fold = 0

    for train_index, test_index in kf.split(aX):
        fold+=1
        X_train = aX[train_index]
        y_train = ay[train_index]
        X_test = aX[test_index]
        y_test = ay[test_index]

        if input_type == "pr":
            X_train =format_X_pr(X_train)
            input_size = 1
            bias = False
        elif input_type == "er":
            X_train =format_X_er(X_train)
            input_size = 1
            bias = False
        elif input_type == "full":
            X_train =format_X_full(X_train)
            input_size = 3
            bias = False
        else:
            input_size = 1
            bias = False
        y_train,_ = format_y(y_train)

        if not randomized:
            model = LogisticRegression(input_size,bias)
            # criterion = torch.nn.BCELoss()
            # criterion = PrefrenceBCELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

            for epoch in range(3000):

                model.train()
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(X_train)
                # Compute Loss

                # y_pred = torch.clamp(y_pred,min=1e-35,max=None)
                loss = prefrence_pred_loss(y_pred, y_train)
                # Backward pass
                loss.backward()
                optimizer.step()

            total_training_log_loss += prefrence_pred_loss(y_pred, y_train)
        else:
            random_pred = torch.tensor([[0.5] for i in range(len(y_train))],dtype=torch.float)
            total_training_log_loss +=prefrence_pred_loss(random_pred, y_train)

        with torch.no_grad():
            # print (name + "\n")
            y_test,class_probs = format_y(y_test)
            if not randomized:
                if input_type == "pr":
                    X_test =format_X_pr(X_test)
                elif input_type == "er":
                    X_test =format_X_er(X_test)
                elif input_type == "full":
                    X_test =format_X_full(X_test)
                y_pred_test = model(X_test)
                total_testing_log_loss+=prefrence_pred_loss(y_pred_test, y_test)


            else:
                random_pred = torch.tensor([[0.5] for i in range(len(y_test))],dtype=torch.float)
                total_testing_log_loss+=prefrence_pred_loss(random_pred, y_test)


    print("%0.5f mean testing log loss" % (total_testing_log_loss/10))
    print("%0.5f mean training log loss" % (total_training_log_loss/10))
    if not randomized:
        print_model_params(aX, ay,name,input_type, randomized)
        torch.save(model,name)
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
#
# def train_and_eval(aX, ay,name,input_type, randomized=False):
#     torch.manual_seed(0)
#     # print (len(aX))
#     X_train = np.array(aX)
#     y_train = np.array(ay)
#
#     total_testing_log_loss = 0
#     total_training_log_loss = 0
#     total_random_log_loss = 0
#
#     if input_type == "pr":
#         X_train =format_X_pr(X_train)
#         input_size = 1
#         bias = False
#     elif input_type == "er":
#         X_train =format_X_er(X_train)
#         input_size = 1
#         bias = False
#     elif input_type == "full":
#         X_train =format_X_full(X_train)
#         input_size = 3
#         bias = False
#     else:
#         input_size = 1
#         bias = False
#     y_train,_ = format_y(y_train)
#
#     if not randomized:
#         model = LogisticRegression(input_size,bias)
#         # criterion = torch.nn.BCELoss()
#         # criterion = PrefrenceBCELoss()
#         optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
#
#         for epoch in range(3000):
#
#             model.train()
#             optimizer.zero_grad()
#             # Forward pass
#             y_pred = model(X_train)
#             # Compute Loss
#
#             # y_pred = torch.clamp(y_pred,min=1e-35,max=None)
#             loss = prefrence_pred_loss(y_pred, y_train)
#             # Backward pass
#             loss.backward()
#             optimizer.step()
#         # print ("----")
#         # print (prefrence_pred_loss(y_pred, y_train))
#         total_training_log_loss += prefrence_pred_loss(y_pred, y_train)
#     else:
#         random_pred = torch.tensor([[0.5] for i in range(len(y_train))],dtype=torch.float)
#         total_training_log_loss +=prefrence_pred_loss(random_pred, y_train)
#
#     print("%0.5f Training log loss" % (total_training_log_loss))
#
#     if not randomized:
#
#         for param_name, param in model.named_parameters():
#             if (param_name == "c"):
#                 print ("logistic(c): " + str(sigmoid(param)))
#
#         print_model_params(aX, ay,name,input_type, randomized)
#         torch.save(model,name)

def execute_quad_log_reg(input_type,X_dicts,prefrences_dict, disp_type, quad, cords_dict, quad_name,multi_quads=False,randomized=False):
    if multi_quads:
        X = []
        Y = []
        for _dict,q in zip(cords_dict,quad):
            for cord in _dict.keys():
                key = str(disp_type) + "_" + q + "_" + str(cord)

                y = np.array(prefrences_dict.get(key))
                vars = []
                for d in X_dicts:
                    vars.append(d.get(key))
                x = np.stack((vars[0],vars[1],vars[2]),axis=1)

                X.extend(x)
                Y.extend(y)

        X = np.array(X)
        Y = np.array(Y)
        aX, ay = augment_data(X,Y)
        if disp_type == 0:
            name = "-- Partial Return --"
        elif disp_type == 1:
            name = "-- Expected Return --"
        elif disp_type == 3:
            name = "-- No Info Shown --"

    else:
        X = []
        Y = []
        for cord in cords_dict.keys():
            key = str(disp_type) + "_" + quad + "_" + str(cord)

            y = np.array(prefrences_dict.get(key))
            vars = []
            for d in X_dicts:
                vars.append(d.get(key))

            x = np.stack((vars[0],vars[1],vars[2]),axis=1)
            X.extend(x)
            Y.extend(y)

        X = np.array(X)
        Y = np.array(Y)

        aX, ay = augment_data(X,Y)
        name = "-- " + quad_name + " --"
        print (name)
    train_and_eval(aX,ay,name,input_type,randomized=randomized)


def unison_shuffled_copies(a, b, c, d):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    assert len(a) == len(b) == len(c) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]

#
# print ("------------------------------------------------------")
# print ("             Augmented Human Data Results             ")
# print ("------------------------------------------------------")
#
#
# vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,_,_,_  = get_all_statistics_aug_human()
#
# vf_r, vf_vs0, vf_vst,vf_pref = unison_shuffled_copies(vf_r, vf_vs0, vf_vst, vf_pref)
# pr_r, pr_vs0, pr_vst, pr_pref = unison_shuffled_copies( pr_r, pr_vs0, pr_vst, pr_pref)
# none_r, none_vs0, none_vst, none_pref = unison_shuffled_copies(none_r, none_vs0, none_vst, none_pref)
#
# print ("--------------------------------------------------------------")
# print ("  Expected Return Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
#
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/er_model_pr_data","er", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/er_model_er_data","er", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/er_model_none_data","er", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("--------------------------------------------------------------")
# print ("  Partial Return Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/pr_model_pr_data","pr", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/pr_model_er_data","pr", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/pr_model_none_data","pr", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("--------------------------------------------------------------")
# print ("  Fully Expressed Model - Logistic Regression Test Results ")
# print ("Note: The order of the input is (1) partial return, (2) end state value, and (3) start state value")
# print ("--------------------------------------------------------------")
#
# # #
#
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/full_model_pr_data","full", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/full_model_er_data","full", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# end = int(len(X)/5)
# n = 0
# for i in range(0,len(X),end):
#     if n >= 5:
#         break
#     n+=1
#     aX, ay =  augment_data(X[i:i+end],Y[i:i+end])
#     train_and_eval(aX, ay,"preference_models/full_model_none_data","full", randomized=False)
#     print ("\n")
# print ("\n\n")
#
# print ("--------------------------------------------------------------")
# print (" Uninformed Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
# #
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"random_model","", randomized=True)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"random_model","", randomized=True)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"random_model","", randomized=True)
# print ("\n")


# #
# vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,vf_res,pr_res,none_res  = get_all_statistics_aug_human()
# # #
# print ("------------------------------------------------------")
# print ("       Spearman’s rank correlation test results       ")
# print ("------------------------------------------------------")
#
#
# print ("------------ Condition 1 - partial return shown ------------")
# execute_quad_spearmans(pr_res,"Augmented Data", "partial return")
#
# print ("------------ Condition 2 - expected return shown ------------")
# execute_quad_spearmans(vf_res,"Augmented Data","expected return")
#
# print ("------------ Condition 3 - No info shown ------------")
# execute_quad_spearmans(none_res,"Augmented Data","no info")


#
#
# vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,vf_res,pr_res,none_res  = get_all_statistics_aug_human()
# #
#
# print ("------------ Condition 1 - partial return shown ------------")
# pr_ys2res = {}
# for cord in pr_res:
#     cord_ = cord.replace("(","").replace(")","")
#     cord_ = cord_.split(",")
#     x = float(cord_[0])
#     y = float(cord_[1])
#
#     if y not in pr_ys2res:
#         pr_ys2res[y] = {}
#         pr_ys2res[y].update({cord:pr_res[cord]})
#     else:
#         pr_ys2res[y].update({cord:pr_res[cord]})
# pr_ys2res_sorted = sorted(pr_ys2res)
# for y in pr_ys2res_sorted:
#     # if y == -55:
#     #     print (pr_ys2res[y])
#     if len(pr_ys2res[y].keys()) <= 2:
#         continue
#
#     print ("Y coordinate: " + str(y))
#     print ("Number of x values in dataset: " + str(len(pr_ys2res[y].keys())))
#     print ("Represented coordinates in dataset: " + str(sorted(pr_ys2res[y].keys())))
#     total_prefs = 0
#     for c in pr_ys2res[y].keys():
#         total_prefs+= len(pr_ys2res[y][c])
#     print ("Total number of preferences: " + str(total_prefs))
#
#     execute_quad_spearmans(pr_ys2res[y],"", "partial return")
#     print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# vf_ys2res = {}
# for cord in vf_res:
#     cord_ = cord.replace("(","").replace(")","")
#     cord_ = cord_.split(",")
#     x = float(cord_[0])
#     y = float(cord_[1])
#
#     if y not in vf_ys2res:
#         vf_ys2res[y] = {}
#         vf_ys2res[y].update({cord:vf_res[cord]})
#     else:
#         vf_ys2res[y].update({cord:vf_res[cord]})
#
#
# vf_ys2res_sorted = sorted(vf_ys2res)
# for y in vf_ys2res_sorted:
#     if len(vf_ys2res[y].keys()) <= 2:
#         continue
#
#     print ("Y coordinate: " + str(y))
#     print ("Number of x values in dataset: " + str(len(vf_ys2res[y].keys())))
#     print ("Represented coordinates in dataset: " + str(sorted(vf_ys2res[y].keys())))
#     total_prefs = 0
#     for c in vf_ys2res[y].keys():
#         total_prefs+= len(vf_ys2res[y][c])
#     print ("Total number of preferences: " + str(total_prefs))
#
#     execute_quad_spearmans(vf_ys2res[y],"", "expected return")
#
#
# print ("------------ Condition 3 - No info shown ------------")
# none_ys2res = {}
# for cord in none_res:
#     cord_ = cord.replace("(","").replace(")","")
#     cord_ = cord_.split(",")
#     x = float(cord_[0])
#     y = float(cord_[1])
#
#     if y not in none_ys2res:
#         none_ys2res[y] = {}
#         none_ys2res[y].update({cord:none_res[cord]})
#     else:
#         none_ys2res[y].update({cord:none_res[cord]})
#
# none_ys2res_sorted = sorted(none_ys2res)
# for y in none_ys2res_sorted:
#     if len(none_ys2res[y].keys()) <= 2:
#         continue
#
#     print ("Y coordinate: " + str(y))
#     print ("Number of x values in dataset: " + str(len(none_ys2res[y].keys())))
#     print ("Represented coordinates in dataset: " + str(sorted(none_ys2res[y].keys())))
#
#     total_prefs = 0
#     for c in none_ys2res[y].keys():
#         total_prefs+= len(none_ys2res[y][c])
#     print ("Total number of preferences: " + str(total_prefs))
#
#     execute_quad_spearmans(none_ys2res[y],"", "no info")
#

# print ("------------------------------------------------------")
# print ("             Augmented Human Data Results             ")
# print ("------------------------------------------------------")
#
# #
# vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,_,_,_  = get_all_statistics_aug_human()
#
#
# print ("--------------------------------------------------------------")
# print ("  Expected Return Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
#
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
#
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/er_model_pr_data","er", randomized=False)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/er_model_er_data","er", randomized=False)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/er_model_none_data","er", randomized=False)
# print ("\n")
#
# print ("--------------------------------------------------------------")
# print ("  Partial Return Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/pr_model_pr_data","pr", randomized=False)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/pr_model_er_data","pr", randomized=False)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/pr_model_none_data","pr", randomized=False)
# print ("\n")
#
# print ("--------------------------------------------------------------")
# print ("  Fully Expressed Model - Logistic Regression Test Results ")
# print ("Note: The order of the input is (1) partial return, (2) end state value, and (3) start state value")
# print ("--------------------------------------------------------------")
#
# # #
#
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/full_model_pr_data","full", randomized=False)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/full_model_er_data","full", randomized=False)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"preference_models/full_model_none_data","full", randomized=False)
# print ("\n")
#
# print ("--------------------------------------------------------------")
# print (" Uninformed Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
# #
# print ("------------ Condition 1 - partial return shown ------------")
# X = np.stack([pr_r, pr_vst, pr_vs0], axis = 1)
# Y = pr_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"random_model","", randomized=True)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# X = np.stack([vf_r, vf_vst, vf_vs0], axis = 1)
# Y = vf_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"random_model","", randomized=True)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# X = np.stack([none_r, none_vst, none_vs0], axis = 1)
# Y = none_pref
# aX, ay =  augment_data(X,Y)
# train_and_eval(aX, ay,"random_model","", randomized=True)
# print ("\n")
#
#



#
# X_dicts, delta_cis_dict,prefrences_dict, prefrences, delta_rs, delta_v_sts, delta_v_s0s, delta_cis,pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res, vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res, none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res = get_all_statistics(questions,answers)
#
#
# X = np.array([delta_rs, delta_v_sts,delta_v_s0s]).T
# y = np.array(prefrences)
# #
# print ("------------------------------------------------------")
# print ("       Spearman’s rank correlation test results       ")
# print ("------------------------------------------------------")
#
#
# print ("------------ Condition 1 - partial return shown ------------")
# execute_quad_spearmans(pr_dsdt_res,"different start state / different end state","partial return")
# execute_quad_spearmans(pr_dsst_res,"different start state / same end state","partial return")
# execute_quad_spearmans(pr_ssst_res,"same start state / same end state","partial return")
# execute_quad_spearmans(pr_sss_res,"same start state / different end state","partial return")
# execute_quad_spearmans([pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res],"All data","partial_return",multi_quads=True)
#
# print ("------------ Condition 2 - expected return shown ------------")
# execute_quad_spearmans(vf_dsdt_res,"different start state / different end state","expected return")
# execute_quad_spearmans(vf_dsst_res,"different start state / same end state","expected return")
# execute_quad_spearmans(vf_ssst_res,"same start state / same end state","expected return")
# execute_quad_spearmans(vf_sss_res,"same start state / different end state","expected return")
# execute_quad_spearmans([vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res],"All data","expected_return",multi_quads=True)
#
# print ("------------ Condition 3 - No info shown ------------")
# execute_quad_spearmans(none_dsdt_res,"different start state / different end state","no info")
# execute_quad_spearmans(none_dsst_res,"different start state / same end state","no info")
# execute_quad_spearmans(none_ssst_res,"same start state / same end state","no info")
# execute_quad_spearmans(none_sss_res,"same start state / different end state","no info")
# execute_quad_spearmans([none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res],"All data","no info",multi_quads=True)

# print ("--------------------------------------------------------------")
# print ("  Expected Return Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
# #
# print ("------------ Condition 1 - partial return shown ------------")
# execute_quad_log_reg("er",X_dicts,prefrences_dict, 0, ["dsdt","dsst","ssst","sss"], [pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# execute_quad_log_reg("er",X_dicts,prefrences_dict, 1, ["dsdt","dsst","ssst","sss"], [vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# execute_quad_log_reg("er",X_dicts,prefrences_dict, 3, ["dsdt","dsst","ssst","sss"], [none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 4 - All Data Used ------------")
# train_and_eval(X, y,"","er", randomized=False)
#
# print ("--------------------------------------------------------------")
# print ("  Partial Return Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
# #
# print ("------------ Condition 1 - partial return shown ------------")
# execute_quad_log_reg("pr",X_dicts,prefrences_dict, 0, ["dsdt","dsst","ssst","sss"], [pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# execute_quad_log_reg("pr",X_dicts,prefrences_dict, 1, ["dsdt","dsst","ssst","sss"], [vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# execute_quad_log_reg("pr",X_dicts,prefrences_dict, 3, ["dsdt","dsst","ssst","sss"], [none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 4 - All Data Used ------------")
# train_and_eval(X, y,"","pr", randomized=False)
#
#
# print ("--------------------------------------------------------------")
# print ("  Fully Expressed Model - Logistic Regression Test Results ")
# print ("Note: The order of the input is (1) partial return, (2) end state value, and (3) start state value")
# print ("--------------------------------------------------------------")
#
# # #
# print ("------------ Condition 1 - partial return shown ------------")
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 0, "dsdt",pr_dsdt_res,"Different Start State / Different End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 0, "dsst",pr_dsst_res,"Different Start State / Same End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 0, "ssst",pr_ssst_res,"Same Start State / Same End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 0, "sss",pr_sss_res,"Same Start State / Different End State",multi_quads=False,randomized=False)
# print ("-- All Quadrants -- ")
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 0, ["dsdt","dsst","ssst","sss"], [pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 1, "dsdt",vf_dsdt_res,"Different Start State / Different End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 1, "dsst",vf_dsst_res,"Different Start State / Same End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 1, "ssst",vf_ssst_res,"Same Start State / Same End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 1, "sss",vf_sss_res,"Same Start State / Different End State",multi_quads=False,randomized=False)
# print ("-- All Quadrants -- ")
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 1, ["dsdt","dsst","ssst","sss"], [vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 3, "dsdt",none_dsdt_res,"Different Start State / Different End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 3, "dsst",none_dsst_res,"Different Start State / Same End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 3, "ssst",none_ssst_res,"Same Start State / Same End State",multi_quads=False,randomized=False)
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 3, "sss",none_sss_res,"Same Start State / Different End State",multi_quads=False,randomized=False)
# print ("-- All Quadrants -- ")
# execute_quad_log_reg("full",X_dicts,prefrences_dict, 3, ["dsdt","dsst","ssst","sss"], [none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res],"",multi_quads=True,randomized=False)
# print ("\n")
#
# print ("------------ Condition 4 - All Data Used ------------")
# train_and_eval(X, y,"","full", randomized=False)
#
#
# print ("--------------------------------------------------------------")
# print (" Uninformed Model - Logistic Regression Test Results ")
# print ("--------------------------------------------------------------")
#
# #
# print ("------------ Condition 1 - partial return shown ------------")
# execute_quad_log_reg("",X_dicts,prefrences_dict, 0, ["dsdt","dsst","ssst","sss"], [pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res],"",multi_quads=True,randomized=True)
# print ("\n")
#
# print ("------------ Condition 2 - expected return shown ------------")
# execute_quad_log_reg("",X_dicts,prefrences_dict, 1, ["dsdt","dsst","ssst","sss"], [vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res],"",multi_quads=True,randomized=True)
# print ("\n")
#
# print ("------------ Condition 3 - No info shown ------------")
# execute_quad_log_reg("",X_dicts,prefrences_dict, 3, ["dsdt","dsst","ssst","sss"], [none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res],"",multi_quads=True,randomized=True)
# print ("\n")
#
# print ("------------ Condition 4 - All Data Used ------------")
# train_and_eval(X, y,"","", randomized=True)
