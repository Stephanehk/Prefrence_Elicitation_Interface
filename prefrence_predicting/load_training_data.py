import numpy as np
import pickle
import re
import json
import math

with open('../2021_08_18_woi_questions.data', 'rb') as f:
    questions = pickle.load(f)
with open('../2021_08_18_woi_answers.data', 'rb') as f:
    answers = pickle.load(f)

dsdt_data = "../saved_data/2021_07_29_dsdt_chosen.json"
dsst_data = "../saved_data/2021_07_29_dsst_chosen.json"
ssst_data = "../saved_data/2021_07_29_ssst_chosen.json"
sss_data = "../saved_data/2021_07_29_sss_chosen.json"
board = "../saved_data/2021-07-29_sparseboard2-notrap_board.json"
board_vf = "../saved_data/2021-07-29_sparseboard2-notrap_value_function.json"
board_rf = "../saved_data/2021-07-29_sparseboard2-notrap_rewards_function.json"



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
with open(board, 'r') as j:
    board = json.loads(j.read())
with open(board_rf, 'r') as j:
    board_rf = json.loads(j.read())


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



def generate_t_nt_samples(terminating, non_terminating):
    phis = []
    y = []
    rewards = []
    ses = []
    for i in range (min(len(terminating),len(non_terminating))):

        #[gas, goal, sheep, coin, roadblock, mud]

        # assert (terminating[i][1] == np.dot(terminating[i][0], [-1,50,-50,1,-1,-2]))

        # print (non_terminating[i][1])
        # print (non_terminating[i][0])
        # print ("\n")
        # assert (non_terminating[i][1] == np.dot(non_terminating[i][0], [-1,50,-50,1,-1,-2]))

        # if terminating[i][1] > non_terminating[i][1]:
        #     y.append([1,0])
        # elif terminating[i][1] < non_terminating[i][1]:
        #     y.append([0,1])
        # elif terminating[i][1] == non_terminating[i][1]:
        #     continue
        #     y.append([0.5,0.5])
        y.append(None)

        phis.append([terminating[i][0],non_terminating[i][0]])
        rewards.append([terminating[i][1], non_terminating[i][1]])
        ses.append([terminating[i][2],non_terminating[i][2]])

    return phis,y,rewards,ses

def truncate_traj_1(traj):
    #truncate traj and get new start state
    x,y,_ = find_end_state (traj[:3],traj_length=2)
    traj= traj[2:]
    traj[0] = [x,y]

    traj_ts_x,traj_ts_y, pr =find_end_state(traj,traj_length=1)
    phi = find_reward_features(traj,traj_length=1)
    return pr, phi,[(x,y),(traj_ts_x,traj_ts_y)]

def truncate_traj_2(traj):
    #truncate traj and get new start state
    x,y,_ = find_end_state (traj[:2],traj_length=1)
    traj= traj[1:]
    traj[0] = [x,y]

    traj_ts_x,traj_ts_y, pr =find_end_state(traj,traj_length=2)
    phi = find_reward_features(traj,traj_length=2)
    return pr, phi,[(x,y),(traj_ts_x,traj_ts_y)]


def get_all_statistics(questions=questions,answers=answers,include_dif_traj_lengths = False):
    pr_X = []
    vf_X = []
    none_X = []

    pr_X_terminating = []
    vf_X_terminating = []
    none_X_terminating = []

    pr_X_non_terminating = []
    vf_X_non_terminating = []
    none_X_non_terminating = []

    pr_y = []
    vf_y = []
    none_y = []

    pr_r = []
    vf_r = []
    none_r = []

    pr_ses = []
    vf_ses = []
    none_ses = []

    n_incorrect = 0
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

            phi1 = find_reward_features(traj1)

            traj2 = poi[1]
            traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2)
            traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
            traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]

            phi2 = find_reward_features(traj2)

            #make sure that our calculated pr/sv are the same as what the trajectory pair is marked as
            assert ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x)
            assert (pr2 - pr1 == y)

            #get truncated trajectories
            os_trunc_traj1_pr,os_trunc_traj1_phi, os_ses1 = truncate_traj_1(traj1)
            os_trunc_traj2_pr,os_trunc_traj2_phi, os_ses2 = truncate_traj_1(traj2)

            ts_trunc_traj1_pr,ts_trunc_traj1_phi, ts_ses1 = truncate_traj_2(traj1)
            ts_trunc_traj2_pr,ts_trunc_traj2_phi, ts_ses2 = truncate_traj_2(traj2)

            # if (ts_trunc_traj1_pr != ts_trunc_traj2_pr):
            #     print (ts_trunc_traj1_phi)
            #     print (ts_trunc_traj2_phi)
            #     print ("\n")
            #
            # if (os_trunc_traj1_pr != os_trunc_traj2_pr):
            #     print (os_trunc_traj1_phi)
            #     print (os_trunc_traj2_phi)
            #     print ("\n")
            # assert False

            #TODO: THIS IS A POTENTIALLY MAJOR BUG, RIGHT NOW IT IS NOT VERY IMPACTFUL BUT MAKE SURE TO FIX LATER
            if (pr1 != np.dot(phi1, [-1,50,-50,1,-1,-2])) or (pr2 != np.dot(phi2, [-1,50,-50,1,-1,-2])):
                n_incorrect+=1
                continue

            disp_type = point.get("disp_type")
            dom_val = point.get("dom_val")

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
                vf_X.append([phi1,phi2])
                vf_r.append([pr1,pr2])
                vf_y.append(encoded_a)
                vf_ses.append([traj1_ses,traj2_ses])
                if quad == "dsst" or quad == "ssst":
                    vf_X_terminating.append([phi1,pr1,traj1_ses])
                    vf_X_terminating.append([phi2,pr2,traj2_ses])

                    if include_dif_traj_lengths:
                        vf_X.append([phi1, os_trunc_traj1_phi])
                        vf_r.append([pr1, os_trunc_traj1_pr])
                        vf_y.append(None)
                        vf_ses.append([traj1_ses,os_ses1])


                        vf_X.append([phi1, ts_trunc_traj1_phi])
                        vf_r.append([pr1, ts_trunc_traj1_pr])
                        vf_y.append(None)
                        vf_ses.append([traj1_ses,ts_ses1])

                        vf_X.append([phi2, os_trunc_traj2_phi])
                        vf_r.append([pr2, os_trunc_traj2_pr])
                        vf_y.append(None)
                        vf_ses.append([traj2_ses,os_ses2])

                        vf_X.append([phi2, ts_trunc_traj2_phi])
                        vf_r.append([pr2, ts_trunc_traj2_pr])
                        vf_y.append(None)
                        vf_ses.append([traj2_ses,ts_ses2])

                elif quad == "dsdt" or quad == "sss":
                    vf_X_non_terminating.append([phi1,pr1,traj1_ses])
                    vf_X_non_terminating.append([phi2,pr2,traj2_ses])

            elif disp_id == "pr":
                pr_X.append([phi1,phi2])
                pr_r.append([pr1,pr2])
                pr_y.append(encoded_a)
                pr_ses.append([traj1_ses,traj2_ses])
                if quad == "dsst" or quad == "ssst":
                    pr_X_terminating.append([phi1, pr1, traj1_ses])
                    pr_X_terminating.append([phi2, pr2, traj2_ses])

                    if include_dif_traj_lengths:
                        pr_X.append([phi1, os_trunc_traj1_phi])
                        pr_r.append([pr1, os_trunc_traj1_pr])
                        pr_y.append(None)
                        pr_ses.append([traj1_ses, os_ses1])

                        pr_X.append([phi1, ts_trunc_traj1_phi])
                        pr_r.append([pr1, ts_trunc_traj1_pr])
                        pr_y.append(None)
                        pr_ses.append([traj1_ses, ts_ses1])

                        pr_X.append([phi2, os_trunc_traj2_phi])
                        pr_r.append([pr2, os_trunc_traj2_pr])
                        pr_y.append(None)
                        pr_ses.append([traj2_ses, os_ses2])

                        pr_X.append([phi2, ts_trunc_traj2_phi])
                        pr_r.append([pr2, ts_trunc_traj2_pr])
                        pr_y.append(None)
                        pr_ses.append([traj2_ses, ts_ses2])


                elif quad == "dsdt" or quad == "sss":
                    pr_X_non_terminating.append([phi1 ,pr1, traj1_ses])
                    pr_X_non_terminating.append([phi2, pr2, traj2_ses])

            elif disp_id == "none":
                none_X.append([phi1,phi2])
                none_r.append([pr1,pr2])
                none_y.append(encoded_a)
                none_ses.append([traj1_ses,traj2_ses])
                if quad == "dsst" or quad == "ssst":
                    none_X_terminating.append([phi1, pr1, traj1_ses])
                    none_X_terminating.append([phi2, pr2, traj2_ses])

                    if include_dif_traj_lengths:
                        none_X.append([phi1, os_trunc_traj1_phi])
                        none_r.append([pr1, os_trunc_traj1_pr])
                        none_y.append(None)
                        none_ses.append([traj1_ses, os_ses1])

                        none_X.append([phi1, ts_trunc_traj1_phi])
                        none_r.append([pr1, ts_trunc_traj1_pr])
                        none_y.append(None)
                        none_ses.append([traj1_ses, ts_ses1])

                        none_X.append([phi2, os_trunc_traj2_phi])
                        none_r.append([pr2, os_trunc_traj2_pr])
                        none_y.append(None)
                        none_ses.append([traj2_ses, os_ses2])

                        none_X.append([phi2, ts_trunc_traj2_phi])
                        none_r.append([pr2, ts_trunc_traj2_pr])
                        none_y.append(None)
                        none_ses.append([traj2_ses, ts_ses2])

                elif quad == "dsdt" or quad == "sss":
                    none_X_non_terminating.append([phi1, pr1, traj1_ses])
                    none_X_non_terminating.append([phi2, pr2, traj2_ses])

    vf_X_add, vf_y_add, vf_add_r, vf_add_ses = generate_t_nt_samples(vf_X_terminating, vf_X_non_terminating)
    pr_X_add, pr_y_add, pr_add_r, pr_add_ses = generate_t_nt_samples(pr_X_terminating, pr_X_non_terminating)
    none_X_add, none_y_add, none_add_r, none_add_ses = generate_t_nt_samples(none_X_terminating, none_X_non_terminating)

    # adds syntheitc prefrences between termianting and non-terminating trajectory
    vf_X.extend(vf_X_add)
    vf_r.extend(vf_add_r)
    vf_y.extend(vf_y_add)
    vf_ses.extend(vf_add_ses)

    pr_X.extend(pr_X_add)
    pr_r.extend(pr_add_r)
    pr_y.extend(pr_y_add)
    pr_ses.extend(pr_add_ses)

    none_X.extend(none_X_add)
    none_r.extend(none_add_r)
    none_y.extend(none_y_add)
    none_ses.extend(none_add_ses)

    # print (n_incorrect)

    return vf_X, vf_r, vf_y, vf_ses, pr_X, pr_r, pr_y, pr_ses, none_X, none_r, none_y, none_ses
