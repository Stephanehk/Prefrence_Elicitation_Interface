import numpy as np
import random
import pickle
import itertools
from itertools import permutations
from itertools import combinations


from grid_world import GridWorldEnv
from value_iteration import value_iteration, follow_policy, learn_successor_feature,get_gt_avg_return,build_pi,iterative_policy_evaluation
from generate_random_policies import generate_all_policies, calc_value


def randomly_place_item_exact(env,id,N,height,width):
    #randomly place mud
    for i in range(N):
        x = random.randint(0,height-1)
        y = random.randint(0,width-1)
        while env.board[x][y] != 0:
            x = random.randint(0,height-1)
            y = random.randint(0,width-1)
        env.board[x][y] = id

def contains_cords(arr1,arr2):
    for a in arr1:
        if a[0] == arr2[0] and a[1] == arr2[1]:
            return True
    return False

def find_action_index(actions, action):
    i = 0
    for a in actions:
        if a[0] == action[0] and a[1] == action[1]:
            return i
        i+=1
    return False


def is_in_gated_area(x,y,board):
    val = board[x][y]
    if  val >= 6:
        return True
    else:
        return False

def is_in_blocked_area(x,y,board):
    val = board[x][y]
    if val == 2 or val == 8:
        return True
    else:
        return False

def find_end_state(traj,board):
    in_gated =False
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    if is_in_gated_area(traj_ts_x,traj_ts_y,board):
        in_gated = True

    for i in range (1,4):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(env.board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(env.board[0]):
            if not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1],env.board):
                next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1],env.board)
                if in_gated == False or  (in_gated and next_in_gated):
                    traj_ts_x += traj[i][0]
                    traj_ts_y += traj[i][1]
    return traj_ts_x, traj_ts_y


def get_state_feature(env,x,y):
    reward_feature = np.zeros(6)
    if env.board[x][y] == 0:
        reward_feature[0] = 1
    elif env.board[x][y] == 1:
        #flag
        # reward_feature[0] = 1
        reward_feature[1] = 1
    elif env.board[x][y] == 2:
        #house
        # reward_feature[0] = 1
        pass
    elif env.board[x][y] == 3:
        #sheep
        # reward_feature[0] = 1
        reward_feature[2] = 1
    elif env.board[x][y] == 4:
        #coin
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[3] = 1
    elif env.board[x][y] == 5:
        #road block
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[4] = 1
    elif env.board[x][y] == 6:
        #mud area
        # reward_feature[0] = 1
        reward_feature[5] = 1
    elif env.board[x][y] == 7:
        #mud area + flag
        reward_feature[1] = 1
    elif env.board[x][y] == 8:
        #mud area + house
        pass
    elif env.board[x][y] == 9:
        #mud area + sheep
        reward_feature[2] = 1
    elif env.board[x][y] == 10:
        #mud area + coin
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[3] = 1
    elif env.board[x][y] == 11:
        #mud area + roadblock
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[4] = 1
    return reward_feature


def find_reward_features(traj,env,traj_length=3):
    GAMMA=1
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    phi = np.zeros(6)
    phi_dis = np.zeros(6)

    for i in range (1,traj_length+1):
        # print ("===========================")
        # print (traj_ts_x,traj_ts_y)
        # print ("===========================")
        #check if we are at terminal state
        if env.board[prev_x, prev_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
            continue
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(env.board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(env.board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], env.board):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
            # print ("state feature: ")
            # print (get_state_feature(traj_ts_x,traj_ts_y,env))
            dis_state_sf = (GAMMA**(i-1))*get_state_feature(env,traj_ts_x,traj_ts_y)
            state_sf = get_state_feature(env,traj_ts_x,traj_ts_y)
        else:
            #check if we are at terminal state
            if env.board[prev_x, prev_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
                dis_state_sf = [0,0,0,0,0,0]
                state_sf = [0,0,0,0,0,0]
            else:
                dis_state_sf = (GAMMA**(i-1))*(get_state_feature(env,traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])
                state_sf = (get_state_feature(env,traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])
        phi_dis += dis_state_sf
        phi+= state_sf

        prev_x = traj_ts_x
        prev_y = traj_ts_y
    # print ("--done--\n")
    return phi_dis,phi
#
# def find_reward_features(traj, env, traj_length=3):
#     GAMMA = 1
#     traj_ts_x = traj[0][0]
#     traj_ts_y = traj[0][1]
#     # if is_in_gated_area(traj_ts_x,traj_ts_y):
#     #     in_gated = True
#
#     partial_return = 0
#     prev_x = traj_ts_x
#     prev_y = traj_ts_y
#
#     phi = np.zeros(6)
#     phi_dis = np.zeros(6)
#
#     for i in range (1,traj_length+1):
#         if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(env.board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(env.board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], env.board):
#             # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
#             # if in_gated == False or  (in_gated and next_in_gated):
#             traj_ts_x += traj[i][0]
#             traj_ts_y += traj[i][1]
#         if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
#             phi_dis += (GAMMA**(i-1))*get_state_feature(env,traj_ts_x,traj_ts_y)
#             phi += get_state_feature(env,traj_ts_x,traj_ts_y)
#         else:
#             #only keep the gas/mud area score
#             phi_dis += (GAMMA**(i-1))*(get_state_feature(env,traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])
#             phi += (get_state_feature(env,traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])
#
#         prev_x = traj_ts_x
#         prev_y = traj_ts_y
#
#     return phi_dis,phi

def create_traj(s0_x, s0_y,action_seq,traj_length, board, rewards_function,terminal_states,blocking_cords,values):
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #generate trajectory 1
    # t_partial_r_sum = rewards_model[s0_x][s0_y]
    t_partial_r_sum = 0

    traj = [(s0_x,s0_y)]
    x = s0_x
    y = s0_y
    step_n = 0
    is_terminal = False
    states = [(s0_x,s0_y)]
    # print ("start: " + str((x,y)))
    for step1 in range(traj_length+1):
        if contains_cords(terminal_states,[x,y]):
            # if step_n != traj_length:
            #     return False, False, False, False, False, False
            # else:
            is_terminal = True

        if step_n != traj_length:
            a = action_seq[step1]

            traj.append(a)
            a_i = find_action_index(actions,a)
            # print (((x,y),rewards_function[x][y][a_i]))
            t_partial_r_sum += rewards_function[x][y][a_i]
            #and not (contains_cords(oneway_cords, [x,y]) and not contains_cords(oneway_cords, [x+a[0],y+a[1]]))
            if (x + a[0] >= 0 and x + a[0] < len(board) and y + a[1] >= 0 and y + a[1] < len(board[0])) and not contains_cords(blocking_cords,[x + a[0],  y + a[1]]) and not contains_cords(terminal_states,[x + a[0],  y + a[1]]):
                x = x + a[0]
                y = y + a[1]
            states.append((x,y))
        # else:
        #     # print ("end state: " + str(((x,y))))
        #     traj.append(((x,y),[0,0]))

        step_n+=1

    return traj, t_partial_r_sum, values[x][y],is_terminal, x, y, states



def generate_MDP():
    dimensions_width = [1,3,6,10]
    dimensions_height = [3, 6, 10]
    # dimensions_width = [1]
    # dimensions_height = [3]

    # dimensions_width = [10]
    # dimensions_height = [10]
    height = random.choice(dimensions_height)
    width = random.choice(dimensions_width)

    env = GridWorldEnv(None, height, width)

    #sheep
    sheep_props = [0, 0.1, 0.3]
    sheep_prop = random.choice(sheep_props)
    n_sheep = int(sheep_prop*height*width)
    randomly_place_item_exact(env,3,n_sheep,height,width)


    #mildly bad
    mildly_bad_props = [0, 0.1, 0.5, 0.8]
    mildly_bad_prop = random.choice(mildly_bad_props)
    while (mildly_bad_prop + sheep_prop >= 1):
        mildly_bad_prop = random.choice(mildly_bad_props)
    n_mildly_bad = int(mildly_bad_prop*height*width)
    randomly_place_item_exact(env,5,n_mildly_bad,height,width)

    #mildly good
    mildly_good_props = [0, 0.1, 0.2]
    mildly_good_prop = random.choice(mildly_good_props)
    while (mildly_bad_prop + sheep_prop + mildly_good_prop >= 1):
        mildly_good_prop = random.choice(mildly_good_props)
    n_mildly_good = int(mildly_good_prop*height*width)
    randomly_place_item_exact(env,4,n_mildly_good,height,width)

    #goal
    x = random.randint(0,height-1)
    y = random.randint(0,width-1)
    env.board[x][y] = 1

    goal_rews = [0, 1, 5, 10, 50]
    sheep_rews = [-5, -10, -50]
    mildly_bad_rews = [-2, -5, -10]
    mildly_good_rews = [0]
    rew_vec = [-1,random.choice(goal_rews),random.choice(sheep_rews),random.choice(mildly_good_rews),random.choice(mildly_bad_rews),0]
    env.set_custom_reward_function(rew_vec,set_global=True)
    env.find_n_starts()

    env.get_blocking_cords()
    env.get_terminal_cords()
    V,Qs = value_iteration(rew_vec = np.array(rew_vec),GAMMA=0.999,env=env)
    #collect N randomly samples segment pairs
    all_X, all_r, all_ses, all_trajs, all_env_boards = get_env_trajs(env)

    return env, all_X, all_r, all_ses, all_trajs, all_env_boards



def get_env_trajs(env):
    V,Qs = value_iteration(rew_vec = np.array(env.reward_array),GAMMA=0.999,env=env)
    print ("finding start states...")
    start_states = []
    for i in range(len(env.board)):
        for j in range(len(env.board[0])):
            if not contains_cords(env.terminal_cords,(i,j)) and not contains_cords(env.blocking_cords,(i,j)):


                for i2 in range(len(env.board)):
                    for j2 in range(len(env.board[0])):
                        if not contains_cords(env.terminal_cords,(i2,j2)) and not contains_cords(env.blocking_cords,(i2,j2)):
                            start_states.append([(i,j), (i2,j2)])
            # else:
            #     print (contains_cords(env.terminal_cords,(i,j)))
            #     print (contains_cords(env.blocking_cords,(i,j)))
            #     print ((i,j))

    # print (start_states)
    # print ("*************************************")
    all_action_seqs = list(combinations(itertools.product(env.actions,env.actions,env.actions),2))



    all_collected_trajs = []
    n_pairs_found = 0
    max_pairs = 10000-1

    all_X = []
    all_r = []
    all_ses = []
    all_trajs = []
    all_env_boards = []

    seen_traj_pairs = set()

    for start_state in start_states:
        # print ("======")
        print (start_state)
        # print (len(all_action_seqs))
        for action_seqs in all_action_seqs:
            # print (start_state)
            action_seqs = list(action_seqs)

            ss1 = start_state[0]
            t1_s0_x,t1_s0_y = ss1
            action_seq_1 = action_seqs[0]
            traj1, t1_partial_r_sum, v_t1,is_terminal1,traj1_ts_x,traj1_ts_y, states1= create_traj(t1_s0_x, t1_s0_y,action_seq_1,3, env.board, env.reward_function, env.terminal_cords,env.blocking_cords,V)
            if t1_partial_r_sum == False:
                # assert False
                # print ("here")
                continue

            ss2 = start_state[1]
            t2_s0_x,t2_s0_y = ss2
            action_seq_2 = action_seqs[1]
            traj2, t2_partial_r_sum, v_t2,is_terminal2,traj2_ts_x,traj2_ts_y, states2 = create_traj(t2_s0_x, t2_s0_y,action_seq_2,3, env.board, env.reward_function, env.terminal_cords,env.blocking_cords,V)
            if t2_partial_r_sum == False:
                # assert False
                # print ("here")
                continue

            if traj1 == traj2:
                continue

            phi1,_ = find_reward_features(traj1, env, len(traj1)-1)
            phi2,_ = find_reward_features(traj2, env, len(traj2)-1)


            # only keep unique trajectorries
            big_traj_pair_tuple = ((traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y), (traj2[0][0],traj2[0][1]),(traj2_ts_x,traj2_ts_y), tuple(phi1), tuple(phi2))
            if big_traj_pair_tuple not in seen_traj_pairs:
                seen_traj_pairs.add(big_traj_pair_tuple)
            else:
                continue

            v_dif1 = v_t1 - V[t1_s0_x][t1_s0_y]
            v_dif2 = v_t2 - V[t2_s0_x][t2_s0_y]
            partial_sum_dif = float(t2_partial_r_sum - t1_partial_r_sum)
            v_dif = float(v_dif2 - v_dif1)

            traj1_ses = [(traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y)]
            traj2_ses = [(traj2[0][0],traj2[0][1]),(traj2_ts_x,traj2_ts_y)]

            quad = [traj1, traj2, v_dif, partial_sum_dif,(is_terminal1,is_terminal2)]

            all_collected_trajs.append(quad)

            all_X.append([phi1, phi2])
            all_r.append([t1_partial_r_sum, t2_partial_r_sum])
            all_ses.append([traj1_ses,traj2_ses])
            all_trajs.append([traj1,traj2])
            all_env_boards.append(env.board)


            n_pairs_found+=1
    return all_X, all_r, all_ses, all_trajs, all_env_boards



# for trial in range(0,30):
#     print ("processing ",trial)
#     with open("random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
#         env = pickle.load(rf)
#
#     all_X, all_r, all_ses, all_trajs, all_env_boards = get_env_trajs(env)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_trajs.npy", all_trajs)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_new_ses.npy", all_ses)




# #
# env, all_X, all_r, all_ses,all_trajs, all_env_boards = generate_MDP()
# print ("=====================================")
# print (len(all_X))
# # print (env.board)
# # print ("\n")
# #
# for x in all_X:
#     print (x)
# assert False
# #
# gt_rew_vec = env.reward_array.copy()
# # GAMMA=0.999
# # succ_feats, pis = generate_all_policies(100,GAMMA,env,gt_rew_vec)
# np.save("random_MDPs/MDP_test1_all_trajs.npy", all_trajs)
# np.save("random_MDPs/MDP_test1_all_env_boards.npy", all_env_boards)
# # np.save("random_MDPs/MDP_test1_succ_feats.npy", succ_feats)
# np.save("random_MDPs/MDP_test1_gt_rew_vec.npy", gt_rew_vec)
# np.save("random_MDPs/MDP_test1_all_X.npy", all_X)
# np.save("random_MDPs/MDP_test1_all_r.npy", all_r)
# np.save("random_MDPs/MDP_test1_all_ses.npy", all_ses)
# with open(f'random_MDPs/MDP_test1_env.pickle', 'wb') as file:
#     pickle.dump(env, file)

# #
# GAMMA = 0.999
# for trial in range(47,48):
#     print ("generating MDP: " + str(trial))
#     env, all_X, all_r, all_ses,all_trajs, all_env_boards = generate_MDP()
#     print (len(all_X))
#     #
#     #
#     # print (env.board)
#     # for i,traj in enumerate(all_trajs):
#     #     print (traj)
#     #     print (all_X[i])
#     #     print("\n")
#     #
#     #
#
#     gt_rew_vec = env.reward_array.copy()
#
#     # succ_feats, pis = generate_all_policies(100,GAMMA,env,gt_rew_vec)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_trajs.npy", all_trajs)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_env_boards.npy", all_env_boards)
#     # np.save("random_MDPs/MDP_" + str(trial) + "succ_feats.npy", succ_feats)
#     np.save("random_MDPs/MDP_" + str(trial) + "gt_rew_vec.npy", gt_rew_vec)
#     # np.save("random_MDPs/MDP_" + str(trial) + "all_X.npy", all_X)
#     # np.save("random_MDPs/MDP_" + str(trial) + "all_r.npy", all_r)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_ses.npy", all_ses)
#     with open(f'random_MDPs/MDP_' + str(trial) + 'env.pickle', 'wb') as file:
#         pickle.dump(env, file)
