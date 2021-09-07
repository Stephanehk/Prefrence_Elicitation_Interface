import gym
import numpy as np
from collections import defaultdict
# import plotting
from grid_world import GridWorldEnv
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def value_iteration():

    # env = GridworldEnv()
    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    GAMMA = 1
    FGAMMA = 1
    THETA = 0.001
    # initialize Q
    # Q = defaultdict(lambda: np.zeros(env.action_space))
    V = np.zeros((int(np.sqrt(env.observation_space)), int(np.sqrt(env.observation_space))))
    M = [[np.zeros(env.feature_size) for i in range(int(np.sqrt(env.observation_space)))] for j in range (int(np.sqrt(env.observation_space)))]

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    n = 0

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        #loop through every state
        delta = 0
        for i in range (10):
            for j in range (10):
                if env.is_blocked(i,j):
                    continue
                v = V[i][j]
                state_Qs = []
                for a_index in range(len(actions)):
                    next_state, reward, done, _ = env.get_next_state((i,j),a_index)
                    ni,nj = next_state
                    if not done:
                        Q = reward + GAMMA*V[ni][nj]
                    else:
                        Q = reward
                    state_Qs.append(Q)
                V[i][j] = max(state_Qs)
                delta = max(delta,np.abs(v-V[i][j]))
        if delta < THETA:
            break


    # ----------------------------------------------------------------------------------------------------------------------------------------
    # iterativley learn reward succesor features
    # ----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        #loop through every state
        delta = 0
        for i in range (10):
            for j in range (10):
                if env.is_blocked(i,j):
                    continue
                m_original = M[i][j]
                # s_tab,N = env.state2tab(i,j)
                # print ("here")
                state_Qs = []
                state_Ms = []
                total_occupancies = []
                for a_index in range(len(actions)):
                    next_state, reward, done, reward_feature = env.get_next_state((i,j),a_index)
                    # print (reward)
                    ni,nj = next_state

                    if not done:
                        Q = reward + GAMMA*V[ni][nj]
                        m = reward_feature + FGAMMA*M[ni][nj]
                    else:
                        # print (reward_feature)
                        Q = reward
                        # assert (sum(reward_feature) == 1)
                        # m = reward_feature
                        m = np.zeros(env.feature_size)
                    state_Qs.append(Q)
                    state_Ms.append(m)

                #value iteration
                M[i][j] = state_Ms[np.argmax(state_Qs)]
                #policy evaluation
                # M[i][j] = state_Ms[np.argmax(np.sum(state_Ms,axis=1))]
                delta = max(delta,np.sum(np.abs((m_original-M[i][j]))))

        if delta < THETA:
            break

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state succesor features
    #----------------------------------------------------------------------------------------------------------------------------------------
    # while True:
    #     #loop through every state
    #     delta = 0
    #     for i in range (10):
    #         for j in range (10):
    #             if env.is_blocked(i,j):
    #                 continue
    #             m_original = M[i][j]
    #             s_tab,N = env.state2tab(i,j)
    #             # print ("here")
    #             state_Qs = []
    #             state_Ms = []
    #             total_occupancies = []
    #             for a_index in range(len(actions)):
    #                 next_state, reward, done, _ = env.get_next_state((i,j),a_index)
    #                 # print (reward)
    #                 ni,nj = next_state
    #
    #                 if not done:
    #                     Q = reward + GAMMA*V[ni][nj]
    #                     m = s_tab + FGAMMA*M[ni][nj]
    #                 else:
    #                     Q = reward
    #                     m = s_tab
    #                 state_Qs.append(Q)
    #                 state_Ms.append(m)
    #
    #             #value iteration
    #             M[i][j] = state_Ms[np.argmax(state_Qs)]
    #             #policy evaluation
    #             # M[i][j] = state_Ms[np.argmax(np.sum(state_Ms,axis=1))]
    #             delta = max(delta,np.sum(np.abs((m_original-M[i][j]))))
    #
    #     if delta < THETA:
    #         break
    #----------------------------------------------------------------------------------------------------------------------------------------
    #check succesor features
    #----------------------------------------------------------------------------------------------------------------------------------------

    #subtract inital state reward
    # for i in range (10):
    #     for j in range (10):
    #         state_Qs = []
    #         for a_index in range(len(actions)):
    #             next_state, reward, done, reward_feature = env.get_next_state((i,j),a_index)
    #             if not done:
    #                 Q = reward + GAMMA*V[ni][nj]
    #             else:
    #                 Q = reward
    #             state_Qs.append(Q)
    #
    #         next_state, reward, done, reward_feature = env.get_next_state((i,j),np.argmax(state_Qs))
    #         M[i][j] -= reward_feature

    #test code
    for i in range (10):
        for j in range (10):
            if env.is_blocked(i,j):
                continue
            m = M[i][j]
            v_pred = np.dot(m,env.reward_array)
            # print ((i,j))
            # print (m)
            # print (v_pred)
            # print (V[i][j])
            # print ("\n")
            assert (v_pred == V[i][j])
    # print (M[8][9])
# Q = np.load("/Users/stephanehatgiskessell/Desktop/Kivy_stuff/boards/2021-07-29_sparseboard2-notrap_Qs.npy")
value_iteration()
