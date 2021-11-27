from grid_world import GridWorldEnv
import random
import numpy as np
import cv2

def get_random_reward_vector():
    space = [-1,50,-50,1,-1,-2]
    vector = []
    for i in range(6):
        s = random.choice(space)
        space.remove(s)
        vector.append(s)
    return vector

def learn_successor_feature(Q,V,FGAMMA,rew_vec=None):

    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    if type(rew_vec) is np.ndarray:
        env.set_custom_reward_function(rew_vec)

    THETA = 0.001
    # initialize Q
    # Q = defaultdict(lambda: np.zeros(env.action_space))
    psi = [[np.zeros(env.feature_size) for i in range(int(np.sqrt(env.observation_space)))] for j in range (int(np.sqrt(env.observation_space)))]

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    n = 0

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn reward succesor features
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        #loop through every state
        delta = 0
        new_psi = psi.copy()
        for i in range (10):
            for j in range (10):
                if env.is_blocked(i,j):
                    continue
                psi_original = psi[i][j]
                # s_tab,N = env.state2tab(i,j)
                # print ("here")
                state_Qs = []
                state_psi = []
                total_occupancies = []
                for a_index in range(len(actions)):
                    next_state, reward, done, phi = env.get_next_state((i,j),a_index)
                    ni,nj = next_state

                    if not done:
                        psi_sas = phi + FGAMMA*psi[ni][nj]
                    else:
                        psi_sas = np.zeros(env.feature_size)
                    state_psi.append(psi_sas)

                #value iteration
                new_psi[i][j] = state_psi[np.argmax(Q[i][j])]
                delta = max(delta,np.sum(np.abs((psi_original-new_psi[i][j]))))
        psi = new_psi
        if delta < THETA:
            break
    #----------------------------------------------------------------------------------------------------------------------------------------
    #check succesor features
    #----------------------------------------------------------------------------------------------------------------------------------------
    # with open('psi.npy', 'wb') as f:
    #     np.save(f, psi)
    # test code
    # for i in range (10):
    #     for j in range (10):
    #         if env.is_blocked(i,j):
    #             continue
    #         psi_sas = psi[i][j]
    #         v_pred = np.dot(psi_sas,env.reward_array)
    #         print (psi_sas)
    #         print (v_pred)
    #         print (V[i][j])
    #         print ((i,j))
    #         print (env.reward_array)
    #         print ("\n")
    #         assert (np.round(v_pred,3) == np.round(V[i][j],3))
    return psi

def learn_successor_feature_iter(pi,FGAMMA,rew_vec=None):

    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    if type(rew_vec) is np.ndarray:
        env.set_custom_reward_function(rew_vec)

    THETA = 0.001
    # initialize Q
    # Q = defaultdict(lambda: np.zeros(env.action_space))
    psi = [[np.zeros(env.feature_size) for i in range(int(np.sqrt(env.observation_space)))] for j in range (int(np.sqrt(env.observation_space)))]
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        delta = 0
        # new_psi = psi.copy()
        new_psi = np.copy(psi)

        for i in range (10):
            for j in range (10):
                if env.is_blocked(i,j):
                    continue
                # total = 0

                state_psi = []
                for trans in pi[(i,j)]:
                    prob, a_index = trans
                    next_state, reward, done, phi = env.get_next_state((i,j),a_index)
                    ni,nj = next_state
                    if not done:
                        psi_sas = prob*(phi + FGAMMA*psi[ni][nj])
                    else:
                        psi_sas = np.zeros(env.feature_size)
                    state_psi.append(psi_sas)
                new_psi[i][j] = sum(state_psi)
                delta = max(delta,np.sum(np.abs(psi[i][j]-new_psi[i][j])))
                # print (np.sum(np.abs(psi[i][j]-new_psi[i][j])))

        psi = new_psi

        if delta < THETA:
            break
    return psi

def build_pi(Q):
    pi = {}
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    for i in range (10):
        for j in range (10):
            V = max(Q[i][j])
            V_count = Q[i][j].tolist().count(V)
            pi[(i,j)] = [(1/V_count if Q[i][j][a_index] == V else 0, a_index) for a_index in range(len(actions))]
    return pi


def iterative_policy_evaluation(pi,rew_vec=None, set_rand_rew = False, GAMMA=0.999):
    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    # rand_rew_vec = get_random_reward_vector()
    # rand_rew_vec = [-1, -50, 50, 1, -1, -2]
    if  type(rew_vec) is np.ndarray:
        env.set_custom_reward_function(rew_vec)
    elif set_rand_rew:
        rand_rew_vec = get_random_reward_vector()
        env.set_custom_reward_function(rew_vec)

    THETA = 0.001
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    V = np.zeros((int(np.sqrt(env.observation_space)), int(np.sqrt(env.observation_space))))

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        delta = 0
        new_V = V.copy()
        for i in range (10):
            for j in range (10):
                if env.is_blocked(i,j):
                    continue
                # total = 0
                state_qs = []
                for trans in pi[(i,j)]:
                    prob, a_index = trans
                    next_state, reward, done, _ = env.get_next_state((i,j),a_index)
                    ni,nj = next_state
                    if not done:
                        state_qs.append(prob*(reward + GAMMA*V[ni][nj]))
                    else:
                        state_qs.append(prob*reward)
                new_V[i][j] = sum(state_qs)
                delta = max(delta,np.abs(V[i][j]-new_V[i][j]))

        V = new_V
        if delta < THETA:

            break

    return V

# def policy_evaluation(pi,rew_vec,set_rand_rew,GAMMA):
#     env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
#     # rand_rew_vec = get_random_reward_vector()
#     # rand_rew_vec = [-1, -50, 50, 1, -1, -2]
#     if  type(rew_vec) is np.ndarray:
#         env.set_custom_reward_function(rew_vec)
#     elif set_rand_rew:
#         rand_rew_vec = get_random_reward_vector()
#         env.set_custom_reward_function(rew_vec)
#
#     THETA = 0.001
#     actions = [[-1,0],[1,0],[0,-1],[0,1]]
#     V = np.zeros((int(np.sqrt(env.observation_space)), int(np.sqrt(env.observation_space))))
#
#     #----------------------------------------------------------------------------------------------------------------------------------------
#     #iterativley learn state value
#     #----------------------------------------------------------------------------------------------------------------------------------------
#     while True:
#         delta = 0
#         new_V = V.copy()
#         for i in range (10):
#             for j in range (10):
#                 if env.is_blocked(i,j):
#                     continue
#                 total = 0
#                 for trans in pi[(i,j)]:
#                     prob, a_index = trans
#                     next_state, reward, done, _ = env.get_next_state((i,j),a_index)
#                     ni,nj = next_state
#                     if not done:
#                         total = prob*(reward + GAMMA*V[i][j])
#                     else:
#                         total = prob*reward
#                 new_V[i][j] = total
#                 delta = max(delta,np.abs(V[i][j]-new_V[i][j]))
#
#         V = new_V
#         if delta < THETA:
#             break
#
#     return V


def policy_improvement(rew_vec=None, set_rand_rew = False, GAMMA=0.999):
    # env = GridworldEnv()
    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    # rand_rew_vec = get_random_reward_vector()
    # rand_rew_vec = [-1, -50, 50, 1, -1, -2]
    if  type(rew_vec) is np.ndarray:
        env.set_custom_reward_function(rew_vec)
    elif set_rand_rew:
        rand_rew_vec = get_random_reward_vector()
        env.set_custom_reward_function(rew_vec)

    # V = np.zeros((int(np.sqrt(env.observation_space)), int(np.sqrt(env.observation_space))))
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    policy_stable = True
    first_iter_done = False
    pi = {}

    for i in range (10):
        for j in range (10):
            pi[(i,j)] = [(0.25,0), (0.25,1), (0.25,2), (0.25,3)]

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while not policy_stable or not first_iter_done:
        first_iter_done = True
        # print (pi)
        print ("here\n")
        V = iterative_policy_evaluation(pi,rew_vec,set_rand_rew,GAMMA)
        print (V)
        # assert False

        for i in range (10):
            for j in range (10):
                if env.is_blocked(i,j):
                    continue
                temp = pi.get((i,j))
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
                # state_Qs = np.array(state_Qs)
                # pi[(i,j)] = np.random.choice(np.flatnonzero(state_Qs == state_Qs.max()))
                largest_q = np.max(state_Qs)
                n_ties = state_Qs.count(largest_q)
                pi[(i,j)] = [(1/n_ties if state_Qs[p_a_index] == largest_q else 0, p_a_index) for p_a_index in range(len(actions))]
                # for p_a_index in len(actions):
                #     updated.append()

                # pi[(i,j)] = np.argmax(state_Qs)

                if pi[(i,j)] != temp:
                    # print (temp)
                    # print (pi[(i,j)])
                    # print ("\n")
                    policy_stable = False
        # print ("\n\n")
        # print ("=================================================================================\n")

    print (np.round(V,1))
    print ("=================================================================================\n")
    return V,pi



def value_iteration(rew_vec=None, set_rand_rew = False, GAMMA=0.999):
    # env = GridworldEnv()
    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    # rand_rew_vec = get_random_reward_vector()
    # rand_rew_vec = [-1, -50, 50, 1, -1, -2]
    if  type(rew_vec) is np.ndarray:
        env.set_custom_reward_function(rew_vec)
    elif set_rand_rew:
        rand_rew_vec = get_random_reward_vector()
        env.set_custom_reward_function(rew_vec)


    THETA = 0.001
    # initialize Q
    # Q = defaultdict(lambda: np.zeros(env.action_space))
    V = np.zeros((int(np.sqrt(env.observation_space)), int(np.sqrt(env.observation_space))))
    Qs = [[np.zeros(4) for i in range(int(np.sqrt(env.observation_space)))] for j in range (int(np.sqrt(env.observation_space)))]

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    n = 0

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        delta = 0
        new_V = V.copy()
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
                    Qs[i][j][a_index] = Q

                new_V[i][j] = max(Qs[i][j])
                delta = max(delta,np.abs(v-new_V[i][j]))
        V = new_V
        if delta < THETA:
            break
    # print (np.round(V,1))
    # print ("=================================================================================\n")
    return V,Qs


def ground_truth_follow_policy():
    V,Q = value_iteration()
    print ("ground truth value function:")
    print (V)
    print ("=================================================================================\n")
    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    epsiode_returns = []
    #try out every possile start state
    for i in range (10):
        for j in range (10):
            if env.is_blocked(i,j) or env.is_terminal(i,j):
                continue
            x = i
            y = j
            done = False
            n_steps = 0
            epsiode_return = 0
            while not done:
                a_index = np.argmax(Q[x][y])
                next_state, reward, done, _ = env.get_next_state((x,y),a_index)
                x,y = next_state
                n_steps +=1
                epsiode_return+=reward
            epsiode_returns.append(epsiode_return)
    avg_return = np.mean(epsiode_returns)
    return avg_return


def get_gt_avg_return(GAMMA):
    V,Q = value_iteration()
    pi = build_pi(Q)
    V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
    gt_avg_return = np.sum(V_under_gt/92)
    print ("average return following ground truth policy: ")
    print (gt_avg_return)
    print ("=================================================================================\n")

def follow_policy(Q, max_steps,viz_policy=False):
    env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    epsiode_returns = []
    #try out every possile start state

    if viz_policy:
        img = cv2.imread("/Users/stephanehatgiskessell/Desktop/board.png")
        #display action matrix
        for i in range (10):
            for j in range (10):
                # print (env.is_blocked(3,4))
                if env.is_blocked(i,j) or env.is_terminal(i,j):
                    continue

                optimal_actions = []
                best_q = float("-inf")
                for q_index in range(len(Q[i][j])):
                    q = Q[i][j][q_index]
                    if q > best_q:
                        optimal_actions = [q_index]
                        best_q = q
                    elif q == best_q:
                        optimal_actions.append(q_index)

                for a_index in optimal_actions:
                    # a_index = np.argmax(Q[i][j])

                    next_state, reward, done, _ = env.get_next_state((i,j),a_index)
                    # if (i == 2 and j == 4):
                        # print (a_index)
                        # print (next_state)
                    i_p, j_p = next_state
                    cv2.circle(img, ((j*110)+45,(i*110)+45), 10, (255,0,0), -1)
                    if j_p - j == 0: #vertical
                        cv2.arrowedLine(img,((j*110)+45,(i*110)+45), ((j_p*110)+45,(i_p*110)+45-20),(255,0,0), 5, 8, 0, 0.2)
                    elif i_p - i == 0: #horizontal
                        cv2.arrowedLine(img,((j*110)+45,(i*110)+45), ((j_p*110)+45-20,(i_p*110)+45),(255,0,0), 5, 8, 0, 0.2)


        cv2.imshow("path",img)
        cv2.waitKey(0)

    for i in range (10):
        for j in range (10):
            if env.is_blocked(i,j) or env.is_terminal(i,j):
                continue
            if viz_policy:
                img = cv2.imread("/Users/stephanehatgiskessell/Desktop/board.png")

            x = i
            y = j
            done = False
            n_steps = 0
            epsiode_return = 0
            # print ("\n\n")
            # print (Q[i][j])
            while not done and n_steps < max_steps:
                if viz_policy:
                    cv2.circle(img,((y*130)+20,(x*130)+20), 12, (255,0,0), -1)

                #a_index = np.argmax(Q[x][y])
                a_index = np.random.choice(np.flatnonzero(Q[x][y] == Q[x][y].max()))

                next_state, reward, done, _ = env.get_next_state((x,y),a_index)
                # print ((x,y))
                # print (Q[x][y])
                # print (reward)
                # print ("\n")

                x,y = next_state
                n_steps +=1
                epsiode_return+=reward
            # print(epsiode_return)

            # if viz_policy:
            #     cv2.imshow("path",img)
            #     cv2.waitKey(0)
            epsiode_returns.append(epsiode_return)
    avg_return = np.mean(epsiode_returns)
    gt_avg_return = ground_truth_follow_policy()

    print ("average return following learned policy: ")
    print (avg_return)

    print ("average return following ground truth policy: ")
    print (gt_avg_return)
    print ("=================================================================================\n")
    return avg_return
