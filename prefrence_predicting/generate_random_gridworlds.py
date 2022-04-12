import numpy as np
from grid_world import GridWorldEnv
import random
from value_iteration import value_iteration, follow_policy, learn_successor_feature,get_gt_avg_return,build_pi,iterative_policy_evaluation


def randomly_place_item(env,id,N,height,width):
    #randomly place mud
    for i in range(random.randint(0,N)):
        x = random.randint(0,height-1)
        y = random.randint(0,width-1)
        if env.board[x][y] == 6:
            env.board[x][y] = id+6
        else:
            env.board[x][y] = id


def generate_grid_world():
    dimensions = list(range(5, 20))
    height = random.choice(dimensions)
    width = random.choice(dimensions)

    env = GridWorldEnv(None, height, width)

    #randomly place mud
    for i in range(random.randint(0,height*width)):
        x = random.randint(0,height-1)
        y = random.randint(0,width-1)
        env.board[x][y] = 6

    #house
    randomly_place_item(env,2,int(height*width/3),height,width)
    #sheap
    randomly_place_item(env,3,int(height*width/10),height,width)
    #coin
    randomly_place_item(env,4,int(height*width/3),height,width)
    #roadblock
    randomly_place_item(env,5,int(height*width/3),height,width)
    #goal
    x = random.randint(0,height-1)
    y = random.randint(0,width-1)
    if env.board[x][y] == 6:
        env.board[x][y] = 7
    else:
        env.board[x][y] = 1

    env.set_custom_reward_function([-1,50,-50,1,-1,-2],set_global=True)
    env.find_n_starts()
    return env

def eval_under_grid_worlds(rew_vec,GAMMA,N=100):

    n_near_opt = 0

    for grid_n in range(N):
        env = generate_grid_world()

        # print ("performing value iteration...")
        learned_V,Q = value_iteration(rew_vec = np.array(rew_vec),GAMMA=GAMMA,env=env)

        #reset enviroment reward function to ground truth
        env.set_custom_reward_function([-1,50,-50,1,-1,-2],set_global=True)
        assert np.array_equal(env.reward_array, [-1,50,-50,1,-1,-2])

        # print ("average return following learned policy: ")
        pi = build_pi(Q,env=env)
        V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA,env=env)
        learned_avg_return = np.sum(V_under_gt)/env.n_starts #number of possible start states
        # print (learned_avg_return)


        # print ("average return with ground truth reward function: ")
        V,Q = value_iteration(rew_vec = np.array([-1,50,-50,1,-1,-2]),GAMMA=GAMMA,env=env)
        pi = build_pi(Q,env=env)
        V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA,env=env)

        avg_return = np.sum(V_under_gt)/env.n_starts #number of possible start states
        # print (avg_return)
        if (learned_avg_return/avg_return) >= 0.9:
            n_near_opt += 1

    print ("% of instantiations where the learned reward functions max ent. policy achieve >= 90% of the g.t reward functions max ent policy: " + str(100*(n_near_opt/N)) + "%")
    print ("--------------------------------------------------------------------\n")


GAMMA = 0.999

rew_vecs = np.load("/Users/stephanehatgiskessell/Downloads/all_reward_vecs_pr_er.npy")
for i,rew_vec in enumerate(rew_vecs):
    print (rew_vec)
    eval_under_grid_worlds(rew_vec,GAMMA)
