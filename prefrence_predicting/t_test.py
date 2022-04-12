import numpy as np
import itertools
import scipy.stats

er_er_det_ret = np.load("random_MDPs/avg_return_deterministic_er_er.npy")
er_er_prob_ret =np.load("random_MDPs/avg_return_sigmoid_er_er.npy")
pr_er_det_ret = np.load("random_MDPs/avg_return_deterministic_pr_er.npy")
pr_er_prob_ret = np.load("random_MDPs/avg_return_sigmoid_pr_er.npy")
pr_pr_det_ret = np.load("random_MDPs/avg_return_deterministic_pr_pr.npy")
pr_pr_prob_ret = np.load("random_MDPs/avg_return_sigmoid_pr_pr.npy")
er_pr_det_ret = np.load("random_MDPs/avg_return_deterministic_er_pr.npy")
er_pr_prob_ret = np.load("random_MDPs/avg_return_sigmoid_er_pr.npy")

name2dataset = {"change in expected return generated prefs/change in expected return reward learning model, deterministic data": er_er_det_ret, "change in expected return generated prefs/change in expected return reward learning model, probabilistic data": er_er_prob_ret,
"partial return generated prefs/change in expected return reward learning model, deterministic data": pr_er_det_ret, "partial return generated prefs/change in expected return reward learning model, probabilistic data": pr_er_prob_ret,
"partial return generated prefs/partial return reward learning model, deterministic data": pr_pr_det_ret, "partial return generated prefs/partial return reward learning model, probabilistic data": pr_pr_prob_ret,
"change in expected return generated prefs/partial return reward learning model, deterministic data": er_pr_det_ret, "change in expected return generated prefs/partial return reward learning model, probabilistic data": er_pr_prob_ret}
# 
# for key in name2dataset.keys():
#     print (key + ":")
#
combos = list(map(dict, itertools.combinations(name2dataset.items(), 2)))

for combo in combos:
    keys = list(combo.keys())
    if ("deterministic" in keys[0] and "probabilistic" in keys[1]) or ("deterministic" in keys[1] and "probabilistic" in keys[0]):
        continue
    print ("==================================================================================== \nT-test between " + str(keys[0]) + "\n and \n" + str(keys[1]) + " \n====================================================================================")
    res1 = combo[keys[0]]
    res2 = combo[keys[1]]
    t_stat, p_val = scipy.stats.ttest_rel(res1, res2)
    print ("T-staistic: " + str(t_stat))
    print ("P-value: " + str(p_val))
    print ("\n\n")
