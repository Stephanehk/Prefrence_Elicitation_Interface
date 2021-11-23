import numpy as np
import torch

softmax = torch.nn.Softmax(dim=0)

#toy coordinates
cords = [[0,1], [1,2], [0,2]]
#toy successor feature
succ_feats = [[[1,2,3],[1,2,3],[1,2,3]], [[4,5,3],[1,2,8],[1,0,3]],[[1,-1,3],[1,2,-3],[1,2,3]]]

cords = torch.tensor(cords)
succ_feats = torch.tensor(succ_feats)

v_pi_approx = []
for c in cords:
    ss_vs = []
    x = c[0]
    y = c[1]
    #find value at given x,y pair under each successor feature
    for i in range(len(succ_feats)):

        succ_phi = succ_feats[i][x][y]
        succ_phi = succ_phi.double()
        v = torch.mul(succ_phi, 1)  #this would be our linear layer but for examples sake I have removed it
        ss_vs.append(v)

    #find the maximum value for given x,y pair
    ss_vs = torch.tensor(ss_vs)
    max_ = torch.sum(torch.mul(softmax(ss_vs),ss_vs))
    v_pi_approx.append(max_)


v_pi_approx =  torch.stack(v_pi_approx)

print (v_pi_approx)
