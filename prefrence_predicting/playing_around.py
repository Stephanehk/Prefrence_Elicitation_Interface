import torch
import numpy as np
from value_iteration import value_iteration, follow_policy, learn_successor_feature,get_gt_avg_return,build_pi,iterative_policy_evaluation

arr = [ -2.28238783, -20.06451856, -67.45879582,   2.04663161,  -2.18512413,  -4.42418019]
arr = np.array(arr)
print (arr/abs(arr[0]))


# for i in range(10):
#     for j in range(10):
#         gt_v = np.dot(gt_succ_feats[0][i][j],[-1,50,-50,1,-1,-2])
#
#         vs = []
#         for n in range(len(succ_feats)):
#             v_pred = np.dot(succ_feats[n][i][j],[-1,50,-50,1,-1,-2])
#             vs.append(v_pred)
#
#         vs = torch.tensor(vs)
#         v_approx = torch.sum(torch.mul(softmax(vs),vs)).detach().numpy()
#
#         print (v_approx)
#         print (gt_v)
#         print ("\n")



# [0.44855193 0.55144807]
# [0.7432258527564061, 0.25677414724359393]
# [1, 0]

# succ_feats = np.load("succ_feats.npy")
# succ_feats = torch.tensor(succ_feats)
# print (succ_feats[0][0][0])
#

# class arr:
#     def __init__(self, arr):
#         self.arr = arr;
# a = [[[1,2,3],[1,2,3],[1,2,3]], [[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]]
# # # a = [[1,2,3],[4,5,6],[7,8,9]]
# a = [[[[1,1,1,0,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]],[[4,4,4,4,4,4],[5,5,0,5,5,5],[6,6,6,6,6,6]],[[7,7,7,0,7,7],[8,8,0,8,8,8],[9,9,9,9,9,9]]], [[[0,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]],[[1,1,1,0,1,1],[8,8,0,8,8,8],[3,3,3,3,3,3]],[[1,0,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]]],[[[1,1,1,1,1,0],[2,2,2,2,2,2],[3,3,3,3,3,3]],[[1,1,1,1,0,1],[8,8,8,8,8,8],[3,3,3,3,3,3]],[[1,1,1,0,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]]]]
# # a = [[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[1,8,3],[1,2,3]]]
#
# a = torch.tensor(a)
# print (a.shape)
# xs = torch.tensor([2,1])
# ys = torch.tensor([0,1])
# helper_ind = torch.tensor([0,1])
# # xs = torch.tensor([[0,1],[1,2]])
# # ys = torch.tensor([0,1,2])
#
# #given xs = [[x11,x12],[x21,x22],...], ys = [[y11,y12],[y21,y22],...] returns [[M[x11][y11],[M[x12][y12]], ...]
#
#
# #given xs = [x1,x2,...], ys = [y1,y2,...] returns [[M1[x1][y1], M1[x2][y2],etc],[M2[x1][y1], M2[x2][y2],etc]]
# rows = torch.index_select(a,1,xs)
# cols = torch.index_select(rows,2,ys)
#
# print (rows)

# print ("\n")
# print (torch.index_select(cols,0,ys))
# c1 = torch.stack([rows[0][0][0],rows[0][1][1]])
# c2 = torch.stack([rows[1][0][0],rows[1][1][1]])
# c2 = torch.stack([rows[2][0][0],rows[2][1][1]])
#                      .
#                      .
#                      .

# print (torch.stack([c1,c2]))
# print (rows[0][0][0])
# print (rows[0][1][1])
#
# print (rows[1][0][0])
# print (rows[1][1][1])

# ys = torch.unsqueeze(torch.unsqueeze(torch.stack([ys,ys]),2),2)

# ys = torch.stack([ys[0].repeat(1,6),ys[1].repeat(1,6)])
# ys = torch.unsqueeze(torch.squeeze(ys),2)
# ys = ys.repeat(len(a),1,1,1)
# ys = torch.tensor ([[[[0],[0],[0],[0],[0],[0]],[[1],[1],[1],[1],[1],[1]]],[[[0],[0],[0],[0],[0],[0]],[[1],[1],[1],[1],[1],[1]]],[[[0],[0],[0],[0],[0],[0]],[[1],[1],[1],[1],[1],[1]]]])
# ys = torch.tensor ([[[0,0,0,0,0,0],[1,1,1,1,1,1]],[[0,0,0,0,0,0],[1,1,1,1,1,1]],[[0,0,0,0,0,0],[1,1,1,1,1,1]]])

# rows = torch.index_select(a,1,xs)
# #
# # cols = torch.gather(rows,1,ys)
# # cols = torch.squeeze(cols)
# cols = torch.index_select(rows,2,ys)
# print (rows)
# print ("\n")
# print (cols)
# print ("\n")
#
# print (cols.shape)
# cols = torch.index_select(cols,2,helper_ind)
# print (cols)
# # max_ = torch.max(cols,dim=1)[0]
#



#
# #given xs = [x1,x2,...], ys = [y1,y2,...] returns [M[x1][y1], M[x2][y2],etc]
# # ys = torch.unsqueeze(ys,0)
# # rows = torch.index_select(a,0,xs)
# # cols = torch.gather(rows,0,ys)
# # cols = torch.squeeze(cols)
# #
# #
# # print (rows)
# # print (cols)
