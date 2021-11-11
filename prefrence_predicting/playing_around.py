import torch

a = [999.5009,   0.0000,   0.0000,   0.0000, 499.5002,   0.0000]
a = torch.tensor(a).Double()
print (a.dtype)
