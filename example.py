import matplotlib.pyplot as plt
import torch
from pytorch_dmpling.dmp import *

# Use GPU/CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Weights/Trajectories batch size
batch_size = 32

# Weights parameters
weights = torch.randn(batch_size * 2,10) * 100
weights.requires_grad_(True)

# DMP trajectories, using weights parameters
dmp           = DMP(10,0.01,a=10,b=10/4,w=weights,device=device)
trajectories  = dmp.run_sequence()

# Trajectories are differentiable with respect to weights
loss = trajectories.norm()
loss.backward()

print(weights.grad)
