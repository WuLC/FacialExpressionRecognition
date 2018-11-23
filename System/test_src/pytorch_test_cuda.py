# CUDA TEST
import torch
x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)

# CUDNN TEST
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))