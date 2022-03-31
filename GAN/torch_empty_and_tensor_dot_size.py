import torch

t = torch.ones(3, 4, 5)  # returns a tensor filled with uninitialized (random and not to be used) data
print(t)
print(torch.tensor(3) * t + torch.tensor(1))
print(t.size())
print(t.size(0))
print(t.size(1))
print(t.size(2))
